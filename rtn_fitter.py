# Using bias stack, locate pixels with RTN and fit their parameters.
# Chris Layden

import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.stats import norm
import copy
import time
from numba import njit
from scripts.make_lut import ThresholdLUT

def sigma_clip(data, sigma=10, max_iters=5, clip_val='mean'):
    # Sigma-clip data along axis 0 (time axis) to remove outliers.
    # Set clipped points to np.nan
    clipped_data = copy.deepcopy(data)
    std = np.inf
    for _ in range(max_iters):
        mean = np.nanmean(clipped_data, axis=0)
        std2 = np.nanstd(clipped_data, axis=0, ddof=1)
        if np.all(std2 == std):
            break
        std = std2
        diff = np.abs(clipped_data - mean)
        mask = diff < (sigma * std)
        if np.all(mask):
            break
        if clip_val == 'mean':
            # set outliers to mean
            clipped_data = np.where(mask, clipped_data, mean)
        else:
            clipped_data = np.where(mask, clipped_data, np.nan)
    return clipped_data

def get_read_noise_stats(data, adu_unit, plot=False):
    """
    Calculate read noise statistics from a stack of bias images.

    Parameters:
    -----------
    data : ndarray
        3D array of bias images (num_images, height, width).
    adu_unit : astropy.units.Unit
        Unit representing ADU (Analog-to-Digital Units).
    plot : bool, optional
        If True, plot histogram of read noise values. Default is False.
    """
    if len(data.shape) != 3:
        raise ValueError("Input data must be a 3D array of images.")
    if data.shape[0] < 10:
        raise ValueError("At least ten bias images required to compute per-pixel read noise.")
    # data initially has units of ADU
    data = data * adu_unit
    read_var_array = np.nanvar(data, axis=0, ddof=1)
    read_noise_array = np.sqrt(read_var_array)
    mean_read_noise = np.nanmean(read_noise_array)
    median_read_noise = np.nanmedian(read_noise_array)
    rms_read_noise = np.sqrt(np.nanmean(read_var_array))
    read_noise_sigma = np.nanstd(read_noise_array, ddof=1)
    read_noise_stats = {"mean_read_noise": mean_read_noise,
                        "median_read_noise": median_read_noise,
                        "rms_read_noise": rms_read_noise,
                        "read_noise_sigma": read_noise_sigma}
    if plot:
        plt.rcParams.update({'font.size': 14})
        plt.hist(read_noise_array.flatten(), bins=200, histtype='step', alpha=0.7)
        plt.xlabel("Read Noise (e-)" if adu_unit is not None else "Read Noise (ADU)")
        plt.ylabel("Number of Pixels")
        # Set x upper limit to 10. Leave lower limit unbound
        # plt.xlim(right=10)
        plt.yscale('log')
        plt.axvline(mean_read_noise.value, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_read_noise:.2f}")
        plt.axvline(median_read_noise.value, color='blue', linestyle='dashed', linewidth=1, label=f"Median: {median_read_noise:.2f}")
        plt.axvline(rms_read_noise.value, color='green', linestyle='dashed', linewidth=1, label=f"RMS: {rms_read_noise:.2f}")
        plt.legend()
        plt.show()
    return read_noise_stats, read_noise_array

def smooth_data(data):
    # Follow Ozdogru et al. to un-discretize data, making it suitable for testing and fitting.
    smoothing_arr = np.random.uniform(-0.5, 0.5, size=data.shape)
    smoothed_data = data + smoothing_arr
    return smoothed_data

def ad_statistics_normal(data):
    # Compute Anderson-Darling statistic for normality test for all pixels.
    # Change things to cupy/other GPU libraries later for speed.
    n, nx, ny = data.shape
    # Sigma clip and smooth data
    clipped_data = sigma_clip(data, sigma=5, max_iters=3)
    smoothed_data = smooth_data(clipped_data)
    sorted_data = np.sort(smoothed_data, axis=0)
    # Normalize to z-scores at each pixel
    mean = np.mean(sorted_data, axis=0, keepdims=True)
    std = np.std(sorted_data, axis=0, ddof=1, keepdims=True)
    z = (sorted_data - mean) / std
    # Compute normal CDF
    cdf = norm.cdf(z)
    # Clip CDF values to avoid log(0)
    eps = np.finfo(float).eps  # ~2.2e-16
    cdf = np.clip(cdf, eps, 1 - eps)
    # Compute A-squared statistic
    i = np.arange(1, n + 1).reshape(-1, 1, 1)
    term = (2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1, :, :]))
    A2 = -n - np.mean(term, axis=0)  # shape: (nx, ny)
    return A2

def identify_nonnormal_pixels(data, adu_unit, threshold=1.092, plot=False):
    # Identify pixels with non-normal distributions based on AD statistic.
    # Threshold of 1.092 corresponds to 1% significance level for normality test
    # with infinite samples and unknown mean and variance (see https://doi.org/10.2307/2286009)
    A2 = ad_statistics_normal(data)
    # read_noise_array = np.sqrt(np.var(data, axis=0, ddof=1))
    read_noise_stats, read_noise_array = get_read_noise_stats(data, adu_unit, plot=False)
    # Don't try to correct pixels with read noise already close to the median
    low_cutoff = read_noise_stats["median_read_noise"]
    nonnormal_mask = (A2 > threshold) & (read_noise_array > low_cutoff)
    # Plot read noise histogram of non-normal pixels
    if plot:
        read_noise_array = read_noise_array.to(u.electron).value
        nonnormal_read_noise = read_noise_array[nonnormal_mask]
        plt.hist(read_noise_array.flatten(), bins=200, range=(0, np.max(read_noise_array)), histtype='step', alpha=0.3, label='All Pixels')
        plt.hist(nonnormal_read_noise.flatten(), bins=200, range=(0, np.max(read_noise_array)), histtype='step', alpha=0.7, label='Pixels failing AD test')
        plt.xlabel("Read Noise (e-)")
        plt.ylabel("Number of Non-normal Pixels")
        plt.yscale('log')
        plt.title("Read Noise Distribution of Non-normal Pixels")
        plt.legend()
        plt.show()
    return nonnormal_mask

@njit
def rtn_triple_gaussian(x, mu, A, B1, B2, d, sigma):
    s2 = sigma ** 2
    z_c = x - mu
    z_l = x + d - mu
    z_r = x - d - mu
    return (A * np.exp(-z_c**2 / (2 * s2)) + 
            B1 * np.exp(-z_l**2 / (2 * s2)) + 
            B2 * np.exp(-z_r**2 / (2 * s2)))

@njit
def rtn_triple_gaussian_jac(x, mu, A, B1, B2, d, sigma):
    s2 = sigma ** 2
    s3 = sigma ** 3
    
    z_c = x - mu
    z_l = x + d - mu
    z_r = x - d - mu
    
    exp_c = np.exp(-z_c**2 / (2 * s2))
    exp_l = np.exp(-z_l**2 / (2 * s2))
    exp_r = np.exp(-z_r**2 / (2 * s2))
    
    n = len(x)
    jac = np.empty((n, 6))
    jac[:, 0] = (A * exp_c * z_c + B1 * exp_l * z_l + B2 * exp_r * z_r) / s2
    jac[:, 1] = exp_c
    jac[:, 2] = exp_l
    jac[:, 3] = exp_r
    jac[:, 4] = (-B1 * exp_l * z_l + B2 * exp_r * z_r) / s2
    jac[:, 5] = (A * exp_c * z_c**2 + B1 * exp_l * z_l**2 + B2 * exp_r * z_r**2) / s3
    return jac

def fit_rtn_parameters(data, nonnormal_mask, read_noise_stats, adu_unit, min_spacing=3, verbose=False):
    """ Identify pixels that have correctable RTN and return their parameters.

    Parameters:
    -----------
    data : ndarray
        3D array of bias images (num_images, height, width).
    nonnormal_mask : ndarray
        2D boolean array indicating pixels with non-normal distributions.
    read_noise_stats : dict
        Dictionary containing read noise statistics (mean, median, rms).
    adu_unit : astropy.units.Unit
        Unit representing ADU (Analog-to-Digital Units).
    min_spacing : float, optional
        Minimum spacing between central and side peaks that can be corrected,
        in multiples of the noise at 1 e-/frame. Default is 3.
    verbose : bool, optional
        If True, print fit diagnostics. Default is False.
    
    Returns:
    --------
    rtn_params : dict
        Dictionary with keys as (x, y) pixel coordinates and values as fitted RTN parameters:
        (mu, A, B, d, sigma). A is the normalized central peak amplitude and B the left peak amplitude,
        such that the right peak amplitude is (1 - A - B).
    """
    rtn_mask = np.zeros(nonnormal_mask.shape, dtype=bool)
    # Store parameters in an array. Will be 6 x nx x ny: mu, A, B, d, sigma, lambda_max (added later)
    # non-RTN values will be np.nan
    rtn_params_array = np.full((6, data.shape[1], data.shape[2]), np.nan)
    poor_fit_count = 0
    too_close_count = 0
    fit_fail_count = 0
    n_images, nx, ny = data.shape
    # t0 = time.time()
    # hist_time = time.time()
    # fit_time = time.time()
    for ix in range(nx):
        if verbose:
            print(f"Fitting column {ix+1}/{nx}")
        for iy in range(ny):
            if not nonnormal_mask[ix, iy]:
                continue
            # if time.time() - t0 > 1e-6:
            #     print("Hist Time:", hist_time - t0, "Fit Time:", fit_time - hist_time)
            # t0 = time.time()
            pixel_data = data[:, ix, iy]
            # Create histogram
            bins_min = np.round(np.percentile(pixel_data, 0.5) - adu_unit.to(u.electron).value).astype(int)
            bins_max = np.round(np.percentile(pixel_data, 99.5) + adu_unit.to(u.electron).value).astype(int)
            bin_size = np.round(np.max([1, (bins_max - bins_min) / 30])).astype(int)
            bins = np.arange(bins_min, bins_max + bin_size, bin_size)
            counts, bin_edges = np.histogram(pixel_data, bins=bins, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # hist_time = time.time()
            # Initial parameter guesses
            mu_guess = np.mean(pixel_data)
            sigma_guess = read_noise_stats['median_read_noise'].to(adu_unit).value
            A_guess = np.max(counts)
            B1_guess = 0.1 * A_guess
            B2_guess = 0.1 * A_guess
            d_guess = (bins_max - bins_min) / 2
            p0 = [mu_guess, A_guess, B1_guess, B2_guess, d_guess, sigma_guess]
            d_max = (bins_max - bins_min)
            bounds = ([0, A_guess / 2, 0, 0, 0, 0], [np.inf, np.inf, A_guess, A_guess, d_max, np.inf])
            try:
                popt, pcov = curve_fit(rtn_triple_gaussian, bin_centers, counts, p0=p0,
                                       maxfev=100, bounds=bounds, full_output=False,
                                       jac=rtn_triple_gaussian_jac)
                mu_fit = popt[0]
                A_fit = popt[1] / (popt[1] + popt[2] + popt[3])  # Normalize A
                B_fit = popt[2] / (popt[1] + popt[2] + popt[3])  # Normalize B1
                d_fit = popt[4]
                sigma_fit = popt[5]
                # Check uncertainty: skip if uncertainty in any parameter is more than 30%
                errs = np.sqrt(np.diag(pcov))
                # Check if spacing is sufficient for correction when photon flux is 1 e-/frame
                # noise_at_1e = np.sqrt(sigma_fit ** 2 + (1 * u.electron).to(adu_unit).value ** 2)
                if np.any(popt / errs < 3) or B_fit > A_fit:
                    poor_fit_count += 1
                    # print(popt)
                    # print(errs)
                    # plt.hist(pixel_data, bins=bins, density=True, alpha=0.6, label='Data Histogram')
                    # x_fit = np.linspace(np.min(bin_centers), np.max(bin_centers), 200)
                    # y_fit = rtn_triple_gaussian(x_fit, *popt)
                    # plt.plot(x_fit, y_fit, 'r-', label='Fit')
                    # plt.show()
                # elif (d_fit >= min_spacing * noise_at_1e) and A_fit < 0.95:
                elif (d_fit >= 3 * sigma_fit) and A_fit < 0.95:
                    rtn_params_array[:-1, ix, iy] = [(mu_fit * adu_unit).to(u.electron).value,
                                                   A_fit, B_fit,
                                                   (d_fit * adu_unit).to(u.electron).value,
                                                   (sigma_fit * adu_unit).to(u.electron).value]
                    rtn_mask[ix, iy] = True
                    # print(mu_guess, A_guess, B1_guess, d_guess, sigma_guess)
                    # print(mu_fit, popt[1], popt[2], d_fit, sigma_fit)
                    # print(popt)
                    # print(errs)
                    # plt.hist(pixel_data, bins=bins, density=True, alpha=0.6, label='Data Histogram')
                    # x_fit = np.linspace(np.min(bin_centers), np.max(bin_centers), 200)
                    # y_fit = rtn_triple_gaussian(x_fit, *popt)
                    # plt.plot(x_fit, y_fit, 'r-', label='Fit')
                    # plt.show()
                else:
                    too_close_count += 1
                    # print(popt)
                    # print(errs)
                    # plt.hist(pixel_data, bins=bins, density=True, alpha=0.6, label='Data Histogram')
                    # x_fit = np.linspace(np.min(bin_centers), np.max(bin_centers), 200)
                    # y_fit = rtn_triple_gaussian(x_fit, *popt)
                    # plt.plot(x_fit, y_fit, 'r-', label='Fit')
                    # plt.show()
                # fit_time = time.time()
            except RuntimeError:
                fit_fail_count += 1
                # fit_time = time.time()
                # Fit did not converge; skip this pixel
                continue
    print(poor_fit_count, too_close_count, fit_fail_count)
    return rtn_params_array, rtn_mask

def add_lambda_max(rtn_params_array, lut, plot=False):
    for ix in range(rtn_params_array.shape[1]):
        for iy in range(rtn_params_array.shape[2]):
            if np.isnan(rtn_params_array[0, ix, iy]):
                continue
            delta_x = rtn_params_array[3, ix, iy]
            sigma = rtn_params_array[4, ix, iy]
            lambda_max = lut.get_lambda_max(read_noise=sigma, delta_x=delta_x)
            # LUT only goes up to 100 e-
            if lambda_max == np.inf:
                lambda_max = 100.0
            rtn_params_array[5, ix, iy] = lambda_max
    if plot:
        plt.rcParams.update({'font.size': 14})
        plt.hist(rtn_params_array[5, ~np.isnan(rtn_params_array[0])].flatten(), bins=200, histtype='step', alpha=0.7)
        # Plot vertical line at median lambda max
        median_lambda_max = np.median(rtn_params_array[5, ~np.isnan(rtn_params_array[0])])
        plt.axvline(median_lambda_max, color='red', linestyle='dashed', linewidth=1, label=r"Median $\lambda_{max}$" + f": {median_lambda_max:.2f} e-")
        plt.legend()
        plt.xlabel("$\lambda_{max}$ (e-)")
        plt.ylabel("Number of Pixels")
        plt.yscale('log')
        plt.show()
    return rtn_params_array

def get_snr_ratio(read_noise_arr, rtn_params_arr, plot=False):
    plt.rcParams.update({'font.size': 14})
    # set nans to zero
    rtn_params_arr = np.nan_to_num(rtn_params_arr)
    old_read_noise = np.sqrt(np.mean(read_noise_arr**2))
    flux_values = np.logspace(-1, 1.5, 20)
    snr_values_old = np.zeros_like(flux_values)
    snr_values_corr = np.zeros_like(flux_values)
    for i, flux in enumerate(flux_values):
        correctable_mask = (rtn_params_arr[5] >= flux) & (~np.isnan(rtn_params_arr[0]))
        eff_read_noise_arr = read_noise_arr * ~correctable_mask + rtn_params_arr[4] * correctable_mask
        new_read_noise = np.sqrt(np.mean(eff_read_noise_arr**2))
        noise_old = np.sqrt(flux + old_read_noise**2)
        noise_corr = np.sqrt(flux + new_read_noise**2)
        snr_values_old[i] = np.mean(flux / noise_old)
        snr_values_corr[i] = np.mean(flux / noise_corr)
    if plot:
        plt.plot(flux_values, snr_values_corr / snr_values_old, 'b')
        plt.xlabel('Flux (e-)')
        plt.ylabel('SNR Improvement Factor')
        plt.xscale('log')
        plt.axhline(1.0, color='grey', linestyle='-', linewidth=1)
        plt.show()
    return flux_values, snr_values_corr / snr_values_old

if __name__ == "__main__":
    t0 = time.time()
    bias_stack_file = 'bias_stack_subset.fits'
    # gain = 8.9
    gain = 42  # ADU/e-
    bias_stack = fits.open(bias_stack_file)[0].data.astype(np.int32)
    t1 = time.time()
    print(f"Loaded bias stack in {t1 - t0:.2f} seconds.")
    lut = ThresholdLUT.load('rts_threshold_lut.pkl')
    # Not sigma clipping for now. Doesn't do much and introduces NaNs.
    # bias_stack = sigma_clip(bias_stack, sigma=10, max_iters=5)
    adu = u.electron / gain
    read_noise_stats, read_noise_array = get_read_noise_stats(bias_stack, adu, plot=True)
    t2 = time.time()
    print(f"Computed read noise stats in {t2 - t1:.2f} seconds.")
    nonnormal_mask = identify_nonnormal_pixels(bias_stack, adu, plot=True)
    nonnormal_pix = np.sum(nonnormal_mask)
    print(f"Identified {nonnormal_pix} non-normal pixels.")
    t3 = time.time()
    print(f"Identified non-normal pixels in {t3 - t2:.2f} seconds.")
    rtn_params_arr, rtn_mask = fit_rtn_parameters(bias_stack, nonnormal_mask, read_noise_stats, adu, min_spacing=3, verbose=True)
    t4 = time.time()
    print(f"Fit RTN parameters in {t4 - t3:.2f} seconds.")
    print(f"Identified {np.sum(rtn_mask)} correctable RTN pixels.")
    rtn_params_array = add_lambda_max(rtn_params_arr, lut, plot=True)
    # Save rtn_params to a fits file.
    hdu = fits.PrimaryHDU(rtn_params_array)
    hdu.writeto('rtn_params.fits', overwrite=True)
    print("Saved RTN parameters to rtn_params.fits")
    fluxes, snr_ratios = get_snr_ratio(read_noise_array.to(u.electron).value, rtn_params_array, plot=True)
