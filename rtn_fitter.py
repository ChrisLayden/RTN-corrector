# Using bias stack, locate pixels with RTN and fit their parameters.
# Chris Layden

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.stats import anderson, norm
import copy
import time

def sigma_clip(data, sigma=10, max_iters=5):
    # Sigma-clip data along axis 0 (time axis) to remove outliers.
    # Set clipped points to np.nan
    clipped_data = copy.deepcopy(data)
    for _ in range(max_iters):
        mean = np.mean(clipped_data, axis=0)
        std = np.std(clipped_data, axis=0, ddof=1)
        diff = np.abs(clipped_data - mean)
        mask = diff < (sigma * std)
        if np.all(mask):
            break
        # set outliers to mean
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
    if gain is not None:
        data = data.to(u.electron)
    read_var_array = np.nanvar(data, axis=0, ddof=1)
    read_noise_array = np.sqrt(read_var_array)
    mean_read_noise = np.nanmean(read_noise_array)
    median_read_noise = np.nanmedian(read_noise_array)
    rms_read_noise = np.sqrt(np.nanmean(read_var_array))
    read_noise_stats = {"mean_read_noise": mean_read_noise,
                        "median_read_noise": median_read_noise,
                        "rms_read_noise": rms_read_noise}
    if plot:
        plt.hist(read_noise_array.flatten(), bins=200, histtype='step', alpha=0.7)
        plt.xlabel("Read Noise (e-)" if gain is not None else "Read Noise (ADU)")
        plt.ylabel("Number of Pixels")
        plt.yscale('log')
        plt.axvline(mean_read_noise.value, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_read_noise:.2f}")
        plt.axvline(median_read_noise.value, color='blue', linestyle='dashed', linewidth=1, label=f"Median: {median_read_noise:.2f}")
        plt.axvline(rms_read_noise.value, color='green', linestyle='dashed', linewidth=1, label=f"RMS: {rms_read_noise:.2f}")
        plt.legend()
        plt.show()
    return read_noise_stats

def smooth_data(data):
    # Follow Ozdogru et al. to un-discretize data, making it suitable for testing and fitting.
    smoothing_arr = np.random.uniform(-0.5, 0.5, size=data.shape)
    smoothed_data = data + smoothing_arr
    return smoothed_data

def ad_statistics_normal(data):
    # Compute Anderson-Darling statistic for normality test for all pixels.
    # Change things to cupy/other GPU libraries later for speed.
    n, nx, ny = data.shape
    sorted_data = np.sort(data, axis=0)
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

def identify_nonnormal_pixels(data, threshold=1.092, plot=False):
    # Identify pixels with non-normal distributions based on AD statistic.
    # Threshold of 1.092 corresponds to 1% significance level for normality test
    # with infinite samples and unknown mean and variance (see https://doi.org/10.2307/2286009)
    A2 = ad_statistics_normal(data)
    nonnormal_mask = A2 > threshold
    # Plot read noise histogram of non-normal pixels
    if plot:
        read_var_array = np.var(data, axis=0, ddof=1)
        read_noise_array = np.sqrt(read_var_array)
        nonnormal_read_noise = read_noise_array[nonnormal_mask]
        plt.hist(read_noise_array.flatten(), bins=200, range=(0, np.max(read_noise_array)), histtype='step', alpha=0.3, label='All Pixels')
        plt.hist(nonnormal_read_noise.flatten(), bins=200, range=(0, np.max(read_noise_array)), histtype='step', alpha=0.7, label='Pixels failing AD test')
        plt.xlabel("Read Noise (ADU)")
        plt.ylabel("Number of Non-normal Pixels")
        plt.yscale('log')
        plt.title("Read Noise Distribution of Non-normal Pixels")
        plt.legend()
        plt.show()
    return nonnormal_mask

def rtn_triple_gaussian(x, mu, A, B1, B2, d, sigma):
    # Model RTN pixel histogram as sum of three Gaussians.
    central_gaussian = A * np.exp(- (x - mu) ** 2 / (2 * sigma) ** 2)
    left_gaussian = B1 * np.exp(- (x - d - mu) ** 2 / (2 * sigma) ** 2)
    right_gaussian = B2 * np.exp(- (x + d - mu) ** 2 / (2 * sigma) ** 2)
    return central_gaussian + left_gaussian + right_gaussian

def fit_rtn_parameters(data, nonnormal_mask, read_noise_stats, adu_unit, min_spacing=3):
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
        in multiples of central peak standard deviation.
    
    Returns:
    --------
    rtn_params : dict
        Dictionary with keys as (x, y) pixel coordinates and values as fitted RTN parameters:
        (mu, A, B, d, sigma). A is the normalized central peak amplitude and B the left peak amplitude,
        such that the right peak amplitude is (1 - A - B).
    """
    rtn_mask = np.zeros(nonnormal_mask.shape, dtype=bool)
    rtn_params = {}
    poor_fit_count = 0
    too_close_count = 0
    fit_fail_count = 0
    n_images, nx, ny = data.shape
    for ix in range(nx):
        for iy in range(ny):
            if not nonnormal_mask[ix, iy]:
                continue
            pixel_data = data[:, ix, iy]
            # Create histogram
            counts, bin_edges = np.histogram(pixel_data, bins=50, density=True)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            # Initial parameter guesses
            mu_guess = np.mean(pixel_data)
            sigma_guess = read_noise_stats['median_read_noise'].to(adu_unit).value
            A_guess = np.max(counts)
            B1_guess = 0.1 * A_guess
            B2_guess = 0.1 * A_guess
            d_guess = (10 * u.electron).to(adu_unit).value  # Initial guess of 10 e- spacing
            d_min = 3 * sigma_guess
            p0 = [mu_guess, A_guess, B1_guess, B2_guess, d_guess, sigma_guess]
            try:
                popt, pcov = curve_fit(rtn_triple_gaussian, bin_centers, counts, p0=p0, maxfev=1000, 
                                    bounds=([0, 0, 0, 0, d_min, 0], [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
                mu_fit = popt[0]
                A_fit = popt[1] / (popt[1] + popt[2] + popt[3])  # Normalize A
                B_fit = popt[3] / (popt[1] + popt[2] + popt[3])  # Normalize B1
                d_fit = popt[4]
                sigma_fit = popt[5]
                # Check uncertainty: skip if uncertainty in d is >20%
                err_d = np.sqrt(pcov[4, 4])
                if err_d / d_fit > 0.2:
                    poor_fit_count += 1
                    continue
                # Check if spacing is sufficient for correction when photon flux is 1 e-/frame
                noise_at_1e = np.sqrt(sigma_fit ** 2 + (1 * u.electron).to(adu_unit).value ** 2)
                if (d_fit >= min_spacing * noise_at_1e) and A_fit < 0.95:
                    rtn_params[(ix, iy)] = (mu_fit * adu_unit, A_fit, B_fit, d_fit * adu_unit, sigma_fit * adu_unit)
                    rtn_mask[ix, iy] = True
                    # if sigma_fit < 42:
                    #     print(mu_fit, A_fit, B_fit, d_fit / 42, sigma_fit / 42, np.std(pixel_data) / 42)
                    #     plt.hist(pixel_data, bins=50, density=True, alpha=0.6, label='Data Histogram')
                    #     x_fit = np.linspace(np.min(bin_centers), np.max(bin_centers), 200)
                    #     y_fit = rtn_triple_gaussian(x_fit, *popt)
                    #     plt.plot(x_fit, y_fit, 'r-', label='Fit')
                    #     plt.show()
                else:
                    too_close_count += 1
            except RuntimeError:
                fit_fail_count += 1
                # Fit did not converge; skip this pixel
                continue
    return rtn_params, rtn_mask

def plot_naive_corrected(data, rtn_params, adu_unit):
    # Plot histogram of read noise before correction and the best possible correction.
    read_noise_array = np.sqrt(np.nanvar(data, axis=0, ddof=1)) * adu_unit
    old_median = np.nanmedian(read_noise_array)
    old_rms = np.sqrt(np.nanmean(read_noise_array**2))
    corr_read_noise_array = copy.deepcopy(read_noise_array)
    for (ix, iy), params in rtn_params.items():
        corr_read_noise_array[ix, iy] = params[4]  # sigma
    new_rms = np.sqrt(np.nanmean(corr_read_noise_array**2))
    plt.hist(read_noise_array.flatten().to(u.electron).value, bins=200, histtype='step', alpha=0.7, label='Before Correction')
    plt.hist(corr_read_noise_array.flatten().to(u.electron).value, bins=200, histtype='step', alpha=0.7, label='Best-case Correction')
    plt.axvline(old_median.to(u.electron).value, color='red', linestyle='dashed', linewidth=1, label=f"Old Median: {old_median:.2f}")
    plt.axvline(old_rms.to(u.electron).value, color='blue', linestyle='dashed', linewidth=1, label=f"Old RMS: {old_rms:.2f}")
    plt.axvline(new_rms.to(u.electron).value, color='green', linestyle='dashed', linewidth=1, label=f"Best Case RMS: {new_rms:.2f}")
    plt.xlabel("Read Noise (e-)")
    plt.ylabel("Number of Pixels")
    plt.yscale('log')
    plt.legend()
    plt.show()
    return

if __name__ == "__main__":
    t0 = time.time()
    bias_stack_file = 'bias_stack_subset.fits'
    bias_stack = fits.open(bias_stack_file)[0].data.astype(np.int32)
    # Use a smaller subset for testing
    bias_stack = bias_stack[:,:100,:100]
    t1 = time.time()
    print(f"Loaded bias stack in {t1 - t0:.2f} seconds.")
    # Not sigma clipping for now. Doesn't do much and introduces NaNs.
    # bias_stack = sigma_clip(bias_stack, sigma=10, max_iters=5)
    gain = 42  # ADU/e-
    adu = u.electron / gain
    read_noise_stats = get_read_noise_stats(bias_stack, adu, plot=False)
    t2 = time.time()
    print(f"Computed read noise stats in {t2 - t1:.2f} seconds.")
    nonnormal_mask = identify_nonnormal_pixels(bias_stack, plot=False)
    t3 = time.time()
    print(f"Identified non-normal pixels in {t3 - t2:.2f} seconds.")
    rtn_params, rtn_mask = fit_rtn_parameters(bias_stack, nonnormal_mask, read_noise_stats, adu, min_spacing=3)
    t4 = time.time()
    print(f"Fit RTN parameters in {t4 - t3:.2f} seconds.")
    print(f"Identified {len(rtn_params)} correctable RTN pixels.")
    plot_naive_corrected(bias_stack, rtn_params, adu)
    # Look at pixels not in rtn_mask but with read noise above 2 e-
    # read_noise_array = np.sqrt(np.nanvar(bias_stack, axis=0, ddof=1)) * adu
    # high_noise_mask = read_noise_array > 2 * u.electron
    # missed_pixels = np.logical_and(high_noise_mask, np.logical_not(rtn_mask))
    # n_missed = np.sum(missed_pixels)
    # print(f"Number of high read noise pixels (>2 e-) not identified as correctable RTN: {n_missed}")
    # for ix in range(bias_stack.shape[1]):
    #     for iy in range(bias_stack.shape[2]):
    #         if missed_pixels[ix, iy]:
    #             plt.hist(bias_stack[:, ix, iy], bins=50, density=True, alpha=0.6, label='Data Histogram')
    #             plt.xlabel("Pixel Value (ADU)")
    #             plt.ylabel("Density")
    #             plt.title("Histograms of Missed Pixels")
    #             plt.legend()
    #             plt.show()
