# Using bias stack, locate pixels with RTN and fit their parameters.
# Chris Layden

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import units as u
from scipy.optimize import curve_fit
from scipy.stats import norm
import copy
import time
import resource
import os
from glob import glob
from numba import njit
from scripts.make_lut import ThresholdLUT
import multiprocessing
from fitsreader import FITSReader

def sigma_clip(data, sigma=10, max_iters=5, clip_val='mean'):
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
            clipped_data = np.where(mask, clipped_data, mean)
        else:
            clipped_data = np.where(mask, clipped_data, np.nan)
    return clipped_data

def get_read_noise_stats(data, adu_unit, plot=False, save_path=None):
    if len(data.shape) != 3:
        raise ValueError("Input data must be a 3D array of images.")
    if data.shape[0] < 10:
        raise ValueError("At least ten bias images required to compute per-pixel read noise.")
    if adu_unit is not None:
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
    if plot or save_path:
        plt.rcParams.update({'font.size': 14})
        plt.figure()
        plt.hist(read_noise_array.flatten(), bins=200, histtype='step', alpha=0.7)
        plt.xlabel("Read Noise (e-)" if adu_unit is not None else "Read Noise (ADU)")
        plt.ylabel("Number of Pixels")
        plt.yscale('log')
        # Handle both unit and non-unit cases
        mean_val = mean_read_noise.value if hasattr(mean_read_noise, 'value') else mean_read_noise
        median_val = median_read_noise.value if hasattr(median_read_noise, 'value') else median_read_noise
        rms_val = rms_read_noise.value if hasattr(rms_read_noise, 'value') else rms_read_noise
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_read_noise:.2f}")
        plt.axvline(median_val, color='blue', linestyle='dashed', linewidth=1, label=f"Median: {median_read_noise:.2f}")
        plt.axvline(rms_val, color='green', linestyle='dashed', linewidth=1, label=f"RMS: {rms_read_noise:.2f}")
        plt.legend()
        if save_path:
            plt.savefig(os.path.join(save_path, 'read_noise_histogram.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()
    return read_noise_stats, read_noise_array

def smooth_data(data, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)
    smoothing_arr = np.random.uniform(-0.5, 0.5, size=data.shape)
    smoothed_data = data + smoothing_arr
    return smoothed_data

def ad_statistics_normal(data):
    n, nx, ny = data.shape
    clipped_data = sigma_clip(data, sigma=5, max_iters=3)
    # Use fixed random seed for reproducibility
    smoothed_data = smooth_data(clipped_data, random_seed=42)
    sorted_data = np.sort(smoothed_data, axis=0)
    mean = np.mean(sorted_data, axis=0, keepdims=True)
    std = np.std(sorted_data, axis=0, ddof=1, keepdims=True)
    z = (sorted_data - mean) / std
    cdf = norm.cdf(z)
    eps = np.finfo(float).eps
    cdf = np.clip(cdf, eps, 1 - eps)
    i = np.arange(1, n + 1).reshape(-1, 1, 1)
    term = (2 * i - 1) * (np.log(cdf) + np.log(1 - cdf[::-1, :, :]))
    A2 = -n - np.mean(term, axis=0)
    return A2

def identify_nonnormal_pixels(data, adu_unit, threshold=1.092, plot=False, save_path=None):
    A2 = ad_statistics_normal(data)
    read_noise_stats, read_noise_array = get_read_noise_stats(data, adu_unit, plot=False)
    low_cutoff = read_noise_stats["median_read_noise"]
    nonnormal_mask = (A2 > threshold) & (read_noise_array > low_cutoff)
    if plot or save_path:
        # Convert to electron units if available, otherwise use as-is (ADU)
        if hasattr(read_noise_array, 'to'):
            read_noise_array = read_noise_array.to(u.electron).value
            xlabel = "Read Noise (e-)"
        else:
            xlabel = "Read Noise (ADU)"
        nonnormal_read_noise = read_noise_array[nonnormal_mask]
        plt.figure()
        plt.hist(read_noise_array.flatten(), bins=200, range=(0, np.percentile(read_noise_array, 99.95)), histtype='step', alpha=0.5, label='All pixels')
        plt.hist(nonnormal_read_noise.flatten(), bins=200, range=(0, np.percentile(read_noise_array, 99.95)), histtype='step', alpha=0.5, label='Pixels failing AD test')
        plt.xlabel(xlabel)
        plt.ylabel("Number of Non-normal Pixels")
        plt.yscale('log')
        plt.title("Read Noise Distribution of Non-normal Pixels")
        plt.legend(fontsize=11)
        if save_path:
            plt.savefig(os.path.join(save_path, 'nonnormal_pixels.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()
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

def fit_rtn_parameters(data, nonnormal_mask, read_noise_stats, adu_unit, verbose=False):
    rtn_mask = np.zeros(nonnormal_mask.shape, dtype=bool)
    rtn_params_array = np.full((6, data.shape[1], data.shape[2]), np.nan)
    poor_fit_count = 0
    too_close_count = 0
    too_rare_count = 0
    fit_fail_count = 0
    rtn_count = 0
    single_sided_count = 0
    n_images, nx, ny = data.shape
    for ix in range(nx):
        if verbose:
            print(f"Fitting column {ix+1}/{nx}")
        for iy in range(ny):
            if not nonnormal_mask[ix, iy]:
                continue
            pixel_data = data[:, ix, iy]
            # Handle both unit and non-unit cases
            unit_offset = adu_unit.to(u.electron).value if adu_unit is not None else 1.0
            bins_min = np.round(np.percentile(pixel_data, 0.5) - unit_offset).astype(int)
            bins_max = np.round(np.percentile(pixel_data, 99.5) + unit_offset).astype(int)
            bin_size = np.round(np.max([1, (bins_max - bins_min) / 30])).astype(int)
            bins = np.arange(bins_min - bin_size, bins_max + 2 * bin_size, bin_size)
            counts, bin_edges = np.histogram(pixel_data, bins=bins, density=True)
            # Use for ORCA-Quest 2 or other cameras where some digital values are not reported
            # For any bins with zero counts, set to the mean of the two adjacent bins
            # zero_bins = counts == 0
            # for i in np.where(zero_bins)[0]:
            #     if i == 0:
            #         counts[i] = counts[i + 1]
            #     elif i == len(counts) - 1:
            #         counts[i] = counts[i - 1]
            #     else:
            #         counts[i] = (counts[i - 1] + counts[i + 1]) / 2
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            mu_guess = np.mean(pixel_data)
            # Convert sigma_guess to appropriate units
            if adu_unit is not None:
                sigma_guess = read_noise_stats['median_read_noise'].to(adu_unit).value
            else:
                sigma_guess = read_noise_stats['median_read_noise']
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
                errs = np.sqrt(np.diag(pcov))
                mu_fit = popt[0]
                A_fit = popt[1] / (popt[1] + popt[2] + popt[3])  # Normalize A
                B1_fit = popt[2] / (popt[1] + popt[2] + popt[3])  # Normalize B1
                B2_fit = popt[3] / (popt[1] + popt[2] + popt[3])  # Normalize B2
                d_fit = popt[4]
                sigma_fit = popt[5]
                A_err = errs[1] / (popt[1] + popt[2] + popt[3])  # Approximate error on normalized A
                B1_err = errs[2] / (popt[1] + popt[2] + popt[3])  # Approximate error on normalized B1
                B2_err = errs[3] / (popt[1] + popt[2] + popt[3])  # Approximate error on normalized B2
                # Suppress division by zero warnings
                with np.errstate(divide='ignore', invalid='ignore'):
                    bad_fit_checks = abs(popt / errs) < 3
                # Allow for one of the side peaks to be poorly fit if consistent with zero
                if abs(B1_fit) < 0.03 and bad_fit_checks[2]:
                    bad_fit_checks[2] = False
                    single_sided = True
                elif abs(B2_fit) < 0.03 and bad_fit_checks[3]:
                    bad_fit_checks[3] = False
                    single_sided = True
                else:
                    single_sided = False
                if np.any(bad_fit_checks) or B1_fit > A_fit or B2_fit > A_fit:
                    # if (np.std(pixel_data) * adu_unit).to(u.electron).value > 0.4:
                    #     x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
                    #     y_fit = rtn_triple_gaussian(x_fit, *popt)
                    #     plt.hist(pixel_data, bins=np.arange(bins_min, bins_max + bin_size, bin_size), density=True, histtype='step', alpha=0.3, label='Pixel Data')
                    #     plt.plot(x_fit, y_fit, 'r-', label='Fit')
                    #     plt.xlabel("Pixel Value (ADU)")
                    #     plt.ylabel("Density")
                    #     print('not fit success')
                    #     # print(B1_fit, B1_err, B2_fit, B2_err, bad_fit_checks)
                    #     print((sigma_fit * adu_unit).to(u.electron), (np.std(pixel_data) * adu_unit).to(u.electron))
                    #     plt.show(block=True)
                    poor_fit_count += 1
                elif A_fit > 0.95:
                    too_rare_count += 1
                elif (d_fit >= 3 * sigma_fit):
                    # x_fit = np.linspace(bin_centers[0], bin_centers[-1], 1000)
                    # y_fit = rtn_triple_gaussian(x_fit, *popt)
                    # plt.hist(pixel_data, bins=np.arange(bins_min, bins_max + bin_size, bin_size), density=True, histtype='step', alpha=0.3, label='Pixel Data')
                    # plt.plot(bin_centers, counts, 'kx', label='Binned Data')
                    # plt.plot(x_fit, y_fit, 'r-', label='Fit')
                    # plt.xlabel("Pixel Value (ADU)")
                    # plt.ylabel("Density")
                    # print((sigma_fit * adu_unit).to(u.electron), (np.std(pixel_data) * adu_unit).to(u.electron))
                    # plt.show(block=True)
                    # Convert to electrons if adu_unit is available, otherwise keep in ADU
                    if adu_unit is not None:
                        rtn_params_array[:-1, ix, iy] = [(mu_fit * adu_unit).to(u.electron).value,
                                                       A_fit, B1_fit,
                                                       (d_fit * adu_unit).to(u.electron).value,
                                                       (sigma_fit * adu_unit).to(u.electron).value]
                    else:
                        rtn_params_array[:-1, ix, iy] = [mu_fit, A_fit, B1_fit, d_fit, sigma_fit]
                    rtn_mask[ix, iy] = True
                    rtn_count += 1
                    if single_sided:
                        single_sided_count += 1
                else:
                    too_close_count += 1
            except RuntimeError:
                fit_fail_count += 1
                continue
    print(f"Identified {rtn_count} correctable RTN pixels. {single_sided_count} are single-sided.")
    print(f"Poor fits: {poor_fit_count}, Too close: {too_close_count}, Too rare: {too_rare_count}, Fit failures: {fit_fail_count}")
    return rtn_params_array, rtn_mask

def add_lambda_max(rtn_params_array, lut, plot=False, save_path=None):
    for ix in range(rtn_params_array.shape[1]):
        for iy in range(rtn_params_array.shape[2]):
            if np.isnan(rtn_params_array[0, ix, iy]):
                continue
            delta_x = rtn_params_array[3, ix, iy]
            sigma = rtn_params_array[4, ix, iy]
            lambda_max = lut.get_lambda_max(read_noise=sigma, delta_x=delta_x)
            if lambda_max == np.inf:
                lambda_max = 100.0
            rtn_params_array[5, ix, iy] = lambda_max
    if plot or save_path:
        plt.rcParams.update({'font.size': 14})
        plt.figure()
        plt.hist(rtn_params_array[5, ~np.isnan(rtn_params_array[0])].flatten(), bins=200, histtype='step', alpha=0.7)
        median_lambda_max = np.median(rtn_params_array[5, ~np.isnan(rtn_params_array[0])])
        plt.axvline(median_lambda_max, color='red', linestyle='dashed', linewidth=1, label=r"Median $\lambda_{max}$" + f": {median_lambda_max:.2f} e-/pix/frame")
        plt.legend(fontsize=11)
        plt.xlabel(r"$\lambda_{max}$ (e-/pix/frame)")
        plt.ylabel("Number of Pixels")
        plt.yscale('log')
        if save_path:
            plt.savefig(os.path.join(save_path, 'lambda_max_histogram.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()
    return rtn_params_array

def get_snr_ratio(read_noise_arr, rtn_params_arr, plot=False, save_path=None):
    plt.rcParams.update({'font.size': 14})
    rtn_params_arr = np.nan_to_num(rtn_params_arr)
    old_read_noise = np.sqrt(np.mean(read_noise_arr**2))
    flux_values, rn_values = get_new_read_noise(read_noise_arr, rtn_params_arr, plot=False)
    # flux_values = np.logspace(-1, 1.5, 20)
    snr_values_old = np.zeros_like(flux_values)
    snr_values_corr = np.zeros_like(flux_values)
    for i, flux in enumerate(flux_values):
        noise_old = np.sqrt(flux + old_read_noise**2)
        noise_corr = np.sqrt(flux + rn_values[i]**2)
        snr_values_old[i] = np.mean(flux / noise_old)
        snr_values_corr[i] = np.mean(flux / noise_corr)
    if plot or save_path:
        plt.figure()
        plt.plot(flux_values, snr_values_corr / snr_values_old, 'b')
        plt.xlabel('Average Count Rate (e-/pix/frame)')
        plt.ylabel('SNR Improvement Factor')
        plt.xscale('log')
        plt.axhline(1.0, color='grey', linestyle='-', linewidth=1)
        if save_path:
            plt.savefig(os.path.join(save_path, 'snr_improvement.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()
    return flux_values, snr_values_corr / snr_values_old

def get_new_read_noise(read_noise_arr, rtn_params_arr, plot=False, save_path=None):
    plt.rcParams.update({'font.size': 14})
    rtn_params_arr = np.nan_to_num(rtn_params_arr)
    old_read_noise = np.sqrt(np.mean(read_noise_arr**2))
    flux_values = np.logspace(-3, 2, 30)
    # Add zero to flux values to show best-case read noise after correction
    flux_values = np.insert(flux_values, 0, 0.0001)
    new_read_noise_vals = np.zeros_like(flux_values)
    lowest_read_noise_arr = read_noise_arr.copy()
    rtn_mask = rtn_params_arr[0] > 0
    for i, flux in enumerate(flux_values):
        correctable_mask = (rtn_params_arr[5] > flux) & rtn_mask
        # As an approximation, linearly interpolate from original read noise to sigma as lambda goes from 0 to lambda_max for each pixel
        eff_read_noise_arr = read_noise_arr.copy()
        eff_read_noise_arr[correctable_mask] = rtn_params_arr[4, correctable_mask] + (read_noise_arr[correctable_mask] - rtn_params_arr[4, correctable_mask]) * (flux / rtn_params_arr[5, correctable_mask])
        # eff_read_noise_arr = read_noise_arr * ~correctable_mask + rtn_params_arr[4] * correctable_mask
        new_read_noise = np.sqrt(np.mean(eff_read_noise_arr**2))
        new_read_noise_vals[i] = new_read_noise
        if i == 0:
            lowest_read_noise_arr = eff_read_noise_arr
    if plot or save_path:
        # First plot histogram of best-case read noise at zero flux
        plt.figure()
        old_rms = np.sqrt(np.mean(read_noise_arr**2))
        new_rms = np.sqrt(np.mean(lowest_read_noise_arr**2))
        plt.hist(read_noise_arr.flatten(), bins=200, histtype='step', alpha=0.5, label='Original read noise')
        plt.hist(lowest_read_noise_arr.flatten(), bins=200, histtype='step', alpha=0.5, label='Corrected read noise at zero flux')
        plt.axvline(old_rms, color='blue', linestyle='--', linewidth=1, label=f'Original RMS: {old_rms:.3f} e-')
        plt.axvline(new_rms, color='orange', linestyle='--', linewidth=1, label=f'RMS without RTN: {new_rms:.3f} e-')
        plt.xlabel("Read Noise (e-)")
        plt.ylabel("Number of Pixels")
        plt.yscale('log')
        plt.legend(fontsize=11)
        if save_path:
            plt.savefig(os.path.join(save_path, 'eff_read_noise_histogram.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()
        plt.figure()
        plt.plot(flux_values, new_read_noise_vals, 'b')
        plt.xlabel('Average Count Rate (e-/pix/frame)')
        plt.ylabel('Post-correction RMS Read Noise (e-)')
        plt.xscale('log')
        plt.axhline(old_read_noise, color='grey', linestyle='-', linewidth=1)
        # Label the old read noise line
        plt.text(flux_values[0], old_read_noise * 0.99, f'Raw RMS read noise: {old_read_noise:.2f} e-', color='grey', va='top', ha='left')
        plt.text(flux_values[0], new_read_noise_vals[0] * 1.01, f'Min. corrected RMS\nread noise: {new_read_noise_vals[0]:.2f} e-', color='blue', va='bottom', ha='left')
        if save_path:
            plt.savefig(os.path.join(save_path, 'eff_read_noise.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()
    return flux_values, new_read_noise_vals

def get_bias_stack_info(fits_files, frames_to_keep=None):
    """Get frame count and sensor shape without loading pixel data."""

    total_frames = 0
    frame_shape_2d = None
    file_frame_counts = []

    for file in fits_files:
        with fits.open(file) as hdul:
            data_shape = hdul[0].shape
            if len(data_shape) == 2:
                file_frame_counts.append(1)
                total_frames += 1
                if frame_shape_2d is None:
                    frame_shape_2d = data_shape
            elif len(data_shape) == 3:
                file_frame_counts.append(data_shape[0])
                total_frames += data_shape[0]
                if frame_shape_2d is None:
                    frame_shape_2d = data_shape[1:]
            else:
                raise ValueError(f"Unexpected data shape {data_shape} in {file}. Expected 2D or 3D arrays.")

        if frames_to_keep is not None and total_frames >= frames_to_keep:
            break

    if frames_to_keep is None or frames_to_keep > total_frames:
        n_frames = total_frames
    else:
        n_frames = frames_to_keep

    print(f"Bias stack info: {n_frames} frames, sensor shape {frame_shape_2d}")
    return n_frames, frame_shape_2d


def get_reader(fits_files, nparallel):
    # Try to ensure we can have enough open file handles
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    nfhandles = max(soft, nparallel * len(fits_files) + 10)
    if hard < nfhandles:
        hard = nfhandles
    resource.setrlimit(resource.RLIMIT_NOFILE, (nfhandles, hard))
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    ptsgen = FITSReader(fits_files)
    return ptsgen


def load_bias_stack_slice(ptsgen, row_start, row_end, frames_to_keep=None):
    """Load all frames but only for rows [row_start:row_end].

    Returns array of shape (n_frames, row_end - row_start, n_cols).
    """
    return ptsgen.get_rows(row_start, row_end, frames_to_keep=frames_to_keep)


def fit_slice(si, r0, r1, ptsgen, frames_to_keep, nonnormal_mask_full, read_noise_stats, adu, verbose):
    slice_nonnormal = nonnormal_mask_full[r0:r1, :]
    nonnormal_in_slice = np.sum(slice_nonnormal)
    print(f"  Slice {si+1}: rows {r0}-{r1} ({nonnormal_in_slice} non-normal pixels)")
    if nonnormal_in_slice == 0:
        print(f"    No non-normal pixels in this slice, skipping.")
        return si, None

    bias_slice = load_bias_stack_slice(ptsgen, r0, r1, frames_to_keep=frames_to_keep)
    bias_slice = np.array(bias_slice)
    slice_params, slice_rtn_mask = fit_rtn_parameters(bias_slice, slice_nonnormal,
                                                      read_noise_stats, adu,
                                                      verbose=False)
    del bias_slice
    return si, slice_params




def main():
    parser = argparse.ArgumentParser(description='Identify and fit RTN parameters from bias frames (sliced mode).')
    parser.add_argument('files', type=str, nargs='+', help='Folder containing bias FITS files')
    parser.add_argument('-g', '--gain', type=float, default=None, help='Gain in ADU/e- (optional; if not provided, analysis is done in ADU units, and results are not saved.)')
    parser.add_argument('-o', '--output', type=str, default=None,
                        help='Output path for rtn_params.fits (default: ./rtn_fits_output/)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to use from bias stack (default: all available)')
    parser.add_argument('--slice-rows', type=int, default=1,
                        help='Number of rows per slice (default: 1).' 'Use this to limit RAM usage.')
    parser.add_argument('-p', '--parallel', type=int, default=os.cpu_count(), help=f'Number of processors to use (default:{os.cpu_count()}).')
    parser.add_argument('--lut', type=str, default=os.path.join(os.path.dirname(__file__), 'rts_threshold_lut.pkl'),
                        help='Path to threshold LUT file (default: [script_path]/rts_threshold_lut.pkl)')
    parser.add_argument('--plot', action='store_true',
                        help='Display plots (non-blocking) as the script runs')
    parser.add_argument('--save-plots', action='store_true',
                        help='Save plots to output folder')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output during fitting')
    args = parser.parse_args()
    if args.output is None:
        output_dir = os.path.realpath(os.path.join(os.getcwd(), 'rtn_fits_output'))
    else:
        output_dir = os.path.realpath(args.output)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'rtn_params.fits')
    plots_save_path = output_dir if args.save_plots else None
    plot = args.plot
    frames_to_keep = args.frames

    if plot:
        plt.ion()

    t0 = time.time()

    # Get sensor dimensions without loading data
    fits_files = args.files
    n_frames, (n_rows, n_cols) = get_bias_stack_info(fits_files, frames_to_keep)
    if frames_to_keep is None or frames_to_keep > n_frames:
        frames_to_keep = n_frames

    lut = ThresholdLUT.load(args.lut)
    if args.gain is not None:
        adu = u.electron / args.gain
    else:
        adu = None
        print("Note: Gain not specified. Analysis will be performed in ADU units.")

    slice_rows = args.slice_rows if args.slice_rows is not None else n_rows
    slices = [(r, min(r + slice_rows, n_rows)) for r in range(0, n_rows, slice_rows)]
    n_slices = len(slices)
    print(f"Processing sensor in {n_slices} slice(s) of up to {slice_rows} rows each.")

    frames_for_stats = min(1000, frames_to_keep)
    frames_for_nonnormal = min(400, frames_to_keep)

    # Allocate full-sensor output arrays
    read_noise_array_full = np.zeros((n_rows, n_cols))
    mean_bias_frame_full = np.zeros((n_rows, n_cols))
    nonnormal_mask_full = np.zeros((n_rows, n_cols), dtype=bool)
    rtn_params_full = np.full((6, n_rows, n_cols), np.nan)

    # We need global read noise stats (especially median_read_noise) before fitting.
    # Pass 1: Compute per-pixel read noise for each slice to get the global median.
    print(f"\n=== Pass 1: Computing per-pixel read noise ({n_slices} slice(s), {frames_for_stats} frames) ===")
    t1 = time.time()

    ptsgen = get_reader(fits_files, args.parallel)
    for si, (r0, r1) in enumerate(slices):
        print(f"  Slice {si+1}/{n_slices}: rows {r0}-{r1}")
        bias_slice = load_bias_stack_slice(ptsgen, r0, r1, frames_to_keep=frames_for_stats)
        if adu is not None:
            slice_data = bias_slice * adu
        else:
            slice_data = bias_slice.astype(float)
        read_noise_array_full[r0:r1, :] = np.sqrt(np.nanvar(slice_data, axis=0, ddof=1))
        mean_bias_frame_full[r0:r1, :] = np.mean(bias_slice, axis=0)
        del bias_slice, slice_data

    # Compute global read noise stats from the full per-pixel array
    mean_read_noise = np.nanmean(read_noise_array_full)
    median_read_noise = np.nanmedian(read_noise_array_full)
    rms_read_noise = np.sqrt(np.nanmean(read_noise_array_full**2))
    read_noise_sigma = np.nanstd(read_noise_array_full, ddof=1)
    # Attach units if needed
    if adu is not None:
        mean_read_noise = mean_read_noise * u.electron
        median_read_noise = median_read_noise * u.electron
        rms_read_noise = rms_read_noise * u.electron
        read_noise_sigma = read_noise_sigma * u.electron
        read_noise_array_with_units = read_noise_array_full * u.electron
    else:
        read_noise_array_with_units = read_noise_array_full

    read_noise_stats = {"mean_read_noise": mean_read_noise,
                        "median_read_noise": median_read_noise,
                        "rms_read_noise": rms_read_noise,
                        "read_noise_sigma": read_noise_sigma}
    print(f"  Global median read noise: {median_read_noise:.4f}")

    # Save read noise array and mean bias frame
    if args.gain is not None:
        fits.writeto(os.path.join(output_dir, 'read_noise_frame.fits'),
                     read_noise_array_full, overwrite=True)
        fits.writeto(os.path.join(output_dir, 'mean_bias_frame.fits'),
                     mean_bias_frame_full, overwrite=True)

    # Plot read noise histogram (using the full array)
    if plot or plots_save_path:
        plt.rcParams.update({'font.size': 14})
        plt.figure()
        plt.hist(read_noise_array_full.flatten(), bins=200, histtype='step', alpha=0.7)
        unit_label = "Read Noise (e-)" if adu is not None else "Read Noise (ADU)"
        plt.xlabel(unit_label)
        plt.ylabel("Number of Pixels")
        plt.yscale('log')
        mean_val = mean_read_noise.value if hasattr(mean_read_noise, 'value') else mean_read_noise
        median_val = median_read_noise.value if hasattr(median_read_noise, 'value') else median_read_noise
        rms_val = rms_read_noise.value if hasattr(rms_read_noise, 'value') else rms_read_noise
        plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=1, label=f"Mean: {mean_read_noise:.2f}")
        plt.axvline(median_val, color='blue', linestyle='dashed', linewidth=1, label=f"Median: {median_read_noise:.2f}")
        plt.axvline(rms_val, color='green', linestyle='dashed', linewidth=1, label=f"RMS: {rms_read_noise:.2f}")
        plt.legend()
        if plots_save_path:
            plt.savefig(os.path.join(plots_save_path, 'read_noise_histogram.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

    t2 = time.time()
    print(f"Pass 1 completed in {t2 - t1:.2f} seconds.")

    # Pass 2: Identify non-normal pixels per slice
    print(f"\n=== Pass 2: Identifying non-normal pixels ({n_slices} slice(s), {frames_for_nonnormal} frames) ===")
    for si, (r0, r1) in enumerate(slices):
        print(f"  Slice {si+1}/{n_slices}: rows {r0}-{r1}")
        bias_slice = load_bias_stack_slice(ptsgen, r0, r1, frames_to_keep=frames_for_nonnormal)
        # Run AD test on this slice
        A2 = ad_statistics_normal(bias_slice)
        slice_read_noise = read_noise_array_with_units[r0:r1, :]
        low_cutoff = read_noise_stats["median_read_noise"]
        nonnormal_mask_full[r0:r1, :] = (A2 > 1.092) & (slice_read_noise > low_cutoff)
        del bias_slice, A2

    nonnormal_pix = np.sum(nonnormal_mask_full)
    print(f"  Identified {nonnormal_pix} non-normal pixels total.")

    # Plot nonnormal pixels histogram
    if plot or plots_save_path:
        if hasattr(read_noise_array_with_units, 'to'):
            rn_plot = read_noise_array_with_units.to(u.electron).value
            xlabel = "Read Noise (e-)"
        else:
            rn_plot = read_noise_array_full
            xlabel = "Read Noise (ADU)"
        nonnormal_read_noise = rn_plot[nonnormal_mask_full]
        plt.figure()
        plt.hist(rn_plot.flatten(), bins=200, range=(0, np.percentile(rn_plot, 99.95)), histtype='step', alpha=0.5, label='All pixels')
        plt.hist(nonnormal_read_noise.flatten(), bins=200, range=(0, np.percentile(rn_plot, 99.95)), histtype='step', alpha=0.5, label='Pixels failing AD test')
        plt.xlabel(xlabel)
        plt.ylabel("Number of Non-normal Pixels")
        plt.yscale('log')
        plt.title("Read Noise Distribution of Non-normal Pixels")
        plt.legend(fontsize=11)
        if plots_save_path:
            plt.savefig(os.path.join(plots_save_path, 'nonnormal_pixels.png'), dpi=150, bbox_inches='tight')
        if plot:
            plt.show(block=False)
            plt.pause(0.1)
        else:
            plt.close()

    t3 = time.time()
    print(f"Pass 2 completed in {t3 - t2:.2f} seconds.")

    # Pass 3: Fit RTN parameters per slice (uses ALL frames)
    # This can be very slow, use multiprocessing for parallel execution.

    print(f"\n=== Pass 3: Fitting RTN parameters ({n_slices} slice(s), {frames_to_keep} frames) ===")

    with multiprocessing.Pool(args.parallel) as pool:
        results = [pool.apply_async(fit_slice,
                   (si, r0, r1, ptsgen, frames_to_keep, nonnormal_mask_full, read_noise_stats, adu, args.verbose)) \
                   for si, (r0, r1) in enumerate(slices)]
        for r in results:
            si, slice_params = r.get()
            r0, r1 = slices[si]
            if slice_params is not None:
                rtn_params_full[:, r0:r1, :] = slice_params
                del slice_params

    t4 = time.time()
    print(f"Pass 3 completed in {t4 - t3:.2f} seconds.")

    # Only proceed with lambda_max and saving if gain is specified
    if args.gain is not None:
        print("\nComputing lambda_max for RTN pixels...")
        rtn_params_full = add_lambda_max(rtn_params_full, lut, plot=plot, save_path=plots_save_path)

        print("Saving RTN parameters to FITS file...")
        hdu = fits.PrimaryHDU(rtn_params_full)
        hdu.writeto(output_file, overwrite=True)
        print(f"Saved RTN parameters to {output_file}")

        # Only compute and save SNR plots if gain is specified
        if plot or args.save_plots:
            print("Computing SNR improvement...")
            fluxes, snr_ratios = get_snr_ratio(read_noise_array_full, rtn_params_full,
                                                plot=plot, save_path=plots_save_path)
            with open(os.path.join(output_dir, 'snr_improvement.csv'), 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Average Count Rate (e-/pix/frame)', 'SNR Improvement Factor'])
                for f, r in zip(fluxes, snr_ratios):
                    writer.writerow([f, r])
            print("Computing effective read noise after correction...")
            fluxes, new_read_noise_vals = get_new_read_noise(read_noise_array_full, rtn_params_full,
                                                            plot=plot, save_path=plots_save_path)
    else:
        print("Specify gain to save RTN parameters and compute estimated SNR improvement.")

    total_time = time.time() - t0
    print(f"\nDone. Total time: {total_time:.2f} seconds.")
    if plot:
        plt.ioff()
        plt.show()

if __name__ == "__main__":
    main()
