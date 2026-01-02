# Remove random telegraph signal from a series of images using known pixel RTN fits.
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
from rtn_fitter import get_read_noise_stats

def get_rolling_mean_stack(image_stack, window_size=10):
    # image_stack: shape (n_images, nx, ny)
    n_images, nx, ny = image_stack.shape
    half_window = window_size // 2
    smoothed_stack = np.zeros_like(image_stack)
    for i in range(half_window, n_images - half_window):
        smoothed_stack[i] = image_stack[i - half_window:i + half_window + 1].mean(axis=0)
    return smoothed_stack

def get_diff_stack(image_stack, smoothed_stack, window_size=10):
    # image_stack, smoothed_stack: shape (n_images, nx, ny)
    n_images, nx, ny = image_stack.shape
    half_window = window_size // 2
    diff_stack = np.zeros_like(image_stack)
    for i in range(half_window, n_images - half_window):
        diff = image_stack[i] - smoothed_stack[i]
        diff_stack[i] = diff
    return diff_stack

def correct_signal_vals(diff_stack, image_stack, smoothed_stack, rtn_params, adu_unit, threshold=3, window_size=10):
    corrected_vals = image_stack.copy()
    rtn_mask = ~np.isnan(rtn_params[0])
    rtn_jumps = diff_stack * rtn_mask
    rtn_jumps_e = (rtn_jumps * adu_unit).to(u.electron).value
    smoothed_stack_e = (smoothed_stack * rtn_mask * adu_unit - rtn_params[0] * adu_unit).to(u.electron).value
    smoothed_stack_e = np.maximum(smoothed_stack_e, 0)
    noise_stack_e = np.sqrt(smoothed_stack_e + (rtn_params[4] * adu_unit).to(u.electron).value**2)
    delta_x_arr_e = (rtn_params[3] * adu_unit).to(u.electron).value
    high_cutoffs_low = np.max([delta_x_arr_e - threshold * noise_stack_e, threshold * noise_stack_e], axis=0)
    high_cutoffs_high = delta_x_arr_e + threshold * noise_stack_e
    low_cutoff_low = -delta_x_arr_e - threshold * noise_stack_e
    low_cutoff_high = np.min([-delta_x_arr_e + threshold * noise_stack_e, -threshold * noise_stack_e], axis=0)
    high_jumps_mask = (rtn_jumps_e > high_cutoffs_low) & (rtn_jumps_e < high_cutoffs_high)
    low_jumps_mask = (rtn_jumps_e < low_cutoff_high) & (rtn_jumps_e > low_cutoff_low)
    low_jumps_mask = low_jumps_mask.astype(int)
    high_jumps_mask = high_jumps_mask.astype(int)
    corrections_stack_e = (low_jumps_mask - high_jumps_mask) * delta_x_arr_e
    corrections_stack = (corrections_stack_e * u.electron).to(adu_unit).value
    # Replace NaNs with zeros for correction application
    corrections_stack = np.nan_to_num(corrections_stack)
    # Round to nearest integer ADU value
    corrections_stack = np.round(corrections_stack).astype(corrected_vals.dtype)
    corrected_vals += corrections_stack
    return corrected_vals

if __name__ == "__main__":
    image_stack_file = "bias_stack_subset.fits"
    rtn_params_file = "rtn_params.fits"
    gain = 42
    adu = u.electron / gain
    image_stack = fits.getdata(image_stack_file) # shape: (n_images, nx, ny)
    # NaNs in rtn_params indicate non-RTN pixels
    rtn_params = fits.getdata(rtn_params_file) # shape: (5, nx, ny) -- frames mu, A, B, d, sigma
    smoothed_stack = get_rolling_mean_stack(image_stack, window_size=10)
    diff_stack = get_diff_stack(image_stack, smoothed_stack, window_size=10)
    corrected_stack = correct_signal_vals(diff_stack, image_stack, smoothed_stack, rtn_params, adu)
    old_stats, old_rn_array = get_read_noise_stats(image_stack, adu)
    new_stats, new_rn_array = get_read_noise_stats(corrected_stack, adu)
    print(old_stats)
    print(new_stats)