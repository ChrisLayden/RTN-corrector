#!/usr/bin/env python
"""
Apply RTN correction to a stack of frames.

Usage:
    python correct_rtn.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -w 10
"""

import argparse
from pathlib import Path
from collections import deque
import numpy as np
from numba import njit
from astropy.io import fits

def load_lut(filename):
    data = np.load(filename)
    return (data['lam'], data['rn'], data['dx'],
            data['ll'], data['lh'], data['hl'], data['hh'])


@njit
def lookup_thresholds(lam, rn, dx, lam_vals, rn_vals, dx_vals,
                      grid_ll, grid_lh, grid_hl, grid_hh):
    """
    Numba-compatible trilinear interpolation for all four thresholds.
    
    Returns (low_lo, low_hi, high_lo, high_hi).
    Returns NaNs if peaks blend (high_lo > lam + dx).
    """
    i = np.searchsorted(lam_vals, lam) - 1
    j = np.searchsorted(rn_vals, rn) - 1
    k = np.searchsorted(dx_vals, dx) - 1
    
    i = max(0, min(i, len(lam_vals) - 2))
    j = max(0, min(j, len(rn_vals) - 2))
    k = max(0, min(k, len(dx_vals) - 2))
    
    t = (lam - lam_vals[i]) / (lam_vals[i+1] - lam_vals[i])
    u = (rn - rn_vals[j]) / (rn_vals[j+1] - rn_vals[j])
    v = (dx - dx_vals[k]) / (dx_vals[k+1] - dx_vals[k])
    
    w000 = (1-t) * (1-u) * (1-v)
    w001 = (1-t) * (1-u) * v
    w010 = (1-t) * u * (1-v)
    w011 = (1-t) * u * v
    w100 = t * (1-u) * (1-v)
    w101 = t * (1-u) * v
    w110 = t * u * (1-v)
    w111 = t * u * v
    
    def interp(grid):
        return (w000 * grid[i, j, k] + w001 * grid[i, j, k+1] +
                w010 * grid[i, j+1, k] + w011 * grid[i, j+1, k+1] +
                w100 * grid[i+1, j, k] + w101 * grid[i+1, j, k+1] +
                w110 * grid[i+1, j+1, k] + w111 * grid[i+1, j+1, k+1])
    
    high_lo = interp(grid_hl)
    
    # Peaks blend if lower bound of high peak exceeds its center
    if high_lo > lam + dx:
        return np.nan, np.nan, np.nan, np.nan
    
    return interp(grid_ll), interp(grid_lh), high_lo, interp(grid_hh)

lut_lam, lut_rn, lut_dx, lut_ll, lut_lh, lut_hl, lut_hh = load_lut('rts_lut.npz')

@njit
def _correct_frame(frame, rolling_mean, rtn_mask, delta_x_arr_e, mu_e, 
                   read_noise_e, e_per_adu, num_corr_arr):
    ny, nx = frame.shape
    corrected = frame.copy()
    num_corr = 0
    num_skipped = 0
    # Looping turns out to be faster than vectorizing here because of the sparseness
    for y in range(ny):
        for x in range(nx):
            if not rtn_mask[y, x]:
                continue

            lam = rolling_mean[y, x] * e_per_adu - mu_e[y, x]
            if lam < 0:
                lam = 0
            noise = np.sqrt(lam + read_noise_e[y, x]**2)
            delta = delta_x_arr_e[y, x]
            low_lo, low_hi, high_lo, high_hi = lookup_thresholds(
                lam, read_noise_e[y, x], delta,
                lut_lam, lut_rn, lut_dx,
                lut_ll, lut_lh, lut_hl, lut_hh
            )
            if np.isnan(low_lo):
                num_skipped += 1
                continue
            # print(high_lo, max(delta - thr_noise, thr_noise), lam, read_noise_e[y, x], delta)
            # high_lo = max(lam + delta - thr_noise, lam + thr_noise)
            # high_hi = lam + delta + thr_noise
            # low_lo = lam - delta - thr_noise
            # low_hi = min(lam - delta + thr_noise, lam - thr_noise)

            corr = 0.0
            diff = (frame[y, x] - rolling_mean[y, x]) * e_per_adu
            # print(diff, high_lo, high_hi, low_lo, low_hi)
            if diff > high_lo and diff < high_hi:
                num_corr += 1
                corr = -delta
                num_corr_arr[1, y, x] += 1
            elif diff < low_hi and diff > low_lo:
                num_corr += 1
                corr = delta
                num_corr_arr[0, y, x] += 1

            corrected[y, x] = frame[y, x] + round(corr / e_per_adu)
    # print(num_corr, num_skipped)
    return corrected, num_corr_arr


def get_fits_files(folder):
    """Get sorted list of FITS files."""
    folder = Path(folder)
    files = sorted(folder.glob('*.fits')) + sorted(folder.glob('*.fit'))
    if not files:
        raise FileNotFoundError(f"No FITS files found in {folder}")
    return files


def load_frame(path):
    """Load a single FITS frame."""
    with fits.open(path) as hdul:
        return hdul[0].data.astype(np.float64), hdul[0].header


def main():
    parser = argparse.ArgumentParser(description='Apply RTN correction to frames')
    parser.add_argument('rtn_params', help='Path to rtn_params.fits')
    parser.add_argument('input_folder', help='Folder containing FITS frames')
    parser.add_argument('gain', type=float, help='Sensor gain in ADU/e-')
    parser.add_argument('-o', '--output', default='./corrected/',
                        help='Output folder (default: ./corrected/)')
    parser.add_argument('-w', '--window', type=int, default=10,
                        help='Rolling window size (default: 10)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Load RTN parameters
    if args.verbose:
        print(f"Loading RTN params from {args.rtn_params}")
    with fits.open(args.rtn_params) as hdul:
        rtn_params = hdul[0].data

    # Pre-compute constants
    e_per_adu = 1.0 / args.gain
    rtn_mask = ~np.isnan(rtn_params[0])
    print(np.sum(rtn_mask), "RTN pixels detected.")
    delta_x_arr_e = np.ascontiguousarray(rtn_params[3] * e_per_adu)
    mu_e = e_per_adu * ((rtn_params[0] * rtn_params[1]) +
                        (rtn_params[0] - rtn_params[3]) * rtn_params[2] +
                        (rtn_params[0] + rtn_params[3]) * (1 - rtn_params[1] - rtn_params[2]))
    mu_e = np.ascontiguousarray(mu_e)
    read_noise_e = np.ascontiguousarray(rtn_params[4] * e_per_adu)
    num_corr_arr = np.zeros((2, *rtn_mask.shape), dtype=np.int32)  # For statistics

    # Get file list
    files = get_fits_files(args.input_folder)
    n_frames = len(files)
    half_w = args.window // 2
    
    if args.verbose:
        print(f"Found {n_frames} frames")

    # JIT warmup
    if args.verbose:
        print("Compiling JIT...")
    dummy = np.zeros((10, 10), dtype=np.float64)
    _ = _correct_frame(dummy, dummy, rtn_mask[:10, :10], 
                       delta_x_arr_e[:10, :10], mu_e[:10, :10],
                       read_noise_e[:10, :10],
                       e_per_adu, num_corr_arr[:, :10, :10])

    out_folder = Path(args.output)
    out_folder.mkdir(parents=True, exist_ok=True)

    # Initialize rolling window buffer
    window_buffer = deque(maxlen=2 * half_w + 1)
    rolling_sum = None

    # First pass: fill initial window and process
    for i in range(n_frames):
        frame, header = load_frame(files[i])
        
        # Add to rolling buffer
        if rolling_sum is None:
            rolling_sum = np.zeros_like(frame)
        
        if len(window_buffer) == window_buffer.maxlen:
            rolling_sum -= window_buffer[0]
        window_buffer.append(frame)
        rolling_sum += frame

        # Process frame at center of window
        center_idx = i - half_w
        if center_idx < 0 or center_idx >= n_frames - half_w:
            continue

        rolling_mean = rolling_sum / len(window_buffer)
        center_frame = window_buffer[half_w]
        
        corrected, num_corr_arr = _correct_frame(
            center_frame, rolling_mean, rtn_mask,
            delta_x_arr_e, mu_e, read_noise_e,
            e_per_adu, num_corr_arr
        )

        # Load header for center frame and save
        _, center_header = load_frame(files[center_idx])
        center_header['RTNCORR'] = True
        center_header['RTNWIN'] = args.window
        
        out_path = out_folder / files[center_idx].name
        fits.writeto(out_path, corrected.astype(np.int16), center_header, overwrite=True)
        
        if args.verbose:
            print(f"Corrected {center_idx + 1}/{n_frames}: {files[center_idx].name}")

    # Copy edge frames uncorrected
    for i in list(range(half_w)) + list(range(n_frames - half_w, n_frames)):
        frame, header = load_frame(files[i])
        header['RTNCORR'] = False
        out_path = out_folder / files[i].name
        fits.writeto(out_path, frame.astype(np.int16), header, overwrite=True)
        if args.verbose:
            print(f"Copied (edge) {i + 1}/{n_frames}: {files[i].name}")

    print(f"Done. Processed {n_frames} frames.")
    frac_high_corrections = num_corr_arr[1] / len(files) / (1 - rtn_params[1] - rtn_params[2])
    frac_low_corrections = num_corr_arr[0] / len(files) / rtn_params[2]
    import matplotlib.pyplot as plt
    plt.hist(frac_high_corrections.flatten(), bins=50, alpha=0.5, label='High-level corrections', range=(0, 2))
    plt.hist(frac_low_corrections.flatten(), bins=50, alpha=0.5, label='Low-level corrections', range=(0, 2))
    plt.xlabel('Num corrections/expected corrections')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.show()
    print(np.nanmedian(frac_high_corrections), np.nanmedian(frac_low_corrections))


if __name__ == '__main__':
    main()