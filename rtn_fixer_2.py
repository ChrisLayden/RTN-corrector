#!/usr/bin/env python
"""
Apply RTN correction to a stack of frames.

Usage:
    python correct_rtn.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -w 10
    python correct_rtn.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -m 3
"""

import argparse
from pathlib import Path
from collections import deque
import numpy as np
from numba import njit
from scipy.ndimage import median_filter
from astropy.io import fits
from scripts.make_lut import ThresholdLUT

@njit
def interp2d(lam, rn, lam_vals, rn_vals, data):
    """Bilinear interpolation on regular grid."""
    # Find indices
    i = np.searchsorted(lam_vals, lam) - 1
    j = np.searchsorted(rn_vals, rn) - 1
    
    # Clamp to valid range
    i = max(0, min(i, len(lam_vals) - 2))
    j = max(0, min(j, len(rn_vals) - 2))
    
    # Fractional position within cell
    t = (lam - lam_vals[i]) / (lam_vals[i+1] - lam_vals[i])
    u = (rn - rn_vals[j]) / (rn_vals[j+1] - rn_vals[j])
    
    # Clamp fractions
    t = max(0.0, min(1.0, t))
    u = max(0.0, min(1.0, u))
    
    # Bilinear interpolation
    c00 = data[i, j]
    c01 = data[i, j+1]
    c10 = data[i+1, j]
    c11 = data[i+1, j+1]
    
    return (c00 * (1-t) * (1-u) +
            c01 * (1-t) * u +
            c10 * t * (1-u) +
            c11 * t * u)


@njit
def get_thresholds_numba(lam, rn, dx, lam_vals, rn_vals,
                         central_low_data, central_high_data):
    """Get all four peak thresholds from central thresholds + delta_x shift."""
    central_low = interp2d(lam, rn, lam_vals, rn_vals, central_low_data)
    central_high = interp2d(lam, rn, lam_vals, rn_vals, central_high_data)
    
    high_peak_high = central_high + dx
    high_peak_low = max(central_low + dx, central_high)
    low_peak_low = central_low - dx
    low_peak_high = min(central_high - dx, central_low)
    
    return low_peak_low, low_peak_high, high_peak_low, high_peak_high

# Extract once at setup for numba-compatible interpolation
lut = ThresholdLUT.load('rts_threshold_lut.pkl')
lam_vals = lut.lam_vals
rn_vals = lut.rn_vals
central_low_data = lut.central_low
central_high_data = lut.central_high

@njit
def _correct_frame(frame, reference, rtn_mask, delta_x_arr_e, mu_e, 
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

            lam = reference[y, x] * e_per_adu - mu_e[y, x]
            if lam < 0:
                lam = 0
            noise = np.sqrt(lam + read_noise_e[y, x]**2)
            delta = delta_x_arr_e[y, x]
            low_lo, low_hi, high_lo, high_hi = get_thresholds_numba(
                lam, read_noise_e[y, x], delta,
                lam_vals, rn_vals,
                central_low_data, central_high_data
            )
            if np.isnan(low_lo):
                num_skipped += 1
                continue

            corr = 0.0
            diff = (frame[y, x] - reference[y, x]) * e_per_adu
            if diff > high_lo and diff < high_hi:
                num_corr += 1
                corr = -delta
                num_corr_arr[1, y, x] += 1
            elif diff < low_hi and diff > low_lo:
                num_corr += 1
                corr = delta
                num_corr_arr[0, y, x] += 1

            corrected[y, x] = frame[y, x] + round(corr / e_per_adu)
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
    parser.add_argument('-w', '--window', type=int, default=None,
                        help='Rolling mean window size (temporal)')
    parser.add_argument('-m', '--median-size', type=int, default=None,
                        help='Median filter kernel size (spatial)')
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
    num_corr_arr = np.zeros((2, *rtn_mask.shape), dtype=np.int32)

    # Validate reference method arguments
    if args.window is not None and args.median_size is not None:
        parser.error("Specify either -w (rolling mean) or -m (median filter), not both")
    if args.window is None and args.median_size is None:
        parser.error("Must specify either -w (rolling mean) or -m (median filter)")

    use_rolling_mean = args.window is not None

    # Get file list
    files = get_fits_files(args.input_folder)
    n_frames = len(files)
    half_w = args.window // 2 if use_rolling_mean else 0
    
    if args.verbose:
        print(f"Found {n_frames} frames")
        if use_rolling_mean:
            print(f"Reference method: rolling mean (window={args.window})")
        else:
            print(f"Reference method: median filter (size={args.median_size})")

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

    if use_rolling_mean:
        # Initialize rolling window buffer
        window_buffer = deque(maxlen=2 * half_w + 1)
        rolling_sum = None

        for i in range(n_frames):
            frame, header = load_frame(files[i])
            
            if rolling_sum is None:
                rolling_sum = np.zeros_like(frame)
            
            if len(window_buffer) == window_buffer.maxlen:
                rolling_sum -= window_buffer[0]
            window_buffer.append(frame)
            rolling_sum += frame

            center_idx = i - half_w
            if center_idx < 0 or center_idx >= n_frames - half_w:
                continue

            # reference = rolling_sum / len(window_buffer)
            reference = np.median(window_buffer, axis=0)
            center_frame = window_buffer[half_w]
            
            corrected, num_corr_arr = _correct_frame(
                center_frame, reference, rtn_mask,
                delta_x_arr_e, mu_e, read_noise_e,
                e_per_adu, num_corr_arr
            )

            _, center_header = load_frame(files[center_idx])
            center_header['RTNCORR'] = True
            center_header['RTNWIN'] = args.window
            center_header['RTNREF'] = 'rolling_mean'
            
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

    else:  # median_filter
        for i, fpath in enumerate(files):
            frame, header = load_frame(fpath)
            
            reference = median_filter(frame, size=args.median_size)
            
            corrected, num_corr_arr = _correct_frame(
                frame, reference, rtn_mask,
                delta_x_arr_e, mu_e, read_noise_e,
                e_per_adu, num_corr_arr
            )

            header['RTNCORR'] = True
            header['RTNREF'] = 'median_filter'
            header['RTNMEDSZ'] = args.median_size
            
            out_path = out_folder / fpath.name
            fits.writeto(out_path, corrected.astype(np.int16), header, overwrite=True)
            
            if args.verbose:
                print(f"Corrected {i + 1}/{n_frames}: {fpath.name}")

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