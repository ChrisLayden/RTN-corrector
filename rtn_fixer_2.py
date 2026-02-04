#!/usr/bin/env python
"""
Apply RTN correction to a stack of frames.

Usage:
    python correct_rtn.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -w 10
    python correct_rtn.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -m 3
"""
import os
import argparse
from pathlib import Path
from collections import deque
import numpy as np
from numba import njit
from scipy.ndimage import median_filter, convolve
from astropy.io import fits
from scripts.make_lut import ThresholdLUT

kernel = np.array([[1, 1, 1],
                   [1, 0, 1],
                   [1, 1, 1]]) / 8.0

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
                   sigma_r_arr, e_per_adu, num_corr_arr, read_noise_arr, std_reference=None):
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
            std = 0.0 if std_reference is None else std_reference[y, x] * e_per_adu
            # If std is larger than expected for shot noise and read noise, skip correction,
            # as source variability is likely present.
            if std > 1.5 * np.sqrt(lam + read_noise_arr[y, x]**2):
                num_skipped += 1
                continue
            delta = delta_x_arr_e[y, x]
            low_lo, low_hi, high_lo, high_hi = get_thresholds_numba(
                lam, sigma_r_arr[y, x], delta,
                lam_vals, rn_vals,
                central_low_data, central_high_data
            )
            # Check if we were out of the bounds of the interpolation table
            if np.isnan(low_lo):
                num_skipped += 1
                continue

            corr = 0.0
            diff = frame[y, x]  * e_per_adu - mu_e[y, x]
            if diff > high_lo and diff < high_hi:
                # Check if lambda is too big, such that the peaks overlap too much
                if high_lo >= delta:
                    continue
                num_corr += 1
                corr = -delta
                num_corr_arr[1, y, x] += 1
            elif diff + lam < low_hi and diff + lam > low_lo:
                # Check if lambda is too big, such that the peaks overlap too much
                if low_hi <= -delta:
                    continue
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
    parser.add_argument('params_folder', help='Path to folder containing rtn_params.fits and median_bias_frame.fits')
    parser.add_argument('input_folder', help='Folder containing FITS frames')
    parser.add_argument('gain', type=float, help='Sensor gain in ADU/e-')
    parser.add_argument('-o', '--output', default='./corrected/',
                        help='Output folder (default: ./corrected/)')
    parser.add_argument('-w', '--window', type=int, default=None,
                        help='Rolling median window size (temporal)')
    parser.add_argument('-m', '--median-size', type=int, default=None,
                        help='Median filter kernel size (spatial)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Load RTN parameters
    if args.verbose:
        print(f"Loading RTN params from {args.params_folder}")
    with fits.open(os.path.join(args.params_folder, 'rtn_params.fits')) as hdul:
        rtn_params = hdul[0].data
    # with fits.open(os.path.join(args.params_folder, 'median_bias_frame.fits')) as hdul:
    #     med_bias = hdul[0].data
    with fits.open(os.path.join(args.params_folder, 'read_noise_frame.fits')) as hdul:
        read_noise_frame = hdul[0].data

    # Pre-compute constants
    e_per_adu = 1.0 / args.gain
    rtn_mask = ~np.isnan(rtn_params[0])
    print(np.sum(rtn_mask), "RTN pixels detected.")
    delta_x_arr_e = np.ascontiguousarray(rtn_params[3], dtype=np.float64)
    # mu_e = ((rtn_params[0] * rtn_params[1]) +
    #         (rtn_params[0] - rtn_params[3]) * rtn_params[2] +
    #         (rtn_params[0] + rtn_params[3]) * (1 - rtn_params[1] - rtn_params[2]))
    mu_e = np.ascontiguousarray(rtn_params[0], dtype=np.float64)
    sigma_r_arr = np.ascontiguousarray(rtn_params[4], dtype=np.float64)
    num_corr_arr = np.zeros((2, *rtn_mask.shape), dtype=np.int32)
    read_noise_frame = np.ascontiguousarray(read_noise_frame, dtype=np.float64)

    # Validate reference method arguments
    if args.window is not None and args.median_size is not None:
        parser.error("Specify either -w (rolling median) or -m (spatial median filter), not both")
    if args.window is None and args.median_size is None:
        parser.error("Must specify either -w (rolling median) or -m (spatial median filter)")

    use_rolling_median = args.window is not None

    # Get file list
    files = get_fits_files(args.input_folder)
    n_frames = len(files)
    half_w = args.window // 2 if use_rolling_median else 0
    
    if args.verbose:
        print(f"Found {n_frames} frames")
        if use_rolling_median:
            print(f"Reference method: rolling median (window={args.window})")
        else:
            print(f"Reference method: spatial median filter (size={args.median_size})")

    # JIT warmup
    if args.verbose:
        print("Compiling JIT...")
    dummy = np.zeros((10, 10), dtype=np.float64)
    _ = _correct_frame(dummy, dummy, rtn_mask[:10, :10], 
                       delta_x_arr_e[:10, :10], mu_e[:10, :10],
                       sigma_r_arr[:10, :10],
                       e_per_adu, num_corr_arr[:, :10, :10], read_noise_frame[:10, :10])

    out_folder = Path(args.output)
    out_folder.mkdir(parents=True, exist_ok=True)

    if use_rolling_median:
        # Initialize rolling window buffer
        window_buffer = deque(maxlen=2 * half_w + 1)
        
        # old_med_filter_frame = None

        for i in range(n_frames):
            frame, header = load_frame(files[i])
            window_buffer.append(frame)

            center_idx = i - half_w
            # if i == 2 * half_w - 1:
            #     # continue
            #     old_med_filter_frame = convolve(window_buffer[half_w] - med_bias, kernel)
            if i < 2 * half_w or i >= n_frames:
                continue
            
            center_frame = window_buffer[half_w]
            # Mask out non-RTN pixels in buffer for reference calculation for speed.
            # Also exclude the center frame to avoid self-biasing.
            mask = np.arange(len(window_buffer)) != half_w
            masked_buffer = np.asarray(window_buffer)[mask] * rtn_mask[np.newaxis, :, :]
            reference = np.median(masked_buffer, axis=0)
            std_reference = np.std(masked_buffer, axis=0)
            
            corrected, num_corr_arr = _correct_frame(
                center_frame, reference, rtn_mask,
                delta_x_arr_e, mu_e, sigma_r_arr,
                e_per_adu, num_corr_arr, read_noise_frame, std_reference,
            )
            # old_med_filter_frame = new_med_filter_frame

            _, center_header = load_frame(files[center_idx])
            center_header['RTNCORR'] = True
            center_header['RTNWIN'] = args.window
            center_header['RTNREF'] = 'rolling_median'
            
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
                delta_x_arr_e, mu_e, sigma_r_arr,
                e_per_adu, num_corr_arr, read_noise_frame
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
    old_mean_array = ((rtn_params[0] * rtn_params[1]) +
                     (rtn_params[0] - rtn_params[3]) * rtn_params[2] +
                     (rtn_params[0] + rtn_params[3]) * (1 - rtn_params[1] - rtn_params[2]))
    new_mean_array = (rtn_params[0] * (rtn_params[1] + frac_high_corrections * (1 - rtn_params[1] - rtn_params[2]) + frac_low_corrections * rtn_params[2]) +
                     (rtn_params[0] - rtn_params[3]) * rtn_params[2] * (1 - frac_low_corrections) +
                     (rtn_params[0] + rtn_params[3]) * (1 - rtn_params[1] - rtn_params[2]) * (1 - frac_high_corrections))
    import matplotlib.pyplot as plt
    # plt.hist(frac_high_corrections.flatten(), bins=50, alpha=0.5, label='High-level corrections', range=(0, 2))
    # plt.hist(frac_low_corrections.flatten(), bins=50, alpha=0.5, label='Low-level corrections', range=(0, 2))
    # plt.xlabel('Num corrections/expected corrections')
    # plt.ylabel('Number of pixels')
    # plt.legend()
    # plt.show()
    print(np.nanmedian(frac_high_corrections), np.nanmedian(frac_low_corrections))
    print(np.mean(new_mean_array[rtn_mask]) - np.mean(old_mean_array[rtn_mask]))
    updated_bias_frame = new_mean_array / e_per_adu
    fits.writeto(out_folder / 'updated_bias_frame.fits', updated_bias_frame.astype(np.float32), overwrite=True)


if __name__ == '__main__':
    main()