#!/usr/bin/env python
"""
Apply RTN correction to a stack of frames.

Usage:
    python rtn_fixer.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -w 10
    python rtn_fixer.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -m 3
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


def build_frame_manifest(files, max_frames=None):
    """
    Build a manifest of frames across files.

    Args:
        files: List of FITS file paths
        max_frames: Maximum number of frames to include (None = all frames)

    Returns:
        list of (file_path, frame_index, output_name) tuples
    """
    manifest = []
    frame_count = 0

    for file_path in files:
        with fits.open(file_path) as hdul:
            data_shape = hdul[0].data.shape

            if len(data_shape) == 2:
                # Single 2D frame
                manifest.append((file_path, None, file_path.name))
                frame_count += 1
            elif len(data_shape) == 3:
                # 3D cube - each slice is a separate frame
                n_frames = data_shape[0]
                base_name = file_path.stem
                ext = file_path.suffix
                for i in range(n_frames):
                    output_name = f"{base_name}_frame{i:04d}{ext}"
                    manifest.append((file_path, i, output_name))
                    frame_count += 1
                    # Stop if we've reached the frame limit
                    if max_frames is not None and frame_count >= max_frames:
                        break
            else:
                raise ValueError(f"Unexpected data shape {data_shape} in {file_path}. Expected 2D or 3D arrays.")

        # Stop if we've reached the frame limit
        if max_frames is not None and frame_count >= max_frames:
            break

    return manifest


def load_frame(path, frame_index=None):
    """
    Load a single FITS frame.

    Args:
        path: Path to FITS file
        frame_index: If the file is a 3D cube, which frame to load (None for 2D files)

    Returns:
        frame: 2D numpy array
        header: FITS header
    """
    with fits.open(path) as hdul:
        data = hdul[0].data
        header = hdul[0].header.copy()

        if frame_index is not None:
            # Extract specific frame from cube
            frame = data[frame_index].astype(np.float64)
            # Add cube frame info to header
            header['CUBEIDX'] = frame_index
            header['CUBEFRMS'] = data.shape[0]
        else:
            # Single 2D frame
            frame = data.astype(np.float64)

        return frame, header


def main():
    parser = argparse.ArgumentParser(description='Apply RTN correction to frames')
    parser.add_argument('params_folder', help='Path to folder containing rtn_params.fits and median_bias_frame.fits')
    parser.add_argument('input_folder', help='Folder containing FITS frames')
    parser.add_argument('gain', type=float, help='Sensor gain in ADU/e-')
    parser.add_argument('-o', '--output', default=None,
                        help='Output folder (default: input_folder/corrected/)')
    parser.add_argument('-w', '--window', type=int, default=None,
                        help='Rolling median window size (temporal)')
    parser.add_argument('-m', '--median-size', type=int, default=None,
                        help='Median filter kernel size (spatial)')
    parser.add_argument('--frames', type=int, default=None,
                        help='Number of frames to process (default: all available)')
    parser.add_argument('-v', '--verbose', action='store_true')
    args = parser.parse_args()

    # Load RTN parameters
    if args.verbose:
        print(f"Loading RTN params from {args.params_folder}")
    with fits.open(os.path.join(args.params_folder, 'rtn_params.fits')) as hdul:
        rtn_params = hdul[0].data
    with fits.open(os.path.join(args.params_folder, 'mean_bias_frame.fits')) as hdul:
        old_mean_bias = hdul[0].data
    with fits.open(os.path.join(args.params_folder, 'read_noise_frame.fits')) as hdul:
        read_noise_frame = hdul[0].data

    # Pre-compute constants
    e_per_adu = 1.0 / args.gain
    rtn_mask = ~np.isnan(rtn_params[0])
    print(np.sum(rtn_mask), "RTN pixels detected.")
    delta_x_arr_e = np.ascontiguousarray(rtn_params[3], dtype=np.float64)
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

    # Get file list and build frame manifest (only up to max_frames if specified)
    files = get_fits_files(args.input_folder)
    manifest = build_frame_manifest(files, max_frames=args.frames)
    n_frames = len(manifest)

    if args.verbose:
        if args.frames is None:
            print(f"Found {len(files)} FITS file(s) containing {n_frames} total frame(s)")
            print(f"Processing all {n_frames} frames")
        else:
            print(f"Found at least {n_frames} frame(s) in the input files")
            print(f"Processing first {n_frames} frames")

    half_w = args.window // 2 if use_rolling_median else 0

    if args.verbose:
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

    out_folder = Path(args.output) if args.output is not None else Path(args.input_folder) / "corrected"
    out_folder.mkdir(parents=True, exist_ok=True)

    if use_rolling_median:
        # Initialize rolling window buffer
        window_buffer = deque(maxlen=2 * half_w + 1)

        for i in range(n_frames):
            file_path, frame_idx, output_name = manifest[i]
            frame, header = load_frame(file_path, frame_idx)
            window_buffer.append(frame)

            center_idx = i - half_w
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

            center_file, center_frame_idx, center_output_name = manifest[center_idx]
            _, center_header = load_frame(center_file, center_frame_idx)
            center_header['RTNCORR'] = True
            center_header['RTNWIN'] = args.window
            center_header['RTNREF'] = 'rolling_median'

            out_path = out_folder / center_output_name
            fits.writeto(out_path, corrected.astype(np.int16), center_header, overwrite=True)

            if args.verbose:
                print(f"Corrected {center_idx + 1}/{n_frames}: {center_output_name}")

        # Copy edge frames uncorrected
        for i in list(range(half_w)) + list(range(n_frames - half_w, n_frames)):
            file_path, frame_idx, output_name = manifest[i]
            frame, header = load_frame(file_path, frame_idx)
            header['RTNCORR'] = False
            out_path = out_folder / output_name
            fits.writeto(out_path, frame.astype(np.int16), header, overwrite=True)
            if args.verbose:
                print(f"Copied (edge) {i + 1}/{n_frames}: {output_name}")

    else:  # median_filter
        for i in range(n_frames):
            file_path, frame_idx, output_name = manifest[i]
            frame, header = load_frame(file_path, frame_idx)

            reference = median_filter(frame, size=args.median_size)

            corrected, num_corr_arr = _correct_frame(
                frame, reference, rtn_mask,
                delta_x_arr_e, mu_e, sigma_r_arr,
                e_per_adu, num_corr_arr, read_noise_frame
            )

            header['RTNCORR'] = True
            header['RTNREF'] = 'median_filter'
            header['RTNMEDSZ'] = args.median_size

            out_path = out_folder / output_name
            fits.writeto(out_path, corrected.astype(np.int16), header, overwrite=True)

            if args.verbose:
                print(f"Corrected {i + 1}/{n_frames}: {output_name}")

    print(f"Done. Processed {n_frames} frames.")
    frac_high_corrections = num_corr_arr[1] / n_frames / (1 - rtn_params[1] - rtn_params[2])
    frac_low_corrections = num_corr_arr[0] / n_frames / rtn_params[2]
    print(np.median(frac_high_corrections[rtn_mask]), np.median(frac_low_corrections[rtn_mask]))
    updated_bias_frame = old_mean_bias + np.nan_to_num(rtn_params[3]) / e_per_adu * (num_corr_arr[0] - num_corr_arr[1]) / n_frames
    correction_values = np.nan_to_num(rtn_params[3]) / e_per_adu * (num_corr_arr[0] - num_corr_arr[1]) / n_frames
    # new_mean_bias = (rtn_params[0] * (rtn_params[1] + frac_high_corrections * (1 - rtn_params[1] - rtn_params[2]) + frac_low_corrections * rtn_params[2]) +
    #                 (rtn_params[0] - rtn_params[3]) * rtn_params[2] * (1 - frac_low_corrections) +
    #                 (rtn_params[0] + rtn_params[3]) * (1 - rtn_params[1] - rtn_params[2]) * (1 - frac_high_corrections))
    # updated_bias_frame = old_mean_bias * (1 - rtn_mask) + np.nan_to_num(new_mean_bias * rtn_mask / e_per_adu)
    # import matplotlib.pyplot as plt
    # plt.hist(frac_high_corrections.flatten(), bins=50, alpha=0.5, label='High-level corrections', range=(0, 2))
    # plt.hist(frac_low_corrections.flatten(), bins=50, alpha=0.5, label='Low-level corrections', range=(0, 2))
    # plt.xlabel('Num corrections/expected corrections')
    # plt.ylabel('Number of pixels')
    # plt.legend()
    # plt.show()
    # print(np.nanmedian(frac_high_corrections), np.nanmedian(frac_low_corrections))
    print(np.mean(correction_values[rtn_mask]))
    # print(np.mean(updated_bias_frame[~rtn_mask] - old_mean_bias[~rtn_mask]))
    # Make a new folder in the output folder for the updated bias frame
    # updated_bias_folder = out_folder / 'updated_bias_frame'
    # updated_bias_folder.mkdir(exist_ok=True)
    # fits.writeto(updated_bias_folder / 'updated_bias_frame.fits', updated_bias_frame.astype(np.float32), overwrite=True)
    correction_values_folder = out_folder / 'correction_values_frame'
    correction_values_folder.mkdir(exist_ok=True)
    fits.writeto(correction_values_folder / 'correction_values_frame.fits', correction_values.astype(np.float32), overwrite=True)


if __name__ == '__main__':
    main()