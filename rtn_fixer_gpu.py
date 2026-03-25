#!/usr/bin/env python
"""
GPU-accelerated RTN correction using CuPy.

Optimizations over naive GPU port:
  - Sparse dispatch: correction kernel only runs on RTN pixels (~1-5% of sensor)
  - Rolling median computed only at RTN pixel locations with pre-allocated ring buffer

Usage:
    python rtn_fixer_gpu.py ./params_folder/ ./frames/ 0.5 -o ./corrected/ -w 10
    python rtn_fixer_gpu.py ./params_folder/ ./frames/ 0.5 -o ./corrected/ -m 3
"""
import os
import argparse
from pathlib import Path
from collections import deque
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import median_filter as gpu_median_filter
from astropy.io import fits
from scripts.make_lut import ThresholdLUT

# Extract LUT data once at module level for GPU transfer
lut = ThresholdLUT.load('rts_threshold_lut.pkl')
lam_vals = lut.lam_vals
rn_vals = lut.rn_vals
central_low_data = lut.central_low
central_high_data = lut.central_high

# GPU kernel operating on sparse RTN pixels only.
# All inputs are 1D arrays of length n_rtn (the number of RTN pixels).
# When use_std_ref == 0, the std_reference check is skipped.
_correct_sparse_kernel = cp.ElementwiseKernel(
    in_params='''
        float64 frame_val, float64 ref_val, float64 std_ref_val,
        float64 delta_x, float64 mu, float64 sigma_r, float64 read_noise,
        float64 e_per_adu, int32 use_std_ref,
        raw float64 lam_arr, raw float64 rn_arr,
        raw float64 cl_data, raw float64 ch_data,
        int32 n_lam, int32 n_rn
    ''',
    out_params='float64 corrected, int32 corr_type',
    operation='''
        corrected = frame_val;
        corr_type = 0;  // 0 = none, 1 = low-peak (+delta), 2 = high-peak (-delta)

        // Compute lambda (expected signal in electrons)
        double lam = ref_val * e_per_adu - mu;
        if (lam < 0.0) lam = 0.0;

        // Source variability check
        if (use_std_ref) {
            double std_e = std_ref_val * e_per_adu;
            double expected_std = 1.5 * sqrt(lam + read_noise * read_noise);
            if (std_e > expected_std) {
                return;
            }
        }

        // Bilinear interpolation indices for lambda
        int il = 0;
        for (int k = 0; k < n_lam - 1; k++) {
            if (lam_arr[k + 1] > lam) break;
            il = k + 1;
        }
        if (il >= n_lam - 1) il = n_lam - 2;
        if (il < 0) il = 0;

        // Bilinear interpolation indices for read noise (sigma_r)
        int jr = 0;
        for (int k = 0; k < n_rn - 1; k++) {
            if (rn_arr[k + 1] > sigma_r) break;
            jr = k + 1;
        }
        if (jr >= n_rn - 1) jr = n_rn - 2;
        if (jr < 0) jr = 0;

        // Interpolation weights
        double t = (lam - lam_arr[il]) / (lam_arr[il + 1] - lam_arr[il]);
        double u_w = (sigma_r - rn_arr[jr]) / (rn_arr[jr + 1] - rn_arr[jr]);
        if (t < 0.0) t = 0.0; if (t > 1.0) t = 1.0;
        if (u_w < 0.0) u_w = 0.0; if (u_w > 1.0) u_w = 1.0;

        // 2D grid indices into flattened LUT
        int idx00 = il * n_rn + jr;
        int idx01 = il * n_rn + (jr + 1);
        int idx10 = (il + 1) * n_rn + jr;
        int idx11 = (il + 1) * n_rn + (jr + 1);

        double central_low  = cl_data[idx00] * (1-t) * (1-u_w)
                             + cl_data[idx01] * (1-t) * u_w
                             + cl_data[idx10] * t * (1-u_w)
                             + cl_data[idx11] * t * u_w;

        double central_high = ch_data[idx00] * (1-t) * (1-u_w)
                             + ch_data[idx01] * (1-t) * u_w
                             + ch_data[idx10] * t * (1-u_w)
                             + ch_data[idx11] * t * u_w;

        // Check for NaN (out of LUT bounds)
        if (central_low != central_low) return;

        // Compute peak thresholds
        double high_peak_high = central_high + delta_x;
        double high_peak_low  = central_low + delta_x;
        if (high_peak_low < central_high) high_peak_low = central_high;

        double low_peak_low  = central_low - delta_x;
        double low_peak_high = central_high - delta_x;
        if (low_peak_high > central_low) low_peak_high = central_low;

        // Pixel deviation from mean in electrons
        double diff = frame_val * e_per_adu - mu;

        if (diff > high_peak_low && diff < high_peak_high) {
            if (high_peak_low >= delta_x) return;
            corrected = frame_val + round(-delta_x / e_per_adu);
            corr_type = 2;
        } else if ((diff + lam) < low_peak_high && (diff + lam) > low_peak_low) {
            if (low_peak_high <= -delta_x) return;
            corrected = frame_val + round(delta_x / e_per_adu);
            corr_type = 1;
        }
    ''',
    name='correct_sparse_gpu'
)


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
                    if max_frames is not None and frame_count >= max_frames:
                        break
            else:
                raise ValueError(f"Unexpected data shape {data_shape} in {file_path}. Expected 2D or 3D arrays.")

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
            frame = data[frame_index].astype(np.float64)
            header['CUBEIDX'] = frame_index
            header['CUBEFRMS'] = data.shape[0]
        else:
            frame = data.astype(np.float64)

        return frame, header


def main():
    parser = argparse.ArgumentParser(description='GPU-accelerated RTN correction')
    parser.add_argument('params_folder', help='Path to folder containing rtn_params.fits and mean_bias_frame.fits')
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

    # Validate reference method arguments
    if args.window is not None and args.median_size is not None:
        parser.error("Specify either -w (rolling median) or -m (spatial median filter), not both")
    if args.window is None and args.median_size is None:
        parser.error("Must specify either -w (rolling median) or -m (spatial median filter)")

    use_rolling_median = args.window is not None

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
    n_rtn = int(np.sum(rtn_mask))
    print(f"{n_rtn} RTN pixels detected.")

    # Flat indices of RTN pixels for sparse gather/scatter
    rtn_flat_idx = np.flatnonzero(rtn_mask)
    rtn_flat_idx_gpu = cp.asarray(rtn_flat_idx)

    # Pre-gather per-RTN-pixel parameters (1D arrays, length n_rtn)
    delta_x_arr_e = np.ascontiguousarray(rtn_params[3], dtype=np.float64)
    mu_e = np.ascontiguousarray(rtn_params[0], dtype=np.float64)
    sigma_r_arr = np.ascontiguousarray(rtn_params[4], dtype=np.float64)
    read_noise_frame = np.ascontiguousarray(read_noise_frame, dtype=np.float64)

    delta_x_rtn_gpu = cp.asarray(delta_x_arr_e.ravel()[rtn_flat_idx])
    mu_rtn_gpu = cp.asarray(mu_e.ravel()[rtn_flat_idx])
    sigma_r_rtn_gpu = cp.asarray(sigma_r_arr.ravel()[rtn_flat_idx])
    read_noise_rtn_gpu = cp.asarray(read_noise_frame.ravel()[rtn_flat_idx])

    # Transfer LUT to GPU (flattened for raw access in kernel)
    lam_vals_gpu = cp.asarray(lam_vals, dtype=cp.float64)
    rn_vals_gpu = cp.asarray(rn_vals, dtype=cp.float64)
    central_low_gpu = cp.asarray(central_low_data.ravel(), dtype=cp.float64)
    central_high_gpu = cp.asarray(central_high_data.ravel(), dtype=cp.float64)
    n_lam = np.int32(len(lam_vals))
    n_rn_lut = np.int32(len(rn_vals))

    # Correction counters — sparse (n_rtn,), scattered back to full 2D at the end
    num_corr_low_rtn_gpu = cp.zeros(n_rtn, dtype=cp.int32)
    num_corr_high_rtn_gpu = cp.zeros(n_rtn, dtype=cp.int32)

    # Get file list and build frame manifest
    files = get_fits_files(args.input_folder)
    manifest = build_frame_manifest(files, max_frames=args.frames)
    n_frames = len(manifest)
    half_w = args.window // 2 if use_rolling_median else 0

    if args.verbose:
        if args.frames is None:
            print(f"Found {len(files)} FITS file(s) containing {n_frames} total frame(s)")
            print(f"Processing all {n_frames} frames")
        else:
            print(f"Found at least {n_frames} frame(s) in the input files")
            print(f"Processing first {n_frames} frames")
        if use_rolling_median:
            print(f"Reference method: rolling median (window={args.window})")
        else:
            print(f"Reference method: spatial median filter (size={args.median_size})")

    out_folder = Path(args.output) if args.output is not None else Path(args.input_folder) / "corrected"
    out_folder.mkdir(parents=True, exist_ok=True)

    # Helper: run sparse correction kernel and accumulate stats
    def correct_sparse(frame_rtn_gpu, ref_rtn_gpu, std_rtn_gpu=None):
        use_std = np.int32(1 if std_rtn_gpu is not None else 0)
        if std_rtn_gpu is None:
            std_rtn_gpu = cp.zeros(n_rtn, dtype=cp.float64)

        corrected_rtn, corr_type = _correct_sparse_kernel(
            frame_rtn_gpu, ref_rtn_gpu, std_rtn_gpu,
            delta_x_rtn_gpu, mu_rtn_gpu, sigma_r_rtn_gpu, read_noise_rtn_gpu,
            e_per_adu, use_std,
            lam_vals_gpu, rn_vals_gpu,
            central_low_gpu, central_high_gpu,
            n_lam, n_rn_lut
        )
        num_corr_low_rtn_gpu[...] += (corr_type == 1).astype(cp.int32)
        num_corr_high_rtn_gpu[...] += (corr_type == 2).astype(cp.int32)
        return corrected_rtn

    # Helper: scatter sparse corrections back into full frame on GPU
    def scatter_to_frame(frame_gpu, corrected_rtn_gpu):
        corrected_gpu = frame_gpu.copy()
        corrected_gpu.ravel()[rtn_flat_idx_gpu] = corrected_rtn_gpu
        return corrected_gpu

    if use_rolling_median:
        window_size = 2 * half_w + 1

        # Pre-allocated ring buffer: only RTN pixel values, shape (window_size, n_rtn)
        ring_buffer_gpu = cp.zeros((window_size, n_rtn), dtype=cp.float64)
        # Deque of full frames (GPU) needed for writing uncorrected output and
        # for the full-frame scatter at correction time
        frame_buffer = deque(maxlen=window_size)
        ring_pos = 0  # current write position in ring buffer

        for i in range(n_frames):
            file_path, frame_idx, output_name = manifest[i]
            frame, header = load_frame(file_path, frame_idx)
            frame_gpu = cp.asarray(frame)

            # Extract RTN pixels and write into ring buffer slot
            frame_rtn = frame_gpu.ravel()[rtn_flat_idx_gpu]
            ring_buffer_gpu[ring_pos % window_size] = frame_rtn
            ring_pos += 1
            frame_buffer.append(frame_gpu)

            center_idx = i - half_w
            if i < 2 * half_w or i >= n_frames:
                continue

            center_frame_gpu = frame_buffer[half_w]

            # Compute median and std over the ring buffer, excluding the center slot
            center_ring_idx = (ring_pos - 1 - half_w) % window_size
            # Gather all slots except center into a contiguous array
            other_indices = [j for j in range(window_size) if j != center_ring_idx]
            window_data = ring_buffer_gpu[other_indices]  # shape: (window_size-1, n_rtn)
            ref_rtn_gpu = cp.median(window_data, axis=0)
            std_rtn_gpu = cp.std(window_data, axis=0)

            # Run sparse correction
            frame_rtn_gpu = center_frame_gpu.ravel()[rtn_flat_idx_gpu]
            corrected_rtn_gpu = correct_sparse(frame_rtn_gpu, ref_rtn_gpu, std_rtn_gpu)

            # Scatter back into full frame
            corrected_gpu = scatter_to_frame(center_frame_gpu, corrected_rtn_gpu)
            corrected = cp.asnumpy(corrected_gpu)

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
            frame_gpu = cp.asarray(frame)

            # Spatial median filter needs the full frame for neighborhood context
            reference_gpu = gpu_median_filter(frame_gpu, size=args.median_size)

            # Gather RTN pixels from frame and reference
            frame_rtn_gpu = frame_gpu.ravel()[rtn_flat_idx_gpu]
            ref_rtn_gpu = reference_gpu.ravel()[rtn_flat_idx_gpu]

            corrected_rtn_gpu = correct_sparse(frame_rtn_gpu, ref_rtn_gpu)

            corrected_gpu = scatter_to_frame(frame_gpu, corrected_rtn_gpu)
            corrected = cp.asnumpy(corrected_gpu)

            header['RTNCORR'] = True
            header['RTNREF'] = 'median_filter'
            header['RTNMEDSZ'] = args.median_size

            out_path = out_folder / output_name
            fits.writeto(out_path, corrected.astype(np.int16), header, overwrite=True)

            if args.verbose:
                print(f"Corrected {i + 1}/{n_frames}: {output_name}")

    print(f"Done. Processed {n_frames} frames.")

    # Scatter sparse correction counts back to full 2D for statistics
    num_corr_arr = np.zeros((2, *rtn_mask.shape), dtype=np.int32)
    num_corr_arr[0].ravel()[rtn_flat_idx] = cp.asnumpy(num_corr_low_rtn_gpu)
    num_corr_arr[1].ravel()[rtn_flat_idx] = cp.asnumpy(num_corr_high_rtn_gpu)

    frac_high_corrections = num_corr_arr[1] / n_frames / (1 - rtn_params[1] - rtn_params[2])
    frac_low_corrections = num_corr_arr[0] / n_frames / rtn_params[2]
    print(np.median(frac_high_corrections[rtn_mask]), np.median(frac_low_corrections[rtn_mask]))
    updated_bias_frame = old_mean_bias + np.nan_to_num(rtn_params[3]) / e_per_adu * (num_corr_arr[0] - num_corr_arr[1]) / n_frames
    correction_values = np.nan_to_num(rtn_params[3]) / e_per_adu * (num_corr_arr[0] - num_corr_arr[1]) / n_frames
    print(np.mean(correction_values[rtn_mask]))
    correction_values_folder = out_folder / 'correction_values_frame'
    correction_values_folder.mkdir(exist_ok=True)
    fits.writeto(correction_values_folder / 'correction_values_frame.fits', correction_values.astype(np.float32), overwrite=True)


if __name__ == '__main__':
    main()
