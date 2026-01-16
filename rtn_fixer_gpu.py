#!/usr/bin/env python
"""
GPU-accelerated RTN correction using CuPy.

Usage:
    python rtn_fixer_gpu.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -w 10
    python rtn_fixer_gpu.py rtn_params.fits ./frames/ 0.5 -o ./corrected/ -m 3
"""

import argparse
from pathlib import Path
from collections import deque
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import median_filter as gpu_median_filter
from astropy.io import fits
from scripts.make_lut import ThresholdLUT

# GPU kernel for bilinear interpolation and threshold calculation
_interp_and_correct_kernel = cp.ElementwiseKernel(
    in_params='''
        float64 frame_val, float64 ref_val, bool rtn_mask,
        float64 delta_x, float64 mu, float64 read_noise, float64 e_per_adu,
        raw float64 lam_vals, raw float64 rn_vals,
        raw float64 central_low_data, raw float64 central_high_data,
        int32 n_lam, int32 n_rn
    ''',
    out_params='float64 corrected, int32 corr_type',
    operation='''
        corrected = frame_val;
        corr_type = 0;  // 0 = no correction, 1 = low peak, 2 = high peak
        
        if (!rtn_mask) return;
        
        // Compute lambda
        double lam = ref_val * e_per_adu - mu;
        if (lam < 0) lam = 0;
        
        // Find indices for bilinear interpolation
        int il = 0, jr = 0;
        for (int ki = 0; ki < n_lam - 1; ki++) {
            if (lam_vals[ki + 1] > lam) break;
            il = ki + 1;
        }
        for (int kj = 0; kj < n_rn - 1; kj++) {
            if (rn_vals[kj + 1] > read_noise) break;
            jr = kj + 1;
        }
        
        // Clamp indices
        if (il >= n_lam - 1) il = n_lam - 2;
        if (jr >= n_rn - 1) jr = n_rn - 2;
        if (il < 0) il = 0;
        if (jr < 0) jr = 0;
        
        // Compute interpolation weights
        double t = (lam - lam_vals[il]) / (lam_vals[il + 1] - lam_vals[il]);
        double u = (read_noise - rn_vals[jr]) / (rn_vals[jr + 1] - rn_vals[jr]);
        if (t < 0) t = 0; if (t > 1) t = 1;
        if (u < 0) u = 0; if (u > 1) u = 1;
        
        // Bilinear interpolation for central thresholds
        int idx00 = il * n_rn + jr;
        int idx01 = il * n_rn + (jr + 1);
        int idx10 = (il + 1) * n_rn + jr;
        int idx11 = (il + 1) * n_rn + (jr + 1);
        
        double central_low = central_low_data[idx00] * (1-t) * (1-u) +
                             central_low_data[idx01] * (1-t) * u +
                             central_low_data[idx10] * t * (1-u) +
                             central_low_data[idx11] * t * u;
        
        double central_high = central_high_data[idx00] * (1-t) * (1-u) +
                              central_high_data[idx01] * (1-t) * u +
                              central_high_data[idx10] * t * (1-u) +
                              central_high_data[idx11] * t * u;
        
        // Compute peak thresholds
        double high_peak_high = central_high + delta_x;
        double high_peak_low = central_low + delta_x;
        if (high_peak_low < central_high) high_peak_low = central_high;
        
        double low_peak_low = central_low - delta_x;
        double low_peak_high = central_high - delta_x;
        if (low_peak_high > central_low) low_peak_high = central_low;
        
        // Check for correction
        double diff = (frame_val - ref_val) * e_per_adu;
        
        if (diff > high_peak_low && diff < high_peak_high) {
            corrected = frame_val - round(delta_x / e_per_adu);
            corr_type = 2;
        } else if (diff < low_peak_high && diff > low_peak_low) {
            corrected = frame_val + round(delta_x / e_per_adu);
            corr_type = 1;
        }
    ''',
    name='interp_and_correct'
)


class RTNFixerGPU:
    def __init__(self, rtn_params_path, gain, lut_path='rts_threshold_lut.pkl'):
        """Initialize GPU-based RTN fixer."""
        # Load RTN parameters
        with fits.open(rtn_params_path) as hdul:
            rtn_params = hdul[0].data
        
        self.e_per_adu = 1.0 / gain
        
        # Precompute and transfer to GPU
        rtn_mask = ~np.isnan(rtn_params[0])
        delta_x_arr_e = rtn_params[3] * self.e_per_adu
        mu_e = self.e_per_adu * (
            (rtn_params[0] * rtn_params[1]) +
            (rtn_params[0] - rtn_params[3]) * rtn_params[2] +
            (rtn_params[0] + rtn_params[3]) * (1 - rtn_params[1] - rtn_params[2])
        )
        read_noise_e = rtn_params[4] * self.e_per_adu
        
        # Transfer to GPU
        self.rtn_mask_gpu = cp.asarray(rtn_mask)
        self.delta_x_gpu = cp.asarray(delta_x_arr_e, dtype=cp.float64)
        self.mu_gpu = cp.asarray(mu_e, dtype=cp.float64)
        self.read_noise_gpu = cp.asarray(read_noise_e, dtype=cp.float64)
        
        # Load LUT and transfer to GPU
        lut = ThresholdLUT.load(lut_path)
        self.lam_vals_gpu = cp.asarray(lut.lam_vals, dtype=cp.float64)
        self.rn_vals_gpu = cp.asarray(lut.rn_vals, dtype=cp.float64)
        self.central_low_gpu = cp.asarray(lut.central_low.ravel(), dtype=cp.float64)
        self.central_high_gpu = cp.asarray(lut.central_high.ravel(), dtype=cp.float64)
        self.n_lam = len(lut.lam_vals)
        self.n_rn = len(lut.rn_vals)
        
        # Statistics
        self.shape = rtn_mask.shape
        self.num_rtn_pixels = int(np.sum(rtn_mask))
        self.num_corr_low = cp.zeros(self.shape, dtype=cp.int32)
        self.num_corr_high = cp.zeros(self.shape, dtype=cp.int32)
        
        print(f"{self.num_rtn_pixels} RTN pixels detected.")
    
    def correct_frame(self, frame_gpu, reference_gpu):
        """Apply RTN correction to a single frame on GPU."""
        corrected, corr_type = _interp_and_correct_kernel(
            frame_gpu, reference_gpu, self.rtn_mask_gpu,
            self.delta_x_gpu, self.mu_gpu, self.read_noise_gpu, self.e_per_adu,
            self.lam_vals_gpu, self.rn_vals_gpu,
            self.central_low_gpu, self.central_high_gpu,
            self.n_lam, self.n_rn
        )
        
        # Update statistics
        self.num_corr_low += (corr_type == 1).astype(cp.int32)
        self.num_corr_high += (corr_type == 2).astype(cp.int32)
        
        return corrected
    
    def get_stats(self):
        """Return correction statistics as numpy arrays."""
        return cp.asnumpy(self.num_corr_low), cp.asnumpy(self.num_corr_high)


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
    parser = argparse.ArgumentParser(description='GPU-accelerated RTN correction')
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

    # Validate arguments
    if args.window is not None and args.median_size is not None:
        parser.error("Specify either -w (rolling mean) or -m (median filter), not both")
    if args.window is None and args.median_size is None:
        parser.error("Must specify either -w (rolling mean) or -m (median filter)")

    use_rolling_mean = args.window is not None

    # Initialize GPU fixer
    fixer = RTNFixerGPU(args.rtn_params, args.gain)

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

    out_folder = Path(args.output)
    out_folder.mkdir(parents=True, exist_ok=True)

    if use_rolling_mean:
        window_buffer = deque(maxlen=2 * half_w + 1)
        rolling_sum_gpu = None

        for i in range(n_frames):
            frame, header = load_frame(files[i])
            frame_gpu = cp.asarray(frame)
            
            if rolling_sum_gpu is None:
                rolling_sum_gpu = cp.zeros_like(frame_gpu)
            
            if len(window_buffer) == window_buffer.maxlen:
                rolling_sum_gpu -= window_buffer[0]
            window_buffer.append(frame_gpu)
            rolling_sum_gpu += frame_gpu

            center_idx = i - half_w
            if center_idx < 0 or center_idx >= n_frames - half_w:
                continue

            # reference_gpu = rolling_sum_gpu / len(window_buffer)
            reference_gpu = cp.median(cp.stack(window_buffer), axis=0)
            center_frame_gpu = window_buffer[half_w]
            
            corrected_gpu = fixer.correct_frame(center_frame_gpu, reference_gpu)
            corrected = cp.asnumpy(corrected_gpu)

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
            frame_gpu = cp.asarray(frame)
            
            reference_gpu = gpu_median_filter(frame_gpu, size=args.median_size)
            
            corrected_gpu = fixer.correct_frame(frame_gpu, reference_gpu)
            corrected = cp.asnumpy(corrected_gpu)

            header['RTNCORR'] = True
            header['RTNREF'] = 'median_filter'
            header['RTNMEDSZ'] = args.median_size
            
            out_path = out_folder / fpath.name
            fits.writeto(out_path, corrected.astype(np.int16), header, overwrite=True)
            
            if args.verbose:
                print(f"Corrected {i + 1}/{n_frames}: {fpath.name}")

    print(f"Done. Processed {n_frames} frames.")
    
    # Statistics
    num_corr_low, num_corr_high = fixer.get_stats()
    
    with fits.open(args.rtn_params) as hdul:
        rtn_params = hdul[0].data
    
    frac_high = num_corr_high / n_frames / (1 - rtn_params[1] - rtn_params[2])
    frac_low = num_corr_low / n_frames / rtn_params[2]
    
    import matplotlib.pyplot as plt
    plt.hist(frac_high.flatten(), bins=50, alpha=0.5, label='High corrections', range=(0, 2))
    plt.hist(frac_low.flatten(), bins=50, alpha=0.5, label='Low corrections', range=(0, 2))
    plt.xlabel('Num corrections / expected corrections')
    plt.ylabel('Number of pixels')
    plt.legend()
    plt.show()
    print(f"Median fractions: high={np.nanmedian(frac_high):.3f}, low={np.nanmedian(frac_low):.3f}")


if __name__ == '__main__':
    main()