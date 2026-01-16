# Create a LUT that allows determination of thresholds for RTN correction
# based on input parameters: signal level (lam), read noise (rn), and delta_x

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import poisson, norm
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import pickle

class ThresholdLUT:
    """Lookup table for RTS peak identification thresholds."""
    
    def __init__(self, lam_range, read_noise_range, delta_x_range, alpha=0.003):
        """
        Build lookup table over parameter grid.
        
        Parameters
        ----------
        lam_range : array-like
            Lambda values to sample (e.g., np.linspace(1, 1000, 50))
        read_noise_range : array-like
            Read noise values in same units as lambda
        delta_x_range : array-like
            RTS amplitude values to sample
        alpha : float
            Tail probability for outlier detection
        """
        self.lam_vals = np.asarray(lam_range)
        self.rn_vals = np.asarray(read_noise_range)
        self.dx_vals = np.asarray(delta_x_range)
        self.alpha = alpha
        
        # Output arrays: 4 threshold values at each grid point
        shape = (len(self.lam_vals), len(self.rn_vals), len(self.dx_vals))
        self.low_peak_low = np.zeros(shape)
        self.low_peak_high = np.zeros(shape)
        self.high_peak_low = np.zeros(shape)
        self.high_peak_high = np.zeros(shape)
        # 2D array for delta_x_min (only depends on read_noise and lambda)
        self.delta_x_min = np.zeros((len(self.lam_vals), len(self.rn_vals)))
        
        self._build_table()
        self._build_interpolators()
    
    def _poisson_gaussian_cdf(self, x_vals, lam, read_noise):
        """Compute CDF of Poisson(lam) convolved with N(0, read_noise)"""
        n_terms = int(lam + 10 * np.sqrt(lam)) + 20
        cdf = np.zeros_like(x_vals, dtype=float)
        for n in range(n_terms):
            p_n = poisson.pmf(n, lam)
            if p_n < 1e-15:
                continue
            cdf += p_n * norm.cdf(x_vals, loc=n, scale=read_noise)
        return cdf
    
    def _get_quantile_thresholds(self, lam, read_noise):
        """Find thresholds for given tail probability"""
        sigma_tot = np.sqrt(lam + read_noise**2)
        x = np.linspace(lam - 10*sigma_tot, lam + 10*sigma_tot, 1000)
        cdf = self._poisson_gaussian_cdf(x, lam, read_noise)
        
        # Handle potential issues with monotonicity
        valid = np.diff(cdf) > 0
        valid = np.concatenate([[True], valid])
        
        inv_cdf = interp1d(cdf[valid], x[valid], bounds_error=False, 
                          fill_value=(x[0], x[-1]))
        lower = inv_cdf(self.alpha / 2)
        upper = inv_cdf(1 - self.alpha / 2)
        return float(lower), float(upper)
    
    def _build_table(self):
        """Populate the lookup table"""
        total = len(self.lam_vals) * len(self.rn_vals) * len(self.dx_vals)
        count = 0
        
        for i, lam in enumerate(self.lam_vals):
            for j, rn in enumerate(self.rn_vals):
                # Compute quantiles once per (lam, rn) pair
                central_low, central_high = self._get_quantile_thresholds(lam, rn)
                self.delta_x_min[i, j] = central_high - lam
                
                for k, dx in enumerate(self.dx_vals):
                    # Use pre-computed quantiles
                    high_peak_high = central_high + dx
                    high_peak_low = max(central_low + dx, central_high)
                    low_peak_low = central_low - dx
                    low_peak_high = min(central_high - dx, central_low)
                    
                    self.low_peak_low[i, j, k] = low_peak_low
                    self.low_peak_high[i, j, k] = low_peak_high
                    self.high_peak_low[i, j, k] = high_peak_low
                    self.high_peak_high[i, j, k] = high_peak_high
                    
                    count += 1
                    if count % 500 == 0:
                        print(f"Building LUT: {count}/{total} ({100*count/total:.1f}%)")
    
    def _build_interpolators(self):
        """Create interpolator objects"""
        grid = (self.lam_vals, self.rn_vals, self.dx_vals)
        self._interp_ll = RegularGridInterpolator(grid, self.low_peak_low)
        self._interp_lh = RegularGridInterpolator(grid, self.low_peak_high)
        self._interp_hl = RegularGridInterpolator(grid, self.high_peak_low)
        self._interp_hh = RegularGridInterpolator(grid, self.high_peak_high)
        
        # 2D interpolator for delta_x_min
        grid_2d = (self.lam_vals, self.rn_vals)
        self._interp_delta_x_min = RegularGridInterpolator(grid_2d, self.delta_x_min)
    
    def __call__(self, lam, read_noise, delta_x):
        """
        Get thresholds via interpolation.
        
        Can handle scalar or array inputs.
        
        Returns
        -------
        (low_peak_low, low_peak_high), (high_peak_low, high_peak_high)
        """
        # Check that lam, read_noise, delta_x have compatible shapes.
        # All should either be scalar or 1D arrays of the same length.
        if np.isscalar(lam) and np.isscalar(read_noise) and np.isscalar(delta_x):
            pass
        elif (np.ndim(lam) == 1 and np.ndim(read_noise) == 1 and np.ndim(delta_x) == 1 and
              len(lam) == len(read_noise) == len(delta_x)):
            pass
        else:
            raise ValueError("lam, read_noise, delta_x must be all scalars or 1D arrays of the same length.")

        pts = np.column_stack((lam, read_noise, delta_x))
        
        ll = self._interp_ll(pts)
        lh = self._interp_lh(pts)
        hl = self._interp_hl(pts)
        hh = self._interp_hh(pts)
        
        if np.isscalar(lam) and np.isscalar(read_noise) and np.isscalar(delta_x):
            return np.array([ll.item(), lh.item()]), np.array([hl.item(), hh.item()])
        return np.array([ll, lh, hl, hh]).T
    
    def get_lambda_max(self, read_noise, delta_x):
        """Get maximum lambda for reliable peak separation via interpolation."""
        # Get delta_x_min vs lambda at this read_noise
        pts = np.column_stack([self.lam_vals, np.full_like(self.lam_vals, read_noise)])
        dx_min_curve = self._interp_delta_x_min(pts)
        
        # Invert: lambda as function of delta_x_min (monotonically increasing)
        inv_interp = interp1d(dx_min_curve, self.lam_vals,
                            bounds_error=False, fill_value=(0.0, np.inf))
        result = inv_interp(delta_x)
        if np.isscalar(delta_x):
            return float(result)
        return result
    
    def save(self, filename):
        """Save LUT to disk"""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @staticmethod
    def load(filename):
        """Load LUT from disk"""
        with open(filename, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    lut = ThresholdLUT(
                       lam_range=np.linspace(0, 100, 100),
                       read_noise_range=np.linspace(0.5, 10.0, 200),
                       delta_x_range=np.linspace(1, 50, 100),
                       alpha=0.003
                       )
    lut.save('rts_threshold_lut.pkl')

    print(lut(lam=5, read_noise=1.5, delta_x=10))
    print(lut.get_lambda_max(read_noise=1.5, delta_x=10))