# Create a LUT for central peak thresholds based on signal level (lam) and read noise (rn)
# Delta_x shifting is done at runtime since it's just addition/subtraction

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.stats import poisson, norm
from scipy.interpolate import interp1d
import pickle

class ThresholdLUT:
    """Lookup table for central peak thresholds."""
    
    def __init__(self, lam_range, read_noise_range, alpha=0.003):
        """
        Build lookup table over parameter grid.
        
        Parameters
        ----------
        lam_range : array-like
            Lambda values to sample (e.g., np.linspace(1, 1000, 50))
        read_noise_range : array-like
            Read noise values in same units as lambda
        alpha : float
            Tail probability for outlier detection
        """
        self.lam_vals = np.asarray(lam_range)
        self.rn_vals = np.asarray(read_noise_range)
        self.alpha = alpha
        
        # Output arrays: 2 threshold values at each grid point
        shape = (len(self.lam_vals), len(self.rn_vals))
        self.central_low = np.zeros(shape)
        self.central_high = np.zeros(shape)
        
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
        total = len(self.lam_vals) * len(self.rn_vals)
        count = 0
        
        for i, lam in enumerate(self.lam_vals):
            for j, rn in enumerate(self.rn_vals):
                central_low, central_high = self._get_quantile_thresholds(lam, rn)
                self.central_low[i, j] = central_low
                self.central_high[i, j] = central_high
                
                count += 1
                if count % 100 == 0:
                    print(f"Building LUT: {count}/{total} ({100*count/total:.1f}%)")
    
    def _build_interpolators(self):
        """Create interpolator objects"""
        grid = (self.lam_vals, self.rn_vals)
        self._interp_low = RegularGridInterpolator(grid, self.central_low)
        self._interp_high = RegularGridInterpolator(grid, self.central_high)
    
    def get_central_thresholds(self, lam, read_noise):
        """
        Get central peak thresholds via interpolation.
        
        Returns
        -------
        (central_low, central_high)
        """
        pts = np.atleast_2d(np.column_stack([
            np.atleast_1d(lam), 
            np.atleast_1d(read_noise)
        ]))
        
        low = self._interp_low(pts)
        high = self._interp_high(pts)
        
        if np.isscalar(lam) and np.isscalar(read_noise):
            return low.item(), high.item()
        return low, high
    
    def get_peak_thresholds(self, lam, read_noise, delta_x):
        """
        Get thresholds for low and high peaks.
        
        Returns
        -------
        (low_peak_low, low_peak_high), (high_peak_low, high_peak_high)
        """
        central_low, central_high = self.get_central_thresholds(lam, read_noise)
        
        high_peak_high = central_high + delta_x
        high_peak_low = np.maximum(central_low + delta_x, central_high)
        low_peak_low = central_low - delta_x
        low_peak_high = np.minimum(central_high - delta_x, central_low)
        
        return (low_peak_low, low_peak_high), (high_peak_low, high_peak_high)
    
    def get_lambda_max(self, read_noise, delta_x):
        """Get maximum lambda for reliable peak separation."""
        # delta_x_min = central_high - lam, so lambda_max is where delta_x = central_high - lam
        # i.e., lam = central_high - delta_x
        # We need to find lam where central_high(lam, rn) - lam = delta_x
        
        pts = np.column_stack([self.lam_vals, np.full_like(self.lam_vals, read_noise)])
        central_high_curve = self._interp_high(pts)
        delta_x_min_curve = central_high_curve - self.lam_vals
        
        inv_interp = interp1d(delta_x_min_curve, self.lam_vals,
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
        lam_range=np.linspace(0, 100, 200),
        read_noise_range=np.linspace(0.5, 10.0, 200),
        alpha=0.003
    )
    lut.save('rts_threshold_lut.pkl')

    print("Central thresholds:", lut.get_central_thresholds(lam=5, read_noise=1.5))
    print("Peak thresholds:", lut.get_peak_thresholds(lam=5, read_noise=1.5, delta_x=10))
    print("Lambda max:", lut.get_lambda_max(read_noise=1.5, delta_x=10))