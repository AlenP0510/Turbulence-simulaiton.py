[turbulence_analysis.py](https://github.com/user-attachments/files/24918838/turbulence_analysis.py)
"""
3D Turbulence Analysis: Enhanced Implementation
Author: [Alen]
Date: [1/27]
"""

import numpy as np
from scipy import fft, ndimage
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy import stats
from skimage import measure
import warnings

warnings.filterwarnings('ignore')

# ============== Original Code Section ==============
@dataclass
class TurbulenceParameters:
    """Container for physical parameters"""
    Re_lambda: float = 200.0      # Taylor Reynolds number
    urms: float = 1.0            # Root mean square velocity
    L_int: float = None          # Integral length scale
    epsilon: float = None        # Dissipation rate
    
    def __post_init__(self):
        """Calculate derived parameters based on Reynolds number"""
        if self.L_int is None:
            self.L_int = 1.0  # Normalized integral scale
        if self.epsilon is None:
            self.epsilon = self.urms**3 / self.L_int

class PhysicalTurbulenceGenerator:
    """Physics-based turbulence field generator"""
    
    def __init__(self, params: TurbulenceParameters, grid_size: int = 64):
        self.params = params
        self.N = grid_size
        self.dx = 2 * np.pi / grid_size
        
    def generate_velocity_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate velocity field satisfying physical constraints"""
        # Simplified version: generate random but physically reasonable velocity field
        np.random.seed(42)  # For reproducibility
        
        # Generate random velocity field
        u = np.random.randn(self.N, self.N, self.N)
        v = np.random.randn(self.N, self.N, self.N)
        w = np.random.randn(self.N, self.N, self.N)
        
        # Apply low-pass filter to simulate energy cascade
        u = self._apply_spectral_filter(u)
        v = self._apply_spectral_filter(v)
        w = self._apply_spectral_filter(w)
        
        # Enforce incompressibility condition
        u, v, w = self._project_to_incompressible(u, v, w)
        
        # Scale to achieve target statistical properties
        u, v, w = self._scale_to_parameters(u, v, w)
        
        return u, v, w
    
    def _apply_spectral_filter(self, field: np.ndarray) -> np.ndarray:
        """Apply energy spectrum filter"""
        # Fourier transform
        field_hat = fft.fftn(field)
        
        # Create wavenumber grid
        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        kz = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k_mag = np.sqrt(Kx**2 + Ky**2 + Kz**2)
        
        # Kolmogorov energy spectrum: E(k) ∝ k^{-5/3}
        # Apply low-pass filter
        k_cutoff = self.N / 4  # Cutoff wavenumber
        filter_func = np.exp(-(k_mag / k_cutoff)**4)
        field_hat *= filter_func
        
        # Inverse transform
        return np.real(fft.ifftn(field_hat))
    
    def _project_to_incompressible(self, u: np.ndarray, v: np.ndarray, 
                                   w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Project to incompressible space"""
        # Calculate divergence
        du_dx = np.gradient(u, self.dx, axis=0)
        dv_dy = np.gradient(v, self.dx, axis=1)
        dw_dz = np.gradient(w, self.dx, axis=2)
        divergence = du_dx + dv_dy + dw_dz
        
        # Solve Poisson equation for pressure field
        divergence_hat = fft.fftn(divergence)
        kx = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        ky = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        kz = np.fft.fftfreq(self.N, d=self.dx) * 2 * np.pi
        Kx, Ky, Kz = np.meshgrid(kx, ky, kz, indexing='ij')
        k2 = Kx**2 + Ky**2 + Kz**2
        k2[0, 0, 0] = 1  # Avoid division by zero
        
        # Pressure field
        p_hat = divergence_hat / k2
        p_hat[0, 0, 0] = 0  # Set mean pressure to zero
        
        # Pressure gradient
        p = np.real(fft.ifftn(p_hat))
        dp_dx = np.gradient(p, self.dx, axis=0)
        dp_dy = np.gradient(p, self.dx, axis=1)
        dp_dz = np.gradient(p, self.dx, axis=2)
        
        # Subtract pressure gradient from velocity field
        u -= dp_dx
        v -= dp_dy
        w -= dp_dz
        
        return u, v, w
    
    def _scale_to_parameters(self, u: np.ndarray, v: np.ndarray, 
                            w: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Scale to target parameters"""
        # Calculate current statistics
        u_rms_current = np.std(u)
        v_rms_current = np.std(v)
        w_rms_current = np.std(w)
        total_rms_current = np.sqrt(u_rms_current**2 + v_rms_current**2 + w_rms_current**2)
        
        # Target RMS
        target_rms = self.params.urms
        
        # Scaling factor
        scale_factor = target_rms / (total_rms_current / np.sqrt(3))
        
        u *= scale_factor
        v *= scale_factor
        w *= scale_factor
        
        return u, v, w

class DirectionalDifferenceAnalyzer:
    """Directional difference parameter analysis"""
    
    def __init__(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                 rho: np.ndarray, dx: float):
        self.u = u
        self.v = v
        self.w = w
        self.rho = rho
        self.dx = dx
        
    def compute_z_parameter(self) -> Tuple[np.ndarray, Dict]:
        """Calculate 3D z parameter"""
        # 1. Calculate gradient
        grad_rho = np.gradient(self.rho, self.dx, edge_order=2)
        
        # 2. Gradient regularization
        grad_mag = np.sqrt(grad_rho[0]**2 + grad_rho[1]**2 + grad_rho[2]**2 + 1e-10)
        grad_mag[grad_mag < 1e-3 * np.max(grad_mag)] = np.nan
        
        # 3. Calculate unit normal vector
        e = [g / grad_mag for g in grad_rho]
        
        # 4. Calculate velocity magnitude
        u_mag = np.sqrt(self.u**2 + self.v**2 + self.w**2 + 1e-10)
        
        # 5. Calculate normal and tangential components
        u_dot_e = self.u * e[0] + self.v * e[1] + self.w * e[2]
        u_perp_mag = np.sqrt(np.maximum(u_mag**2 - u_dot_e**2, 0) + 1e-10)
        
        # 6. Calculate z parameter: tangential velocity magnitude / total velocity magnitude
        z = u_perp_mag / u_mag
        
        # 7. Calculate statistics
        z_clean = z[~np.isnan(z)]
        z_stats = {
            'mean': np.mean(z_clean),
            'std': np.std(z_clean),
            'min': np.min(z_clean),
            'max': np.max(z_clean),
            'skewness': stats.skew(z_clean),
            'kurtosis': stats.kurtosis(z_clean),
            'percentiles': {
                '10th': np.percentile(z_clean, 10),
                '25th': np.percentile(z_clean, 25),
                '50th': np.percentile(z_clean, 50),
                '75th': np.percentile(z_clean, 75),
                '90th': np.percentile(z_clean, 90)
            }
        }
        
        return z, z_stats
    
    def physical_interpretation(self, mean_z: float) -> Dict:
        """Physical interpretation based on z value"""
        interpretations = {
            'flow_regime': None,
            'mixing_efficiency': None,
            'turbulence_intensity': None,
            'comparison_to_literature': None
        }
        
        if mean_z < 0.3:
            interpretations['flow_regime'] = 'Laminar-dominated'
            interpretations['mixing_efficiency'] = 'Low (mostly along isosurfaces)'
            interpretations['turbulence_intensity'] = 'Weak'
            interpretations['comparison_to_literature'] = (
                'Similar to stable stratified flows or strong shear layers '
                '(see Smyth & Moum, 2000)'
            )
        elif mean_z > 0.7:
            interpretations['flow_regime'] = 'Turbulent-dominated'
            interpretations['mixing_efficiency'] = 'High (frequent crossing)'
            interpretations['turbulence_intensity'] = 'Strong'
            interpretations['comparison_to_literature'] = (
                'Consistent with fully developed isotropic turbulence '
                '(see Pope, 2000)'
            )
        else:
            interpretations['flow_regime'] = 'Transitional/Mixed'
            interpretations['mixing_efficiency'] = 'Moderate'
            interpretations['turbulence_intensity'] = 'Medium'
            interpretations['comparison_to_literature'] = (
                'Typical of moderate Reynolds number flows '
                '(see Warhaft, 2000)'
            )
        
        return interpretations

# ============== Enhanced Code Section ==============
class EnhancedTurbulenceGenerator(PhysicalTurbulenceGenerator):
    """Enhanced turbulence generator"""
    
    def __init__(self, params: TurbulenceParameters, grid_size: int = 64):
        super().__init__(params, grid_size)
    
    def generate_with_different_methods(self, method: str = 'random') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Different turbulence generation methods"""
        if method == 'random':
            return self.generate_velocity_field()
        elif method == 'gaussian':
            return self._generate_gaussian_field()
        elif method == 'fractal':
            return self._generate_fractal_field()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _generate_gaussian_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate Gaussian-correlated turbulence field"""
        # Use Gaussian filter to create correlated random field
        noise = np.random.randn(self.N, self.N, self.N, 3)
        
        # Apply Gaussian filter to each component
        sigma = 2.0  # Correlation length
        u = ndimage.gaussian_filter(noise[..., 0], sigma=sigma)
        v = ndimage.gaussian_filter(noise[..., 1], sigma=sigma)
        w = ndimage.gaussian_filter(noise[..., 2], sigma=sigma)
        
        # Enforce incompressibility condition
        u, v, w = self._project_to_incompressible(u, v, w)
        
        # Scale
        u, v, w = self._scale_to_parameters(u, v, w)
        
        return u, v, w
    
    def _generate_fractal_field(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate fractal turbulence field"""
        u = self._create_fractal_noise()
        v = self._create_fractal_noise()
        w = self._create_fractal_noise()
        
        # Enforce incompressibility condition
        u, v, w = self._project_to_incompressible(u, v, w)
        
        # Scale
        u, v, w = self._scale_to_parameters(u, v, w)
        
        return u, v, w
    
    def _create_fractal_noise(self) -> np.ndarray:
        """Create fractal noise"""
        field = np.zeros((self.N, self.N, self.N))
        
        # Multi-scale superposition
        for octave in range(4):
            scale = 2 ** octave
            amplitude = 1.0 / scale
            
            # Generate noise at this scale
            noise = np.random.randn(self.N // scale, self.N // scale, self.N // scale)
            
            # Upsample to original size
            if scale > 1:
                noise = ndimage.zoom(noise, scale, order=1)
                # Truncate to correct size
                noise = noise[:self.N, :self.N, :self.N]
            
            field += amplitude * noise
        
        return field

class EnhancedDirectionalAnalyzer(DirectionalDifferenceAnalyzer):
    """Enhanced directional difference analysis"""
    
    def __init__(self, u: np.ndarray, v: np.ndarray, w: np.ndarray, 
                 rho: np.ndarray, dx: float):
        super().__init__(u, v, w, rho, dx)
    
    def compute_z_by_scale(self, scales: List[float]) -> Dict:
        """Calculate z parameter at different scales"""
        results = {}
        
        for scale in scales:
            # Scale the density field
            smoothed_rho = ndimage.gaussian_filter(self.rho, sigma=scale)
            
            # Recalculate gradient
            grad_rho = np.gradient(smoothed_rho, self.dx, edge_order=2)
            grad_mag = np.sqrt(grad_rho[0]**2 + grad_rho[1]**2 + grad_rho[2]**2 + 1e-10)
            grad_mag[grad_mag < 1e-3 * np.max(grad_mag)] = np.nan
            
            # Calculate unit normal vector
            e = [g / grad_mag for g in grad_rho]
            
            # Calculate velocity magnitude
            u_mag = np.sqrt(self.u**2 + self.v**2 + self.w**2 + 1e-10)
            
            # Calculate z parameter
            u_dot_e = self.u * e[0] + self.v * e[1] + self.w * e[2]
            u_perp_mag = np.sqrt(np.maximum(u_mag**2 - u_dot_e**2, 0) + 1e-10)
            z = u_perp_mag / u_mag
            
            # Statistics
            z_clean = z[~np.isnan(z)]
            results[f'scale_{scale:.1f}'] = {
                'mean': np.mean(z_clean),
                'std': np.std(z_clean),
                'skewness': stats.skew(z_clean),
                'kurtosis': stats.kurtosis(z_clean)
            }
        
        return results
    
    def compute_spatial_correlation(self, max_lag: int = 10) -> np.ndarray:
        """Calculate spatial correlation function of z parameter"""
        z, _ = self.compute_z_parameter()
        z_clean = z[~np.isnan(z)]
        
        # Normalize
        z_normalized = (z_clean - np.mean(z_clean)) / np.std(z_clean)
        
        # Reshape to 3D array (assuming cube)
        n = int(np.cbrt(len(z_normalized)))
        if n**3 != len(z_normalized):
            n = int(len(z_normalized)**(1/3))
        
        z_3d = z_normalized[:n**3].reshape((n, n, n))
        
        # Calculate autocorrelation
        correlation = np.zeros((max_lag, max_lag, max_lag))
        
        for dx in range(max_lag):
            for dy in range(max_lag):
                for dz in range(max_lag):
                    shifted = np.roll(z_3d, shift=(dx, dy, dz), axis=(0, 1, 2))
                    correlation[dx, dy, dz] = np.mean(z_3d * shifted)
        
        return correlation

class TurbulenceVisualizer:
    """Turbulence visualization tool"""
    
    @staticmethod
    def plot_z_distribution(z: np.ndarray, save_path: Optional[str] = None):
        """Plot distribution of z parameter"""
        z_clean = z[~np.isnan(z)]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Histogram
        axes[0, 0].hist(z_clean, bins=50, density=True, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('z parameter')
        axes[0, 0].set_ylabel('Probability Density')
        axes[0, 0].set_title('Distribution of z parameter')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Box plot
        axes[0, 1].boxplot(z_clean, vert=True)
        axes[0, 1].set_ylabel('z value')
        axes[0, 1].set_title('Box plot of z parameter')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Cumulative distribution function
        sorted_z = np.sort(z_clean)
        cdf = np.arange(1, len(sorted_z) + 1) / len(sorted_z)
        axes[1, 0].plot(sorted_z, cdf, 'b-', linewidth=2)
        axes[1, 0].set_xlabel('z parameter')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].set_title('Cumulative Distribution Function')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. QQ plot (compared to uniform distribution)
        stats.probplot(z_clean, dist="uniform", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (vs Uniform Distribution)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_z_slice(z: np.ndarray, slice_index: int = 32, save_path: Optional[str] = None):
        """Plot slices of z parameter"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        titles = ['XY plane', 'XZ plane', 'YZ plane']
        
        # XY plane
        im1 = axes[0].imshow(z[slice_index, :, :], cmap='viridis', 
                            vmin=0, vmax=1, aspect='auto')
        axes[0].set_title(f'{titles[0]} (z={slice_index})')
        axes[0].set_xlabel('Y')
        axes[0].set_ylabel('X')
        plt.colorbar(im1, ax=axes[0])
        
        # XZ plane
        im2 = axes[1].imshow(z[:, slice_index, :], cmap='viridis', 
                            vmin=0, vmax=1, aspect='auto')
        axes[1].set_title(f'{titles[1]} (y={slice_index})')
        axes[1].set_xlabel('Z')
        axes[1].set_ylabel('X')
        plt.colorbar(im2, ax=axes[1])
        
        # YZ plane
        im3 = axes[2].imshow(z[:, :, slice_index], cmap='viridis', 
                            vmin=0, vmax=1, aspect='auto')
        axes[2].set_title(f'{titles[2]} (x={slice_index})')
        axes[2].set_xlabel('Z')
        axes[2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# ============== Main Program ==============
def main():
    """Main function: demonstrate complete analysis pipeline"""
    print("=== 3D Turbulence Analysis: Directional Difference Parameter ===")
    
    # 1. Set parameters
    params = TurbulenceParameters(Re_lambda=200.0, urms=1.0)
    print(f"Turbulence Parameters: Re_λ={params.Re_lambda}, u_rms={params.urms}")
    
    # 2. Generate turbulence field
    print("\n1. Generating turbulence field...")
    generator = EnhancedTurbulenceGenerator(params, grid_size=64)
    u, v, w = generator.generate_velocity_field()
    print(f"   Velocity field shape: {u.shape}")
    print(f"   u: mean={np.mean(u):.3f}, std={np.std(u):.3f}")
    print(f"   v: mean={np.mean(v):.3f}, std={np.std(v):.3f}")
    print(f"   w: mean={np.mean(w):.3f}, std={np.std(w):.3f}")
    
    # 3. Generate scalar field (density)
    print("\n2. Generating scalar field...")
    rho = np.random.randn(u.shape[0], u.shape[1], u.shape[2])
    rho = ndimage.gaussian_filter(rho, sigma=2.0)  # Add spatial correlation
    print(f"   Density field: mean={np.mean(rho):.3f}, std={np.std(rho):.3f}")
    
    # 4. Calculate z parameter
    print("\n3. Computing directional difference parameter (z)...")
    analyzer = EnhancedDirectionalAnalyzer(u, v, w, rho, dx=generator.dx)
    z, z_stats = analyzer.compute_z_parameter()
    
    print(f"   z statistics:")
    print(f"     Mean: {z_stats['mean']:.3f}")
    print(f"     Std: {z_stats['std']:.3f}")
    print(f"     Skewness: {z_stats['skewness']:.3f}")
    print(f"     Kurtosis: {z_stats['kurtosis']:.3f}")
    
    # 5. Physical interpretation
    print("\n4. Physical interpretation...")
    interpretation = analyzer.physical_interpretation(z_stats['mean'])
    for key, value in interpretation.items():
        print(f"   {key}: {value}")
    
    # 6. Multi-scale analysis
    print("\n5. Multi-scale analysis...")
    scales = [0.5, 1.0, 2.0, 4.0]
    scale_results = analyzer.compute_z_by_scale(scales)
    
    print("   Scale-dependent z mean values:")
    for scale, stats_dict in scale_results.items():
        print(f"     {scale}: {stats_dict['mean']:.3f}")
    
    # 7. Visualization
    print("\n6. Creating visualizations...")
    visualizer = TurbulenceVisualizer()
    
    # Distribution plot
    visualizer.plot_z_distribution(z, save_path=None)
    
    # Slice plot
    visualizer.plot_z_slice(z, slice_index=32, save_path=None)
    
    print("\n=== Analysis Complete ===")
    
    return {
        'params': params,
        'z': z,
        'z_stats': z_stats,
        'interpretation': interpretation,
        'scale_results': scale_results
    }

if __name__ == "__main__":
    results = main()
