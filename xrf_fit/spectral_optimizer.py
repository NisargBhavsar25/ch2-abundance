from optimizer import Optimizer
import numpy as np
from astropy import units as u
from specutils import Spectrum1D, SpectralRegion
from specutils.fitting import fit_generic_continuum
from specutils.manipulation import extract_region
from astropy.io import fits
import warnings

class SpectralOptimizer(Optimizer):
    def __init__(self, data_handler, fits_path, arf_path=None, x_range=None):
        """
        Spectral optimizer with background handling and ARF correction.
        
        Args:
            data_handler (DataHandler): Instance of DataHandler class
            fits_path (str): Path to the FITS spectrum file
            arf_path (str, optional): Path to the ARF (Auxiliary Response File)
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
        """
        self.arf_path = arf_path
        self.spectrum = None
        self.arf = None
        self.background = None
        super().__init__(data_handler, fits_path, x_range)
        
    def load_spectrum(self):
        """Load and prepare spectrum data with error handling."""
        try:
            with fits.open(self.fits_path) as hdul:
                data = hdul[1].data
                # Extract energy and counts
                energies = data['CHANNEL'] * 0.0277  # Convert channels to keV
                counts = data['COUNTS']
                errors = np.sqrt(counts)  # Assume Poisson errors
                
                # Create Spectrum1D object
                self.spectrum = Spectrum1D(
                    spectral_axis=energies * u.keV,
                    flux=counts * u.count,
                    uncertainty=errors * u.count
                )
                
                return energies, counts
                
        except Exception as e:
            raise ValueError(f"Error loading spectrum: {str(e)}")
            
    def load_arf(self):
        """Load ARF file if provided."""
        if self.arf_path:
            try:
                with fits.open(self.arf_path) as hdul:
                    self.arf = hdul[1].data
                    # Store effective area vs energy
                    self.arf_energies = self.arf['ENERGY']
                    self.arf_effective_area = self.arf['SPECRESP']
            except Exception as e:
                warnings.warn(f"Error loading ARF file: {str(e)}")
                self.arf = None
    
    def estimate_background(self, polynomial_order=3):
        """
        Estimate spectral background using polynomial fitting.
        
        Args:
            polynomial_order (int): Order of polynomial for continuum fitting
        """
        try:
            # Fit continuum to full spectrum
            self.background = fit_generic_continuum(
                self.spectrum,
                model=f'polynomial{polynomial_order}'
            )
        except Exception as e:
            warnings.warn(f"Error estimating background: {str(e)}")
            self.background = lambda x: np.zeros_like(x)
    
    def apply_arf_correction(self, model_intensity):
        """
        Apply ARF correction to model intensities.
        
        Args:
            model_intensity (numpy.ndarray): Model intensities
            
        Returns:
            numpy.ndarray: ARF-corrected intensities
        """
        if self.arf is not None:
            # Interpolate ARF to match spectrum energies
            arf_correction = np.interp(
                self.energies,
                self.arf_energies,
                self.arf_effective_area
            )
            return model_intensity * arf_correction
        return model_intensity
    
    def calculate_model_intensity(self, parameters):
        """
        Calculate model intensity including background and ARF correction.
        
        Args:
            parameters: Array of parameters (amplitudes, sigmas, and background params)
            
        Returns:
            numpy.ndarray: Total model intensity
        """
        # Split parameters
        amplitudes = parameters[:self.n_elements]
        sigmas = parameters[self.n_elements:2*self.n_elements]
        scale = parameters[-1]
        
        # Update data handler parameters
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
        
        # Generate model spectrum
        model_intensity = np.zeros_like(self.energies)
        for element, gaussian in self.data_handler.gaussian_models.items():
            amp = self.data_handler.element_amplitudes[element]
            model_intensity += gaussian(self.energies) * amp
        
        # Apply scale factor
        model_intensity *= scale
        
        # Add background
        if self.background is not None:
            background_flux = self.background(self.energies * u.keV)
            model_intensity += background_flux.value
        
        # Apply ARF correction
        model_intensity = self.apply_arf_correction(model_intensity)
        
        return model_intensity
    
    def init_optimizer(self):
        """Initialize optimizer-specific parameters."""
        # Load spectrum and ARF
        self.energies, self.counts = self.load_spectrum()
        self.load_arf()
        
        # Estimate background
        self.estimate_background()
        
        # Additional initialization can be added here
        
    def plot_optimization_history(self):
        """Plot optimization history including background fit."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
        
        # Plot data and total fit
        ax1.plot(self.energies, self.counts, 'k.', label='Data', alpha=0.5)
        model = self.calculate_model_intensity(self.best_solution)
        ax1.plot(self.energies, model, 'r-', label='Total Fit')
        
        # Plot background
        if self.background is not None:
            background = self.background(self.energies * u.keV)
            ax1.plot(self.energies, background, 'g--', label='Background')
        
        ax1.set_xlabel('Energy (keV)')
        ax1.set_ylabel('Counts')
        ax1.legend()
        
        # Plot residuals
        residuals = self.counts - model
        ax2.plot(self.energies, residuals, 'k.')
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Energy (keV)')
        ax2.set_ylabel('Residuals')
        
        plt.tight_layout()
        return fig, (ax1, ax2) 