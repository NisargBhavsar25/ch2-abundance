import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from element_model import ElementModel
from element_handler import ElementHandler
from astropy.io import fits

class PhyOptimizer:
    def __init__(self, el_handler, fits_path, bkg_path=None, x_range=None, method='leastsq'):
        """
        Initialize optimizer using scipy.optimize methods.
        
        Args:
            el_handler (ElementHandler): Instance of ElementHandler class
            fits_path (str): Path to the FITS file to fit
            bkg_path (str): Path to background file
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
            method (str): Optimization method ('leastsq', 'levenberg', 'trust-ncg', etc.)
        """
        self.el_handler = el_handler
        self.fits_path = fits_path
        self.x_range = x_range
        self.method = method
        
        # Determine energy conversion factor based on number of channels
        with fits.open(fits_path) as hdul:
            num_channels = len(hdul[1].data['COUNTS'])
            self.kev_per_channel = 0.01385 if num_channels == 2048 else 0.0277
            
        self.energies, self.counts = self.get_fits_data(fits_path)
        
        # Number of parameters to optimize
        self.n_elements = len(self.el_handler.elements)
        # For each element: one amplitude and one sigma, plus global scale
        self.n_params = self.n_elements * 2 + 1
        
        # Initialize bounds and initial guess
        self._setup_optimization_params()

    def _setup_optimization_params(self):
        """Setup optimization parameters, bounds, and initial guess."""
        # Initialize empty lists for bounds and initial guesses
        conc_bounds = []
        std_dev_bounds = []
        self.initial_guess = []
        
        # Get maximum counts for scaling
        max_counts = np.max(self.counts)
        print(f"Maximum counts in data: {max_counts}")

        # For each element
        for element in self.el_handler.elements:
            # --- Concentration parameters ---
            max_peak_intensity = 0
            el_model = self.el_handler.element_models[element]
            
            # Find strongest peak for initial guess
            for line_type, lines in el_model.line_div.items():
                for line in lines:
                    try:
                        energy = el_model.energy_dict[line_type]["mean"]
                        idx = np.abs(self.energies - energy).argmin()
                        intensity = self.counts[idx]
                        max_peak_intensity = max(max_peak_intensity, intensity)
                    except:
                        continue
            
            # Set concentration bounds and initial guess
            conc_bounds.append((0.0, 1.0))  # Updated concentration bounds
            initial_conc = max_peak_intensity / max_counts if max_peak_intensity > 0 else 0.1  # Updated initial guess
            self.initial_guess.append(initial_conc)
            
            # --- Standard deviation parameters ---
            # Use detector resolution for std dev bounds
            fwhm_kev = 0.15  # Typical detector FWHM in keV
            sigma = fwhm_kev / 2.355  # Convert FWHM to sigma
            
            std_dev_bounds.append((sigma * 0.5, sigma * 2.0))
            self.initial_guess.append(sigma)
        
        # --- Scale factor parameter ---
        # Scale bounds based on counts
        scale_factor = 1.0  # Updated scale factor initial guess
        scale_bounds = [(0.1, 10.0)]  # Updated scale factor bounds
        self.initial_guess.append(scale_factor)
        
        # Combine all bounds
        self.bounds = conc_bounds + std_dev_bounds + scale_bounds
        
        # Verify and fix any out-of-bounds values
        for i, (guess, (lower, upper)) in enumerate(zip(self.initial_guess, self.bounds)):
            if not (lower <= guess <= upper):
                # Fix the guess to be within bounds
                self.initial_guess[i] = (lower + upper) / 2
        
        # Debug information
        print(f"Number of elements: {len(self.el_handler.elements)}")
        print(f"Total parameters: {len(self.initial_guess)}")
        print(f"Parameters per element: 2 (concentration and std_dev)")
        print("\nBounds:")
        for i, ((lower, upper), guess) in enumerate(zip(self.bounds, self.initial_guess)):
            if i < len(self.el_handler.elements):
                param_type = f"Concentration for {self.el_handler.elements[i]}"
            elif i < 2 * len(self.el_handler.elements):
                param_type = f"Std Dev for {self.el_handler.elements[i - len(self.el_handler.elements)]}"
            else:
                param_type = "Scale factor"
            print(f"{param_type}: [{lower:.2e}, {upper:.2e}], initial: {guess:.2e}")

    def objective_function(self, params):
        """Calculate fitting objective using full spectrum comparison."""
        # Update concentrations in element handler
        try:
            # Update element handler parameters
            for i, element in enumerate(self.el_handler.elements):
                self.el_handler.conc[element] = max(0, params[i])  # Ensure non-negative
                self.el_handler.std_dev[element] = max(0.01, params[self.n_elements + i])  # Ensure non-zero
            
            # Calculate model intensity
            model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
            model_intensity *= max(1e-10, params[-1])  # Ensure non-zero scale
            
            # Calculate residuals with proper weighting
            # Add small constant to avoid division by zero
            weights = 1.0 / np.sqrt(self.counts + 1)  # Poisson statistics
            residuals = (self.counts - model_intensity) * weights
            
            # Add extra weight to peak regions
            peak_weights = np.ones_like(self.counts)
            for i, element in enumerate(self.el_handler.elements):
                for line_group in self.el_handler.element_models[element].line_div.values():
                    for line in line_group:
                        energy_mean = self.el_handler.element_models[element].energy_dict[line[:2]]["mean"]
                        std_dev = params[self.n_elements + i]
                        
                        # Calculate peak region
                        binstart = int((energy_mean - 3*std_dev) / self.kev_per_channel)
                        binend = int((energy_mean + 3*std_dev) / self.kev_per_channel)
                        
                        # Ensure bins are within valid range
                        binstart = max(0, binstart)
                        binend = min(len(peak_weights), binend)
                        
                        # Increase weight in peak regions
                        peak_weights[binstart:binend] = 2.0
            
            residuals *= peak_weights
            
            # Print diagnostics occasionally
            if np.random.random() < 0.1:  # Print ~10% of iterations
                print(f"Max model: {np.max(model_intensity):.2f}, Max data: {np.max(self.counts):.2f}")
                print(f"Current cost: {np.sum(residuals**2):.2e}")
            
            return residuals
            
        except Exception as e:
            print(f"Error in objective function: {e}")
            return np.ones_like(self.counts) * 1e6

    def run_optimization(self):
        """Run optimization with specified method."""
        if self.method == 'leastsq':
            # Verify dimensions match
            if len(self.initial_guess) != len(self.bounds):
                raise ValueError(f"Dimension mismatch: {len(self.initial_guess)} initial values but {len(self.bounds)} bounds")
            
            # Convert bounds to form needed by least_squares
            bounds_min = [b[0] for b in self.bounds]
            bounds_max = [b[1] for b in self.bounds]

            result = optimize.least_squares(
                self.objective_function,
                self.initial_guess,
                bounds=(bounds_min, bounds_max),
                jac='3-point',
                loss='linear',
                max_nfev=200,
                ftol=1e-4,
                xtol=1e-4,
                verbose=2 
            )   
        elif self.method == 'levenberg':
            result = optimize.root(
                self.objective_function,
                self.initial_guess,
                method='lm',
                options={'maxiter': 1000}  # Increased maximum iterations
            )
        else:
            result = optimize.minimize(
                self.objective_function,
                self.initial_guess,
                method=self.method,
                bounds=self.bounds,
                options={'maxiter': 1000}  # Increased maximum iterations
            )

        self.result = result
        print(result)
        return result.x, result.fun if hasattr(result, 'fun') else None

    def plot_result(self, model_name=""):
        """Plot the optimized fit result."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Update parameters to final values
        for i, element in enumerate(self.el_handler.elements):
            self.el_handler.conc[element] = self.result.x[i]
            self.el_handler.std_dev[element] = self.result.x[self.n_elements + i]
            
        # Calculate final model
        model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
        model_intensity *= self.result.x[-1]
        
        ax.plot(self.energies, self.counts, 'b-', label='Data')
        ax.plot(self.energies, model_intensity, 'r-', label='Model Fit')
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Fit Result: {model_name}')
        ax.legend()
        plt.show()
        return fig

    def fit_from_files(self, fits_path, bkg_path=None, x_range=None, method='leastsq'):
        """
        Fit the data from the provided FITS and background FITS files.
        
        Args:
            fits_path (str): Path to the FITS file to fit.
            bkg_path (str, optional): Path to the background FITS file (default is None).
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit (default is None).
            method (str): Optimization method ('leastsq', 'levenberg', etc.).
        
        Returns:
            tuple: (amplitudes, sigmas) - Optimized element amplitudes and sigmas.
        """
        # Initialize optimizer with the provided paths
        self.fits_path = fits_path
        self.bkg_path = bkg_path
        self.x_range = x_range
        self.method = method
        
        # Run optimization
        solution, _ = self.run_optimization()
        
        # Extract amplitudes and sigmas
        amplitudes = solution[:self.n_elements]
        sigmas = solution[self.n_elements:-1]  # Assuming the last parameter is scale
        
        return amplitudes, sigmas

    def get_fits_data(self, fits_path):
        """
        Read spectral data from a FITS file.
        
        Args:
            fits_path (str): Path to the FITS file
            
        Returns:
            tuple: (energies, counts) arrays
        """
        with fits.open(fits_path) as hdul:
            data = hdul[1].data
            counts = data['COUNTS']
            # Calculate energies based on channel numbers
            channels = np.arange(len(counts))
            energies = channels * self.kev_per_channel
            
            if self.x_range is not None:
                min_kev, max_kev = self.x_range
                mask = (energies >= min_kev) & (energies <= max_kev)
                energies = energies[mask]
                counts = counts[mask]
                
            return energies, counts

if __name__ == "__main__":
    # Initialize ElementHandler
    el_handler = ElementHandler(num_channels=1024)

    # Initialize optimizer
    import argparse
    parser = argparse.ArgumentParser(description="Process fits file path.")
    parser.add_argument("fits_path", type=str, help="Path to the FITS file")
    args = parser.parse_args()
    optimizer = PhyOptimizer(el_handler, args.fits_path, method="leastsq")

    # Run optimization
    solution, residual = optimizer.run_optimization()
    print("Residual", residual)
    print("Solution", solution)
    
    # Plot results
    optimizer.plot_result("Physical Model Fit")

    # Extract fitted parameters
    concentrations = solution[:len(el_handler.elements)]
    std_devs = solution[len(el_handler.elements):-1]
    scale = solution[-1]

    # Print results
    for element, conc, std in zip(el_handler.elements, concentrations, std_devs):
        print(f"{element}: Concentration = {conc:.3f}, Std Dev = {std:.3f}")
    print(f"Scale factor: {scale:.3f}")