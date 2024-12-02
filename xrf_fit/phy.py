import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from element_model import ElementModel  # Import ElementModel
from element_handler import ElementHandler
from astropy.io import fits  # Add this import at the top

class PhyOptimizer:
    def __init__(self, el_handler, fits_path, bkg_path=None, x_range=None, method='leastsq'):
        """
        Initialize optimizer using scipy.optimize methods.
        
        Args:
            el_handler (DataHandler): Instance of DataHandler class
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
        # Initialize bounds for concentrations
        conc_bounds = [(0.0, 50.0)] * self.n_elements  # Adjust range as needed
        
        # Initialize bounds for standard deviations
        std_dev_bounds = [(0.01, 1)] * self.n_elements  # Adjust range as needed
        
        # Scale factor bounds
        scale_bounds = [(1e-7, 1e-2)]
        
        self.bounds = conc_bounds + std_dev_bounds + scale_bounds
        
        # Initial guess
        self.initial_guess = []
        # Initial concentrations from element handler
        for element in self.el_handler.elements:
            self.initial_guess.append(self.el_handler.conc[element])
        
        # Initial standard deviations
        self.initial_guess.extend([0.1] * self.n_elements)
        
        # Initial scale
        self.initial_guess.append(1e-5)

    def objective_function(self, params):
        """Custom objective function for optimization."""
        # Update concentrations in element handler
        for i, element in enumerate(self.el_handler.elements):
            self.el_handler.conc[element] = params[i]
            self.el_handler.std_dev[element] = params[self.n_elements + i]
        
        # Calculate model intensity using element handler
        model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
        model_intensity *= params[-1]  # Scale factor
        
        # Calculate residuals only for valid bins
        residuals = np.zeros_like(self.counts)
        for element in self.el_handler.elements:
            for line_group in self.el_handler.element_models[element].line_div.values():
                for line in line_group:
                    energy_mean = self.el_handler.element_models[element].energy_dict[line[:2]]["mean"]
                    std_dev = self.el_handler.std_dev[element]
                    
                    # Calculate bin range
                    binstart = int((energy_mean - std_dev) / self.kev_per_channel)
                    binend = int((energy_mean + std_dev) / self.kev_per_channel)
                    
                    # Ensure bins are within valid range
                    binstart = max(0, binstart)
                    binend = min(len(self.counts), binend)
                    
                    # Calculate residuals for this range
                    residuals[binstart:binend] = np.abs(self.counts[binstart:binend] - model_intensity[binstart:binend])
        print("Error",np.sum(np.abs(residuals**4)))
        print(np.max(self.counts),np.max(model_intensity))
        return np.sum(np.abs(residuals**2))  # Return only non-zero residuals

    def calculate_model_intensity(self, params):
        """Calculate model intensity for given parameters."""
        # Update concentrations and std_devs in element handler
        for i, element in enumerate(self.el_handler.elements):
            self.el_handler.conc[element] = params[i]
            self.el_handler.std_dev[element] = params[self.n_elements + i]
        
        # Calculate model intensity
        model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
        model_intensity *= params[-1]  # Scale factor
        
        return model_intensity

    def run_optimization(self):
        """Run optimization with specified method."""
        if self.method == 'leastsq':
            result = optimize.least_squares(
                self.objective_function,
                self.initial_guess,
                bounds=tuple(zip(*self.bounds)),
                method='trf',
                loss='linear',
                verbose=2,
                ftol=1e-9,
                xtol=1e-8
            )
        elif self.method == 'levenberg':
            result = optimize.root(
                self.objective_function,
                self.initial_guess,
                method='lm',
                options={'maxiter': 500}
            )
        else:
            result = optimize.minimize(
                self.objective_function,
                self.initial_guess,
                method=self.method,
                bounds=self.bounds,
                options={'maxiter': 500}
            )

        self.result = result
        print(result)
        return result.x, result.fun if hasattr(result, 'fun') else None

    def plot_result(self, model_name=""):
        """Plot the optimized fit result."""
        fig, ax = plt.subplots(figsize=(10, 6))
        model_intensity = self.calculate_model_intensity(self.result.x)
        print(np.max(self.counts),np.max(model_intensity))
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

# Example usage:
# handler = DataHandler(bkg_path="path/to/background.fits")
# optimizer = PhyOptimizer(handler, "path/to/fits.fits", "path/to/background.fits")
# amplitudes, sigmas = optimizer.fit_from_files("path/to/fits.fits", "path/to/background.fits")
if __name__ == "__main__":

    # Initialize ElementHandler
    el_handler = ElementHandler(num_channels=1024)

    # Initialize optimizer
    import argparse
    parser = argparse.ArgumentParser(description="Process fits file path.")
    parser.add_argument("fits_path", type=str, help="Path to the FITS file")
    args = parser.parse_args()
    optimizer = PhyOptimizer(el_handler, args.fits_path,method="leastsq")

    # Run optimization
    solution, residual = optimizer.run_optimization()
    print("Residual",residual)
    print("Solution",solution)
    # calc_intensities = optimizer.calculate_model_intensity(solution)
    # el_handler.plot_intensity(energies=optimizer.energies,intensities=calc_intensities)
    # Plot results
    optimizer.plot_result("Physical Model Fit")

    # Extract fitted parameters
    concentrations = solution[:len(el_handler.elements)]
    std_devs = solution[len(el_handler.elements):-1]
    scale = solution[-1]

    # Print results
    for element, conc, std in zip(el_handler.elements, concentrations, std_devs):
        print(f"{element}: Concentration = {conc:.3f}, Std Dev = {std:.3f}")
    
    for element in el_handler.elements:
        for line in el_handler.element_models[element].lines:
            print(f"{element} {line}: Energy = {el_handler.element_models[element].energy_dict[line[:2]][line]}, Intensity = {el_handler.calc_absolute_intensity(element,line)}")
    
    print(f"Scale factor: {scale:.7f}")
