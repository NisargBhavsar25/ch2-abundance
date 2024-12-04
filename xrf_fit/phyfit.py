import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from element_model import ElementModel  # Import ElementModel
from element_handler import ElementHandler
from astropy.io import fits  # Add this import at the top
import pygad  # Add this import

class PhyOptimizer:
    def __init__(self, el_handler, fits_path, bkg_path=None, x_range=None, method='ga'):
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
        
        # Add GA-specific initialization
        if method == 'ga':
            self.init_ga()

    def _setup_optimization_params(self):
        """Setup optimization parameters, bounds, and initial guess."""
        # Initialize bounds for concentrations
        conc_bounds = [(0.0, 50.0)] * self.n_elements  # Adjust range as needed
        
        # Initialize bounds for standard deviations
        std_dev_bounds = [(0.01, 1)] * self.n_elements  # Adjust range as needed
        
        # Scale factor bounds
        scale_bounds = [(1e-7, 1e-4)]
        
        self.bounds = conc_bounds + std_dev_bounds + scale_bounds
        
        # Initial guess
        self.initial_guess = []
        # Initial concentrations from element handler
        for element in self.el_handler.elements:
            self.initial_guess.append(self.el_handler.conc[element])
        
        # Initial standard deviations
        self.initial_guess.extend([0.1] * self.n_elements)
        
        # Initial scale
        self.initial_guess.append(1e-6)

    def init_ga(self):
        """Initialize genetic algorithm parameters."""
        self.num_generations = 100
        self.num_parents_mating = 4
        
        # Create gene space with specific ranges for each parameter
        self.gene_space = []
        # Concentration ranges
        for _ in range(self.n_elements):
            self.gene_space.append({'low': 0.0, 'high': 50.0})
        # Sigma ranges
        for _ in range(self.n_elements):
            self.gene_space.append({'low': 0.01, 'high': 1.0})
        # Scale factor range
        self.gene_space.append({'low': 1e-7, 'high': 1e-4})
        
        # Define GA instance
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            sol_per_pop=100,
            num_parents_mating=self.num_parents_mating,
            num_genes=self.n_params,
            gene_space=self.gene_space,
            fitness_func=self.ga_fitness_func,
            on_generation=self.on_generation
        )

    def ga_fitness_func(self, ga_instance, solution, solution_idx):
        """Fitness function for genetic algorithm."""
        # Calculate model intensity using the solution parameters
        model_intensity = self.calculate_model_intensity(solution)
        
        # Calculate fitness (negative of objective function)
        return -self.objective_function(solution)
    
    def on_generation(self, ga_instance):
        """Callback for each generation."""
        print(f"Generation {ga_instance.generations_completed}")
        print(f"Best fitness: {ga_instance.best_solution()[1]}")
        
        # Optionally plot current best solution
        best_solution = ga_instance.best_solution()[0]
        model_intensity = self.calculate_model_intensity(best_solution)
        plt.clf()
        plt.plot(self.energies, self.counts, 'b-', label='Data')
        plt.plot(self.energies, model_intensity, 'r-', label='Model')
        plt.legend()
        plt.pause(0.1)

    def objective_function(self, params):
        """Custom objective function for optimization."""
        try:
            # Update concentrations in element handler first
            for i, element in enumerate(self.el_handler.elements):
                self.el_handler.conc[element] = max(0.0, min(50.0, params[i]))
                self.el_handler.std_dev[element] = max(0.05, min(1.0, params[self.n_elements + i]))
            
            # Calculate model intensity using element handler
            model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
            model_intensity *= max(1e-7, min(1e-4, params[-1]))  # Constrain scale factor
            
            # Calculate error using vectorized operation
            error = np.sum((self.counts - model_intensity)**2)
            
            # Print less frequently (only every 100th iteration)
            if hasattr(self, '_iter_count'):
                self._iter_count += 1
            else:
                self._iter_count = 0
            
            if self._iter_count % 100 == 0:
                print(f"Iteration {self._iter_count}, Error: {error:.3f}")
            
            return error
        
        except Exception as e:
            print(f"Error in objective function: {e}")
            return 1e10  # Return large error for invalid parameter combinations

    def calculate_model_intensity(self, params):
        """Calculate model intensity for given parameters."""
        # Update concentrations and std_devs in element handler
        for i, element in enumerate(self.el_handler.elements):
            self.el_handler.conc[element] = np.clip(params[i], 0.0, 50.0)
            self.el_handler.std_dev[element] = np.clip(params[self.n_elements + i], 0.01, 1.0)
        
        # Calculate model intensity
        model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
        scale = np.clip(params[-1], 1e-7, 1e-4)
        model_intensity *= scale
        
        return model_intensity

    def calculate_residuals(self, params):
        """Calculate residuals for optimization."""
        model = self.calculate_model_intensity(params)
        return self.counts - model

    def run_optimization(self):
        """Run optimization with specified method."""
        if self.method == 'ga':
            self.ga_instance.run()
            solution = self.ga_instance.best_solution()[0]
            fitness = self.ga_instance.best_solution()[1]
            self.result = type('Result', (), {'x': solution, 'fun': -fitness})
            return solution, -fitness
        
        elif self.method == 'leastsq':
            # Use least_squares solver with proper residuals
            result = optimize.least_squares(
                self.calculate_residuals,
                self.initial_guess,
                bounds=tuple(zip(*self.bounds)),
                method='trf',
                loss='linear',
                verbose=2
            )
        else:
            # For other methods, use minimize with sum of squared residuals
            result = optimize.minimize(
                lambda x: np.sum(self.calculate_residuals(x)**2),
                self.initial_guess,
                method=self.method,
                bounds=self.bounds,
                options={'maxiter': 500}
            )

        self.result = result
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
        plt.savefig(f'phyFit_result_{model_name}.png')
        plt.close()
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
import pandas as pd


if __name__ == "__main__":


    el_handler = ElementHandler(num_channels=2048,verbose=False)
    
    ca_al_list = pd.read_csv("high_confidence_ca_al.csv")
    fits_path =  ca_al_list.iloc[1]["filename"]  # Replace with your default path
    
    # Optional: Still allow command-line override
    try:
        import argparse
        parser = argparse.ArgumentParser(description="Process fits file path.")
        parser.add_argument("fits_path", type=str, help="Path to the FITS file", nargs='?', default=fits_path)
        args = parser.parse_args()
        fits_path = args.fits_path
    except:
        pass

    # Initialize optimizer
    optimizer = PhyOptimizer(el_handler, fits_path, method="leastsq")

    # Run optimization
    solution, residual = optimizer.run_optimization()
    print("Residual:", residual)
    
    # Plot results
    optimizer.plot_result("Physical Model Fit")
    
    # Print detailed results