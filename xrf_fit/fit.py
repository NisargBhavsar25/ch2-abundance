import pygad
import numpy as np
from .data_handler import DataHandler

class GaussianOptimizer:
    def __init__(self, data_handler, fits_path, x_range=None):
        """
        Initialize optimizer for Gaussian parameters.
        
        Args:
            data_handler (DataHandler): Instance of DataHandler class
            fits_path (str): Path to the FITS file to fit
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
        """
        self.data_handler = data_handler
        self.fits_path = fits_path
        self.x_range = x_range
        
        # Get experimental data
        _, _, self.energies, self.counts = self.data_handler.plot_fits_data(fits_path, x_range)
        
        # Number of parameters to optimize
        self.n_elements = len(self.data_handler.elements)
        # For each element: one amplitude and one sigma
        self.n_params = self.n_elements * 2
        
        # Initialize GA parameters
        self.init_ga()
    
    def init_ga(self):
        """Initialize genetic algorithm parameters."""
        self.num_generations = 100
        self.num_parents_mating = 4
        
        # GA parameters
        self.init_range_low = 0.0
        self.init_range_high = 2.0
        self.num_genes = self.n_params
        
        # Define parameters for PyGAD
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            num_parents_mating=self.num_parents_mating,
            num_genes=self.num_genes,
            init_range_low=self.init_range_low,
            init_range_high=self.init_range_high,
            fitness_func=self.fitness_func,
            mutation_type="random",
            mutation_percent_genes=10
        )
    
    def fitness_func(self, solution, solution_idx):
        """
        Fitness function for genetic algorithm.
        
        Args:
            solution: Array of parameters (amplitudes and sigmas)
            solution_idx: Index of the solution (not used)
            
        Returns:
            float: Fitness value (negative mean squared error)
        """
        # Split solution into amplitudes and sigmas
        amplitudes = solution[:self.n_elements]
        sigmas = solution[self.n_elements:]
        
        # Update data handler parameters
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
        
        # Generate model spectrum
        model_intensity = np.zeros_like(self.energies)
        for element, gaussian in self.data_handler.gaussian_models.items():
            amp = self.data_handler.element_amplitudes[element]
            model_intensity += gaussian(self.energies) * amp
        
        # Calculate fitness (negative MSE)
        mse = np.mean((self.counts - model_intensity) ** 2)
        return -mse
    
    def run_optimization(self):
        """
        Run the genetic algorithm optimization.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        self.ga_instance.run()
        solution, solution_fitness, _ = self.ga_instance.best_solution()
        
        # Update data handler with best parameters
        amplitudes = solution[:self.n_elements]
        sigmas = solution[self.n_elements:]
        
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
        
        return solution, solution_fitness
    
    def plot_result(self):
        """Plot the optimized fit result."""
        fig, (ax1, ax2) = self.data_handler.plot_combined_spectrum(
            self.fits_path, 
            x_range=self.x_range,
            normalize=False  # Don't normalize since we optimized the actual amplitudes
        )
        return fig, (ax1, ax2)

def test_optimization(fits_path, x_range=(2, 8)):
    """
    Test function to demonstrate the optimization.
    
    Args:
        fits_path (str): Path to FITS file
        x_range (tuple): Energy range to fit
    """
    # Create data handler and optimizer
    handler = DataHandler()
    optimizer = GaussianOptimizer(handler, fits_path, x_range)
    
    # Run optimization
    solution, fitness = optimizer.run_optimization()
    
    # Print results
    print("\nOptimization Results:")
    print("Best fitness (negative MSE):", fitness)
    print("\nOptimized Parameters:")
    for i, element in enumerate(handler.elements):
        print(f"{element}:")
        print(f"  Amplitude: {solution[i]:.3f}")
        print(f"  Sigma: {solution[i + len(handler.elements)]:.3f}")
    
    # Plot result
    fig, (ax1, ax2) = optimizer.plot_result()
    
    return optimizer

if __name__ == "__main__":
    test_optimization("path/to/your/file.fits")
