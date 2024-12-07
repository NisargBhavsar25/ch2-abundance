import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Optimizer(ABC):
    def __init__(self, data_handler, fits_path, x_range=None):
        """
        Base class for spectrum optimization.
        
        Args:
            data_handler (DataHandler): Instance of DataHandler class
            fits_path (str): Path to the FITS file to fit
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
        """
        self.data_handler = data_handler
        self.fits_path = fits_path
        self.x_range = x_range
        
        # Load data
        self.energies, self.counts = self.data_handler.get_fits_data(fits_path)
        
        # Number of parameters to optimize
        self.n_elements = len(self.data_handler.elements)
        self.n_params = self.n_elements * 2  # For each element: amplitude and sigma
        
        # Initialize optimizer-specific parameters
        self.init_optimizer()
    
    @abstractmethod
    def init_optimizer(self):
        """Initialize optimizer-specific parameters. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def run_optimization(self):
        """
        Run the optimization algorithm.
        
        Returns:
            tuple: (best_solution, best_fitness)
        """
        pass
    
    def calculate_model_intensity(self, parameters):
        """
        Calculate model intensity for given parameters.
        
        Args:
            parameters: Array of parameters (amplitudes and sigmas)
            
        Returns:
            numpy.ndarray: Model intensity
        """
        amplitudes = parameters[:self.n_elements]
        sigmas = parameters[self.n_elements:]
        
        # Update data handler parameters
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
        
        # Generate model spectrum
        model_intensity = np.zeros_like(self.energies)
        for element, gaussian in self.data_handler.gaussian_models.items():
            amp = self.data_handler.element_amplitudes[element]
            model_intensity += gaussian(self.energies) * amp
        
        return model_intensity
    
    def calculate_fitness(self, parameters):
        """
        Calculate fitness (negative MSE) for given parameters.
        
        Args:
            parameters: Array of parameters (amplitudes and sigmas)
            
        Returns:
            float: Fitness value (negative mean squared error)
        """
        model_intensity = self.calculate_model_intensity(parameters)
        mse = np.mean((self.counts - model_intensity) ** 2)
        return -mse
    
    def plot_result(self):
        """
        Plot the optimized fit result.
        
        Returns:
            tuple: (fig, (ax1, ax2)) matplotlib figure and axes objects
        """
        fig, (ax1, ax2) = self.data_handler.plot_combined_spectrum(
            self.fits_path, 
            x_range=self.x_range,
            normalize=False
        )
        return fig, (ax1, ax2)
    
    @abstractmethod
    def plot_optimization_history(self):
        """Plot the optimization history. Must be implemented by subclasses."""
        pass 