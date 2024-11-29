import pygad
import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt

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
   
        self.energies, self.counts = self.data_handler.get_fits_data(fits_path)
        print("Energy: ",self.energies,self.counts)
        # Number of parameters to optimize
        self.n_elements = len(self.data_handler.elements)
        # For each element: one amplitude and one sigma
        self.n_params = self.n_elements * 2+1
        print("Params: ",self.n_params)
        
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
         # Create gene space with specific ranges for each parameter
        self.gene_space = []
        # Amplitude ranges
        for element in self.data_handler.elements:
            self.gene_space.append({'low':self.data_handler.bounds[element][0] , 'high': self.data_handler.bounds[element][1]})
        # Sigma ranges
        for _ in range(self.n_elements):
            self.gene_space.append({'low': 0, 'high': 2})
        self.gene_space.append({'low': 0, 'high': 10000})
        # Define parameters for PyGAD
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            sol_per_pop=50,
            num_parents_mating=self.num_parents_mating,
            num_genes=self.num_genes,
            # init_range_low=self.init_range_low,
            # init_range_high=self.init_range_high,
            gene_space=self.gene_space,  # Use gene_space instead of init_range

            fitness_func=self.fitness_func,
            mutation_type="random",
            mutation_percent_genes=10,
            on_generation=self.dispProgress
        )
    def dispProgress(self,ga_instance):
                # Update data handler with best parameters
        print(f"Generation {ga_instance.generations_completed}")
        print(f"Best fitness: {ga_instance.best_solution()[1]}")
        self.data_handler.plot_combined_spectrum(self.fits_path,(0,27))
        
        
    def fitness_func(self,ga_instance, solution, solution_idx):
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
        sigmas = solution[self.n_elements:-1]
        scale=solution[-1]
        # Update data handler parameters
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
        
        # Generate model spectrum
        model_intensity = np.zeros_like(self.energies)
        for element, gaussian in self.data_handler.gaussian_models.items():
            amp = self.data_handler.element_amplitudes[element]
            # print(gaussian(self.energies),"\n\n\n")
            count=np.sum(np.isnan(gaussian(self.energies)))
            if count!=0:
                print("NULL",element,count)

            # print("ENERGY...",element,np.sum(np.isnan(gaussian(self.energies))),"...ENERGY ....")
            model_intensity += gaussian(self.energies) * amp
        model_intensity*=scale
        
        # Calculate fitness (negative MSE)
        mse = np.mean((self.counts - model_intensity) ** 2)
        # print(amp,model_intensity,mse)
        return -mse
    
    def plot_fitness_history(self):
        """
        Plot the fitness history of the genetic algorithm optimization.
        Shows both best and mean fitness per generation.
        
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        self.ga_instance.plot_fitness()
        # Get fitness history
        # best_solutions = self.ga_instance.best_solutions_fitness
        # print(self.ga_instance.solutions_fitness)
        # mean_fitness = [-np.mean(gen_fitness) for gen_fitness in self.ga_instance.solutions_fitness]
        # generations = range(len(best_solutions))
        # print(best_solutions,best_solutions,mean_fitness)
        # # Plot best and mean fitness
        # ax.plot(generations, -np.array(best_solutions), 'b-', label='Best Fitness', linewidth=2)
        # ax.plot(generations, mean_fitness, 'r--', label='Mean Fitness', alpha=0.7)
        
        # ax.set_xlabel('Generation')
        # ax.set_ylabel('Fitness (-MSE)')
        # ax.set_title('Genetic Algorithm Optimization Progress')
        # ax.grid(True, alpha=0.3)
        # ax.legend()
        
        # plt.tight_layout()
        return fig, ax

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
        
        # Plot fitness history
        fig, ax = self.plot_fitness_history()
        plt.show()
        
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
    plt.savefig("Fitness_plot.png")
    return optimizer

if __name__ == "__main__":
    test_optimization("data\\ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits")
