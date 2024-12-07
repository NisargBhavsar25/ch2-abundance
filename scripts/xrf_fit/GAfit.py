import pygad
import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt
from numba import jit

class GaussianOptimizer:
    def __init__(self, data_handler, fits_path, bkg_path=None, x_range=None):
        """
        Initialize optimizer for Gaussian parameters.
        
        Args:
            data_handler (DataHandler): Instance of DataHandler class
            fits_path (str): Path to the FITS file to fit
            bkg_path (str, optional): Path to background file (default is None)
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
        """
        self.data_handler = data_handler
        self.fits_path = fits_path
        self.bkg_path = bkg_path
        self.x_range = x_range
        self.energies, self.counts = self.data_handler.get_fits_data(fits_path)
        print("Energy: ",self.energies,self.counts)
        self.counts=np.array(self.counts)
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
            self.gene_space.append({'low': self.data_handler.bounds[element][0],
                                  'high': self.data_handler.bounds[element][1]})
        # Sigma ranges
        for _ in range(self.n_elements):
            self.gene_space.append({'low': 0, 'high': 0.1})
        self.gene_space.append({'low': 0, 'high': 10000})
        
        # Define parameters for PyGAD
        self.ga_instance = pygad.GA(
            num_generations=self.num_generations,
            sol_per_pop=50,
            num_parents_mating=self.num_parents_mating,
            num_genes=self.num_genes,
            gene_space=self.gene_space,
            fitness_func=self.fitness_func,
            # mutation_type="random",
            # mutation_percent_genes=10,
            crossover_type=self.custom_crossover,  # Use our custom crossover
            on_generation=self.dispProgress
        )
    def dispProgress(self,ga_instance):
                # Update data handler with best parameters
        print(f"Generation {ga_instance.generations_completed}")
        print(f"Best fitness: {ga_instance.best_solution()[1]}")
        self.calculate_model_intensity(ga_instance.best_solution()[0])
        self.data_handler.plot_combined_spectrum(self.fits_path,(0,27))
        
    def calculate_model_intensity(self,solution):
                # Generate model spectrum
         # Split solution into amplitudes and sigmas
        amplitudes = solution[:self.n_elements]
        sigmas = solution[self.n_elements:-1]
        scale=solution[-1]
        self.data_handler.scale=scale
        # Update data handler parameters
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = max(1e-9,sigmas[i])
        
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
        # model_intensity=self.data_handler.add_background(self.bkg_path,self.energies,model_intensity)
        return model_intensity


        
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
        model_intensity=self.calculate_model_intensity(solution=solution)
    
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

    def calculate_section_fitness(self, solution, energy_range):
        """
        Calculate fitness (negative chi-square) for a specific energy range.
        
        Args:
            solution: Array of parameters
            energy_range: Tuple of (min_energy, max_energy)
            
        Returns:
            float: Fitness value for the section
        """
        # Get model intensity for full range

        model_intensity = self.calculate_model_intensity(solution=solution)
        
        # Find indices for the energy range
        range_mask = (self.energies >= energy_range[0]) & (self.energies <= energy_range[1])
        
        # Calculate chi-square for this section
        section_counts = self.counts[range_mask]
        section_model = model_intensity[range_mask]
        section_errors = np.sqrt(np.abs(section_counts))  # Poisson errors
        
        chi_square = np.sum(((section_counts - section_model) / (section_errors+1e-6)) ** 2)
        return -chi_square  # Negative because we want to maximize fitness
    
    @jit(nopython=True)
    def calculate_section_fitness_numba(self, solution, e_range, energies, counts):
        """
        Calculate fitness (negative chi-square) for a specific energy range using Numba.
        """
        model_intensity = self.calculate_model_intensity(solution)
        range_mask = (energies >= e_range[0]) & (energies <= e_range[1])
        
        section_counts = counts[range_mask]
        section_model = model_intensity[range_mask]
        section_errors = np.sqrt(np.abs(section_counts))  # Poisson errors
        
        chi_square = np.sum(((section_counts - section_model) / (section_errors + 1e-6)) ** 2)
        return -chi_square  # Negative because we want to maximize fitness

    def custom_crossover(self, parents, offspring_size, ga_instance):
        """
        Custom crossover function that optimizes based on element-specific regions.
        
        Args:
            parents: Array of selected parents
            offspring_size: Tuple of (n_offspring, n_params)
            ga_instance: Instance of the pygad.GA class
            
        Returns:
            numpy.ndarray: Array of offspring
        """
        # Extract necessary data from self
        energies = self.energies
        counts = self.counts
        n_elements = self.n_elements
        
        # Call the JIT-compiled crossover function
        return self._jit_custom_crossover(parents, offspring_size, energies, counts, n_elements)

    @jit(nopython=True)
    def _jit_custom_crossover(self, parents, offspring_size, energies, counts, n_elements):
        """
        JIT-compiled custom crossover function.
        """
        offspring = np.empty(offspring_size, dtype=np.float64)  # Ensure dtype is float64
        
        # Define energy ranges for each element
        element_ranges = np.empty((n_elements, 2, 2), dtype=np.float64)  # Ensure dtype is float64
        for i in range(n_elements):
            means = np.array([0.0], dtype=np.float64)  # Placeholder for means, replace with actual means if needed
            element_ranges[i, 0, 0] = means[0] - 0.05
            element_ranges[i, 0, 1] = means[0] + 0.05
        
        # Pre-calculate fitness for parents
        fitness_parents = np.empty(parents.shape[0], dtype=np.float64)  # Ensure dtype is float64
        for idx in range(parents.shape[0]):
            fitness_parents[idx] = self.fitness_func(None, parents[idx], idx)  # This may need to be adjusted for JIT

        # For each offspring
        for k in range(offspring_size[0]):
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]
            offspring[k] = parent1.copy()  # Initialize offspring with parent1's genes
            
            # For each element
            for i in range(n_elements):
                amp_idx = i
                sigma_idx = i + n_elements
                
                # Calculate fitness for each parent in each energy range for this element
                total_fitness_p1 = 0.0
                total_fitness_p2 = 0.0
                
                for e_range in element_ranges[i]:
                    total_fitness_p1 += self.calculate_section_fitness_numba(parent1, e_range, energies, counts)
                    total_fitness_p2 += self.calculate_section_fitness_numba(parent2, e_range, energies, counts)
                
                # Choose better parent's parameters for this element
                if total_fitness_p2 > total_fitness_p1:
                    offspring[k, amp_idx] = parent2[amp_idx] * (1 + np.random.normal(0, 0.01))
                    crossover_weight = np.random.random()
                    offspring[k, sigma_idx] = (crossover_weight * parent1[sigma_idx] + 
                                                (1 - crossover_weight) * parent2[sigma_idx])
            
            # Handle scale parameter (last parameter)
            scale_idx = -1
            total_fitness = abs(fitness_parents[parent1_idx]) + abs(fitness_parents[parent2_idx])
            w1 = abs(fitness_parents[parent1_idx]) / total_fitness if total_fitness > 0 else 0.5
            
            offspring[k, scale_idx] = w1 * parent1[scale_idx] + (1 - w1) * parent2[scale_idx]
        
        return offspring

def test_optimization(fits_path, bkg_path=None, x_range=(0, 27)):
    """
    Test function to demonstrate the optimization.
    
    Args:
        fits_path (str): Path to FITS file
        bkg_path (str, optional): Path to background file (default is None)
        x_range (tuple): Energy range to fit
    """
    # Create data handler and optimizer
    handler = DataHandler(bkg_path=bkg_path)
    optimizer = GaussianOptimizer(handler, fits_path, bkg_path, x_range)
    
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
    # test_optimization("data\\ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits","data\\ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits")
    test_optimization("data\\ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits")
