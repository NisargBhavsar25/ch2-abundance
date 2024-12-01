import pygad
import numpy as np
from data_handler import DataHandler
import matplotlib.pyplot as plt

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
    
    def custom_crossover(self, parents, offspring_size, ga_instance):
        """
        Custom crossover function that optimizes based on element-specific regions.
        
        Args:
            parents: Array of selected parents
            offspring_size: Tuple of (n_offspring, n_params)
            
        Returns:
            numpy.ndarray: Array of offspring
        """
        offspring = np.empty(offspring_size)
        
        # Define energy ranges for each element
        element_ranges = {}
        for element, gaussian in self.data_handler.gaussian_models.items():
            # Get mean energies for this element
            means = list(gaussian.means.values())
            # Create range around means (±0.5 keV for example)
            element_ranges[element] = [(e - 0.05, e + 0.05) for e in means]
        
        # For each offspring
        for k in range(offspring_size[0]):
            # Select two parents
            parent1_idx = k % parents.shape[0]
            parent2_idx = (k + 1) % parents.shape[0]
            
            parent1 = parents[parent1_idx]
            parent2 = parents[parent2_idx]
            
            # Initialize offspring with parent1's genes
            offspring[k] = parent1.copy()
            
            # For each element
            for i, element in enumerate(self.data_handler.elements):
                # Get amplitude and sigma indices for this element
                amp_idx = i
                sigma_idx = i + self.n_elements
                
                # Calculate fitness for each parent in each energy range for this element
                total_fitness_p1 = 0
                total_fitness_p2 = 0
                
                for e_range in element_ranges[element]:
                    # Create temporary solutions with just this element's parameters
                    temp_sol1 = parent1.copy()
                    temp_sol2 = parent2.copy()
                    
                    # Calculate fitness for each parent in this range
                    fitness_p1 = self.calculate_section_fitness(temp_sol1, e_range)
                    fitness_p2 = self.calculate_section_fitness(temp_sol2, e_range)
                    
                    total_fitness_p1 += fitness_p1
                    total_fitness_p2 += fitness_p2
                
                # Choose better parent's parameters for this element
                if total_fitness_p2 > total_fitness_p1:
                    # Only take amplitude from better parent
                    offspring[k, amp_idx] = parent2[amp_idx]
                    # offspring[k, amp_idx]+=np.random.normal(0, 0.01)
                    offspring[k, amp_idx] *= (1 + np.random.normal(0, 0.01))

                    # For sigma, do standard crossover (average of parents with random weight)
                    crossover_weight = np.random.random()
                    offspring[k, sigma_idx] = (crossover_weight * parent1[sigma_idx] + 
                                            (1 - crossover_weight) * parent2[sigma_idx])
            
            # Handle scale parameter (last parameter)
            scale_idx = -1
            # Calculate weights based on parents' fitness values
            
            fitness_p1 = self.fitness_func(None, parent1, parent1_idx)
            fitness_p2 = self.fitness_func(None, parent2, parent2_idx)
            
            # Convert fitness values to weights (ensure positive)
            total_fitness = abs(fitness_p1) + abs(fitness_p2)
            if total_fitness > 0:
                w1 = abs(fitness_p1) / total_fitness
            else:
                w1 = 0.5  # Equal weights if both fitnesses are zero
                
            # Weighted average of scale parameters
            offspring[k, scale_idx] = w1 * parent1[scale_idx] + (1-w1) * parent2[scale_idx]
            
            
            # eid=0
            # for element, e_ranges in element_ranges.items():
            #     amp=offspring[k][eid]
            #     sdev=offspring[k][self.n_elements+eid]
            #     self.data_handler.gaussian_models[element].std_devs[:] = sdev
            #     intensity = self.data_handler.gaussian_models[element](self.energies)*amp
            #     # print(element,amp,sdev)
            #     # intensities=self.calculate_model_intensity(offspring[k])
            #     # print("itensities:",intensity)
            #     for e_range in e_ranges:
            #         # Calculate fitness for the offspring in this range
            #         fitness_offspring = self.calculate_section_fitness(offspring[k], e_range)
            #         # Identify poor fitness values
            #         if fitness_offspring < -4:  # Assuming 0.5 is the threshold for poor fitness
            #             range_mask = (self.energies >= e_range[0]) & (self.energies <= e_range[1])
            #             # print(np.where(range_mask),"MASK",fitness_offspring)
            #             if sum(self.counts[np.where(range_mask)])>sum(intensity[np.where(range_mask)]):
            #                 # print("max",max(intensity[range_mask]))
            #             # Modify the offspring's value based on the intended curve
            #             # if intended_curve:
            #                 # Add a positive value to increase the curve
            #                 offspring[k, eid]*= (1+np.random.normal(0.0, 0.05))  # Randomly add a small positive value
            #             else:
            #                 # Subtract a positive value to decrease the curve
            #                 offspring[k, eid] *= (1-np.random.normal(0.0, 0.05))  # Randomly subtract a small positive value
            #     eid+=1
            # Ensure bounds are respected
            # Add small random variations (local search)
            mutation_mask = np.random.random(offspring_size[1]) < 0.1  # 10% mutation rate
            mutation = np.random.normal(0, 0.01, offspring_size[1])  # Small variations
            offspring[k, mutation_mask] += mutation[mutation_mask]
            for j, gene_range in enumerate(self.gene_space):
                offspring[k, j] = min(gene_range['high'], 
                                    max(gene_range['low'], 
                                        offspring[k, j]))
        
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
