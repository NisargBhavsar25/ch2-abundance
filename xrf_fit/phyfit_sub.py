import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt
from element_model_sub import ElementModel  # Import ElementModel
from element_handler_sub import ElementHandler
from astropy.io import fits  # Add this import at the top
import pygad  # Add this import
import os
from data_handler_nobkg import DataHandler

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
            self.kev_per_channel = 0.0135 if num_channels == 2048 else 0.0277
            
        self.energies, self.counts = self.get_fits_data(fits_path)
        self.background = self.get_background('/home/ubuntu/ch2-abundance/xrf_fit/ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits', self.energies)
        
        # scale_factor = np.sum(self.counts) / np.sum(self.background)
        self.background = self.background 
        self.counts = self.counts - self.background
        
        self.counts = np.maximum(self.counts, 0)
        
        self.n_elements = len(self.el_handler.elements)
        # For each element: one amplitude and one sigma, plus global scale
        self.n_params = self.n_elements * 2 + 1
        
        # Initialize bounds and initial guess
        self._setup_optimization_params()
        
        # Add GA-specific initialization
        if method == 'ga':
            self.init_ga()
    
    def get_background(self, background_fits_path, energies):
        """
        Add background noise from a reference FITS file to a spectrum.
        
        Args:
            background_fits_path (str): Path to the background FITS file
            energies (np.array): Energy values for the spectrum
            spectrum (np.array): Intensity values of the original spectrum
            
        Returns:
            np.array: New spectrum with background added
        """
        # Get background data
        bg_energies, bg_counts = self.get_fits_data(background_fits_path)
        
        # Ensure the background data matches the spectrum energy points
        # by interpolating the background counts
        bg_interpolated = np.interp(energies, bg_energies, bg_counts)
        
        return bg_interpolated
    
    def _setup_optimization_params(self):
        """Setup optimization parameters, bounds, and initial guess."""
        # Initialize bounds for concentrations
        conc_bounds = [(0.0, 100.0)] * self.n_elements
        
        # Initialize bounds for standard deviations
        std_dev_bounds = [(0.01, 1.0)] * self.n_elements
        
        # Count total number of lines across all elements
        n_beta_params = sum(len(self.el_handler.element_models[element].lines) 
                          for element in self.el_handler.elements)
        beta_bounds = [(0.0, 10.0)] * n_beta_params
        
        # Scale factor bounds
        scale_bounds = [(1e-7, 10)]
        
        self.bounds = conc_bounds + std_dev_bounds + beta_bounds + scale_bounds
        
        # Initial guess
        self.initial_guess = []
        # Initial concentrations with 20% random variation
        for element in self.el_handler.elements:
            base_conc = self.el_handler.conc[element]
            self.initial_guess.append(base_conc * (1 + 0.5 * (np.random.random() - 0.5)))
        
        # Initial standard deviations with variation
        self.initial_guess.extend([0.5 * (1 + 0.5 * (np.random.random() - 0.5)) 
                                 for _ in range(self.n_elements)])
        
        # Initial betas with variation
        self.initial_guess.extend([2.0 * (1 + 0.5 * (np.random.random() - 0.5)) 
                                 for _ in range(n_beta_params)])
        
        # Initial scale
        self.initial_guess.append(1e-5 * (1 + 0.5 * (np.random.random() - 0.5)))

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
        # Beta ranges
        for _ in range(n_beta_params):
            self.gene_space.append({'low': 0.0, 'high': 2.0})
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
        # print(f"Generation {ga_instance.generations_completed}")
        # print(f"Best fitness: {ga_instance.best_solution()[1]}")
        
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
            # Update concentrations and standard deviations
            for i, element in enumerate(self.el_handler.elements):
                self.el_handler.conc[element] = max(0.0, min(50.0, params[i]))
                self.el_handler.std_dev[element] = max(0.05, min(1.0, params[self.n_elements + i]))
            
            # Update beta parameters
            beta_start_idx = 2 * self.n_elements
            beta_idx = 0
            for element in self.el_handler.elements:
                for line in self.el_handler.element_models[element].lines:
                    beta_value = max(0.0, min(2.0, params[beta_start_idx + beta_idx]))
                    self.el_handler.set_beta(element, line, beta_value)
                    beta_idx += 1
            
            # Calculate model intensity
            model_intensity = self.el_handler.calculate_folded_intensity(self.energies)
            model_intensity *= params[-1]  # Scale factor
            
            # Calculate residuals with extra weight on peaks
            residuals = np.zeros_like(self.counts)
            peak_weights = np.ones_like(self.counts)
            
            for element in self.el_handler.elements:
                for line_group in self.el_handler.element_models[element].line_div.values():
                    for line in line_group:
                        energy_mean = self.el_handler.element_models[element].energy_dict[line[:2]]["mean"]
                        std_dev = self.el_handler.std_dev[element]
                        
                        # Calculate bin range around peak
                        binstart = int((energy_mean - 3*std_dev) / self.kev_per_channel)
                        binend = int((energy_mean + 3*std_dev) / self.kev_per_channel)
                        
                        # Ensure bins are within valid range
                        binstart = max(0, binstart)
                        binend = min(len(self.counts), binend)
                        
                        # Add higher weights around peaks using Gaussian profile
                        peak_center = int(energy_mean / self.kev_per_channel)
                        x = np.arange(binstart, binend)
                        peak_weights[binstart:binend] += 5.0 * np.exp(-0.5 * ((x - peak_center)/(std_dev/self.kev_per_channel))**2)
                        
                        # Calculate weighted residuals for this range
                        residuals[binstart:binend] = np.abs(self.counts[binstart:binend] - model_intensity[binstart:binend])
            
            # Apply peak weights to residuals
            weighted_residuals = residuals * peak_weights
            
            # Add penalty for peak height mismatches
            peak_penalty = 10.0 * np.sum((np.maximum(self.counts - model_intensity, 0))**2)
            
            error = np.sum(weighted_residuals**2) + peak_penalty
            
            # print("Error:", error)
            # print(np.max(self.counts), np.max(model_intensity))
            
            if hasattr(self, '_iter_count'):
                self._iter_count += 1
            else:
                self._iter_count = 0
            
            # if self._iter_count % 100 == 0:
            #     print(f"Iteration {self._iter_count}, Error: {error:.3f}")
            
            return error
        except Exception as e:
            # print(f"Error in objective function: {e}")
            return 1e10

    def calculate_model_intensity(self, params):
        """Calculate model intensity for given parameters."""
        # Update concentrations, std_devs, and offsets in element handler
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
        return np.abs(self.counts - model)

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
                # verbose=2
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
        """Plot the optimized fit result with individual element contributions."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data
        ax.plot(self.energies, self.counts, 'k-', label='Data', linewidth=2)
        
        # Calculate and plot total model
        model_intensity = self.calculate_model_intensity(self.result.x)
        ax.plot(self.energies, model_intensity, 'r-', label='Total Fit', linewidth=2)
        
        # Plot individual element contributions
        scale = np.clip(self.result.x[-1], 1e-7, 1e-4)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(self.el_handler.elements)))
        
        for i, element in enumerate(self.el_handler.elements):
            # Get amplitude and sigma for this element
            amplitude = self.result.x[i]
            sigma = self.result.x[self.n_elements + i]
            
            # Temporarily set all concentrations to 0 except current element
            original_conc = self.el_handler.conc.copy()
            self.el_handler.conc = {el: 0.0 for el in self.el_handler.elements}
            self.el_handler.conc[element] = original_conc[element]
            
            # Calculate individual element intensity
            element_intensity = self.el_handler.calculate_folded_intensity(self.energies) * scale
            ax.plot(self.energies, element_intensity, '-', color=colors[i], 
                   label=f'{element} (A={amplitude:.2f}, σ={sigma:.2f})', alpha=0.6)
            
            # Restore original concentrations
            self.el_handler.conc = original_conc
        
        ax.set_xlabel('Energy (keV)')
        ax.set_ylabel('Counts')
        ax.set_title(f'Fit Result: {model_name}')
        # ax.set_yscale('log')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, which='both', linestyle='--', alpha=0.3)
        
        # Extract just the filename without the path
        base_filename = os.path.basename(model_name)
        
        # Create a simpler output filename
        output_filename = f'phyFit_result_{base_filename}.png'
        
        plt.tight_layout()
        print(f'Saving plot as: {output_filename}')
        plt.savefig(output_filename, bbox_inches='tight')
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
import multiprocessing as mp
from functools import partial
from tqdm import tqdm

if __name__ == "__main__":

    # Initialize element handler
    el_handler = ElementHandler(num_channels=2048, verbose=False)
    
    # Read list of files to process
    ca_al_list = pd.read_csv("high_confidence_ca_al.csv")
    
    # Create phyfit-csv directory if it doesn't exist
    os.makedirs("phyfit-csv", exist_ok=True)
    
    def process_file(el_handler, file_info):
        """Process a single file and return results"""
        fits_path = file_info["filename"]
        try:
            # Initialize optimizer
            optimizer = PhyOptimizer(el_handler, fits_path, method="leastsq")
            
            # Run optimization
            solution, residual = optimizer.run_optimization()
            
            # Extract amplitudes and sigmas for each element
            n_elements = len(el_handler.elements)
            amplitudes = solution[:n_elements]
            sigmas = solution[n_elements:-1]
            
            # Create dictionary with filename and parameters for each element
            result = {'filename': fits_path}
            
            # Add amplitude and sigma columns for each element
            for i, element in enumerate(el_handler.elements):
                result[f'{element}_amplitude'] = amplitudes[i]
                result[f'{element}_sigma'] = sigmas[i]
                
            return result
            
        except Exception as e:
            print(f"Error processing {fits_path}: {str(e)}")
            return None

    # Set up multiprocessing
    num_processes = mp.cpu_count() - 1  # Leave one CPU free
    pool = mp.Pool(processes=num_processes)
    
    # Create partial function with fixed el_handler
    process_func = partial(process_file, el_handler)
    
    # Process files in parallel with progress bar
    results = []
    batch_size = 3000
    
    # Create iterator of file info dictionaries
    file_infos = [{"filename": row["filename"]} for _, row in ca_al_list.iterrows()]
    
    # Calculate total number of batches
    total_batches = (len(file_infos) + batch_size - 1) // batch_size
    print(f"Processing {total_batches} total batches")
    
    # Process files in batches
    for batch_num, i in tqdm(enumerate(range(0, len(file_infos), batch_size)), desc="Processing batches"):
        batch = file_infos[i:i + batch_size]
        
        batch_results = []
        for result in tqdm(pool.imap_unordered(process_func, batch),
                         total=len(batch),
                         desc=f"Processing batch {batch_num + 1}/{total_batches}",
                         leave=False):
            if result is not None:
                batch_results.append(result)
        
        # Convert batch results to DataFrame and save
        if batch_results:
            batch_df = pd.DataFrame(batch_results)
            batch_df.to_csv(f"phyfit-csv/batch_{batch_num + 1}.csv", index=False)
            results.extend(batch_results)
    
    # Combine all results and save final CSV
    if results:
        final_df = pd.DataFrame(results)
        final_df.to_csv("phyfit-csv/combined_results.csv", index=False)
        print("Final results saved to phyfit-csv/combined_results.csv")
    
    pool.close()
    pool.join()
