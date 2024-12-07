import numpy as np
from scipy import optimize
from data_handler_nobkg import DataHandler
import matplotlib.pyplot as plt
import pandas as pd
import os
import logging
from tqdm import tqdm

class ScipyOptimizer:
    def __init__(self, data_handler, fits_path, x_range=None, method='leastsq'):
        """
        Initialize optimizer using scipy.optimize methods.
        
        Args:
            data_handler (DataHandler): Instance of DataHandler class
            fits_path (str): Path to the FITS file to fit
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
            method (str): Optimization method ('leastsq', 'levenberg', 'trust-ncg', etc.)
        """
        self.data_handler = data_handler
        self.fits_path = fits_path
        self.x_range = x_range
        self.method = method
        self.energies, self.counts = self.data_handler.get_fits_data(fits_path)
        self.background = self.data_handler.get_background('/home/ubuntu/ch2-abundance/xrf_fit/ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits', self.energies)
        
        # scale background such that the total counts in both self.counts and self. background match
        scale_factor = np.sum(self.counts) / np.sum(self.background)
        self.background = self.background * scale_factor
        self.counts = self.counts - self.background
        
        # replace all negative values in counts with 0
        self.counts = np.maximum(self.counts, 0)
        
        # Number of parameters to optimize
        self.n_elements = len(self.data_handler.elements)
        # For each element: one amplitude and one sigma, plus global scale
        self.n_params = self.n_elements * 2 + 1
        
        # Initialize bounds and initial guess
        self._setup_optimization_params()

    def _setup_optimization_params(self):
        """Setup optimization parameters, bounds, and initial guess."""
        # Initialize bounds with non-negative amplitudes
        amp_bounds = [(0, self.data_handler.bounds[elem][1]) 
                     for elem in self.data_handler.elements]
        sigma_bounds = [(0, 0.1)] * self.n_elements
        scale_bounds = [(0, 10000)]
        
        self.bounds = amp_bounds + sigma_bounds + scale_bounds
        
        # Initial guess: middle of bounds for amplitudes, small value for sigmas
        self.initial_guess = []
        for (low, high) in amp_bounds:
            self.initial_guess.append((low + high) / 2)
        self.initial_guess.extend([0.05] * self.n_elements)  # Initial sigmas
        self.initial_guess.append(1.0)  # Initial scale

    def calculate_residuals(self, params):
        """Calculate residuals for optimization only around elemental peaks."""
        model = self.calculate_model_intensity(params)
        
        # Initialize mask for relevant energy ranges
        mask = np.zeros_like(self.energies, dtype=bool)
        
        # Create mask for energy ranges around each peak
        for element in self.data_handler.elements:
            for energy in self.data_handler.gaussian_models[element].means.values():
                # Find indices within ±0.5 keV of each peak
                peak_mask = (self.energies >= energy - 0.5) & (self.energies <= energy + 0.5)
                mask = mask | peak_mask  # Combine masks using OR operation
        
        # Calculate residuals only for masked regions
        residuals = np.zeros_like(self.counts)
        residuals[mask] = self.counts[mask] - model[mask]
        
        return residuals

    def calculate_model_intensity(self, params):
        """Calculate model intensity for given parameters."""
        amplitudes = np.maximum(params[:self.n_elements], 0)  # Ensure non-negative amplitudes
        sigmas = params[self.n_elements:-1]
        scale = params[-1]
        
        # Update data handler parameters
        for i, element in enumerate(self.data_handler.elements):
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
        
        model_intensity = np.zeros_like(self.energies)
        for element, gaussian in self.data_handler.gaussian_models.items():
            amp = self.data_handler.element_amplitudes[element]
            model_intensity += gaussian(self.energies) * amp
        
        model_intensity *= scale
        return model_intensity

    def calculate_jacobian(self, params):
        """Calculate the Jacobian (gradient) for optimization.""" 
        model = self.calculate_model_intensity(params)
        jacobian = np.zeros((len(self.counts), self.n_params))
        
        # Create mask for relevant energy ranges
        mask = np.zeros_like(self.energies, dtype=bool)
        for element in self.data_handler.elements:
            for energy in self.data_handler.gaussian_models[element].means.values():
                peak_mask = (self.energies >= energy - 0.5) & (self.energies <= energy + 0.5)
                mask = mask | peak_mask
        
        # Calculate the Jacobian for each parameter only in masked regions
        for i in range(self.n_elements):
            # Partial derivative with respect to amplitude
            deriv = -self.data_handler.gaussian_models[self.data_handler.elements[i]](self.energies)
            jacobian[:, i][mask] = deriv[mask]
        
        for i in range(self.n_elements):
            # Partial derivative with respect to sigma
            deriv = -self.data_handler.element_amplitudes[self.data_handler.elements[i]] * \
                self.data_handler.gaussian_models[self.data_handler.elements[i]].derivative(self.energies)
            jacobian[:, self.n_elements + i][mask] = deriv[mask]
        
        # Partial derivative with respect to scale
        jacobian[:, -1][mask] = model[mask]
        
        return jacobian

    def calculate_hessian_vector_product(self, params, vector):
        """Calculate the Hessian-vector product for optimization."""
        hessian_vector_product = np.zeros(self.n_params)
        
        # Calculate the model intensity for the current parameters
        model = self.calculate_model_intensity(params)
        
        # Calculate the residuals
        residuals = self.calculate_residuals(params)
        
        # Calculate the Hessian-vector product for each parameter
        for i in range(self.n_elements):
            # Hessian-vector product with respect to amplitude
            hessian_vector_product[i] = -self.data_handler.gaussian_models[self.data_handler.elements[i]](self.energies) @ vector
        
        for i in range(self.n_elements):
            # Hessian-vector product with respect to sigma
            hessian_vector_product[self.n_elements + i] = -self.data_handler.element_amplitudes[self.data_handler.elements[i]] * \
                self.data_handler.gaussian_models[self.data_handler.elements[i]].derivative(self.energies) @ vector
        
        # Hessian-vector product with respect to scale
        hessian_vector_product[-1] = model @ vector
        
        return hessian_vector_product

    def run_optimization(self):
        """Run optimization with specified method."""
        if self.method == 'leastsq':
            result = optimize.least_squares(
                self.calculate_residuals,
                self.initial_guess,
                bounds=tuple(zip(*self.bounds)),
                method='trf',
                loss='linear',
                verbose=0
            )
        elif self.method == 'levenberg':
            result = optimize.root(
                self.calculate_residuals,
                self.initial_guess,
                method='lm',
                options={'maxiter': 500, 'disp': False}
            )
        else:
            result = optimize.minimize(
                lambda x: np.sum(self.calculate_residuals(x)**2),
                self.initial_guess,
                method=self.method,
                bounds=self.bounds,
                jac=self.calculate_jacobian,
                hessp=self.calculate_hessian_vector_product,
                options={'maxiter': 500, 'disp': False}
            )

        self.result = result
        return result.x, result.fun if hasattr(result, 'fun') else None

    def plot_result(self, model_name="", save_dir="plots"):
        """Plot the optimized fit result with individual element contributions."""
        # Create plots directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot the FITS data
        # energies, counts = self.data_handler.get_fits_data(self.fits_path)
        ax.plot(self.energies, self.counts, 'k-', label='Data', alpha=0.5)
        
        # Calculate and plot the total model intensity
        model_intensity = self.calculate_model_intensity(self.result.x)
        ax.plot(self.energies, model_intensity, 'r--', label='Total Fit', alpha=0.7)
        
        # Plot individual element contributions
        amplitudes = np.maximum(self.result.x[:self.n_elements], 0)  # Ensure non-negative amplitudes
        sigmas = self.result.x[self.n_elements:-1]
        scale = self.result.x[-1]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.data_handler.elements)))
        for i, (element, color) in enumerate(zip(self.data_handler.elements, colors)):
            # Get the individual element's contribution
            self.data_handler.set_amplitude(element, amplitudes[i])
            self.data_handler.gaussian_models[element].std_devs[:] = sigmas[i]
            element_intensity = (self.data_handler.gaussian_models[element](self.energies) * 
                               amplitudes[i] * scale)
            
            # Plot element contribution
            ax.plot(self.energies, element_intensity, '--', 
                    color=color, 
                    label=f'{element} (amp={amplitudes[i]:.2f})', 
                    alpha=0.5)
            
            # Add peak annotations
            for label, energy in self.data_handler.gaussian_models[element].means.items():
                if self.x_range is None or (self.x_range[0] <= energy <= self.x_range[1]):
                    height = element_intensity[np.abs(self.energies - energy).argmin()]
                    ax.annotate(f'{element}-{label}',
                            xy=(energy, height),
                            xytext=(0, 10), 
                            textcoords='offset points',
                            ha='center', 
                            va='bottom',
                            fontsize=8, 
                            color=color,
                            rotation=45)
        
        # Set plot limits if specified
        if self.x_range is not None:
            ax.set_xlim(self.x_range)
        
        ax.set_xlabel('Energy (KeV)')
        ax.set_ylabel('Intensity')
        ax.set_title(f'Optimized Fit Result ({model_name})\nScale: {scale:.2f}')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True)
        
        # Adjust layout to prevent label cutoff
        plt.tight_layout()
        
        # Save plot with full path
        # plot_path = os.path.join(save_dir, os.path.basename(self.fits_path).replace('.fits', f'_{model_name}.png'))
        # fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        # logging.info(f"Plot saved to: {plot_path}")
        
        return fig, ax

    def fit_from_files(self, fits_path, x_range=None, method='leastsq'):
        """
        Fit the data from the provided FITS file.
        
        Args:
            fits_path (str): Path to the FITS file to fit.
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit (default is None).
            method (str): Optimization method ('leastsq', 'levenberg', etc.).
        
        Returns:
            tuple: (amplitudes, sigmas) - Optimized element amplitudes and sigmas.
        """
        # Initialize optimizer with the provided paths
        self.fits_path = fits_path
        self.x_range = x_range
        self.method = method
        
        # Run optimization
        solution, _ = self.run_optimization()
        
        # Extract amplitudes and sigmas
        amplitudes = np.maximum(solution[:self.n_elements], 0)  # Ensure non-negative amplitudes
        sigmas = solution[self.n_elements:-1]  # Assuming the last parameter is scale
        
        return amplitudes, sigmas

    def calculate_total_error(self, params):
        """Calculate total least square error for all peaks."""
        model = self.calculate_model_intensity(params)
        
        # Initialize mask for all peak regions
        mask = np.zeros_like(self.energies, dtype=bool)
        
        # Create mask for energy ranges around each peak
        for element in self.data_handler.elements:
            for energy in self.data_handler.gaussian_models[element].means.values():
                peak_mask = (self.energies >= energy - 0.5) & (self.energies <= energy + 0.5)
                mask = mask | peak_mask
        
        # Calculate error only for masked regions
        residuals = self.counts[mask] - model[mask]
        mse = np.mean(residuals ** 2)
        return mse

def test_optimization(fits_path, method='leastsq', x_range=(0, 27)):
    """Test function to demonstrate the optimization."""
    handler = DataHandler()
    optimizer = ScipyOptimizer(handler, fits_path, x_range, method)
    
    solution, fitness = optimizer.run_optimization()
    
    # logging.info("\nOptimization Results:")
    # logging.info(f"Method: {method}")
    # logging.info(f"Final objective value: {fitness}")
    # logging.info("\nOptimized Parameters:")
    # for i, element in enumerate(handler.elements):
    #     logging.info(f"{element}:")
    #     logging.info(f"  Amplitude: {max(solution[i], 0):.3f}")  # Ensure non-negative amplitude in output
    #     logging.info(f"  Sigma: {solution[i + len(handler.elements)]:.3f}")
    # logging.info(f"Scale: {solution[-1]:.3f}")
    
    # Create plots with proper saving
    fig, ax = optimizer.plot_result(method)
    plt.close(fig)  # Close figure to free memory
    
    return optimizer

if __name__ == "__main__":
    # Suppress scipy optimization messages
    # logging.getLogger('scipy').setLevel(logging.ERROR)
    import multiprocessing as mp
    from functools import partial
    import os
    from tqdm import tqdm

    def process_file(fits_file):
        """Process a single fits file and return results dictionary"""
        try:
            results = {}
            
            optimizer = test_optimization(fits_file, method='leastsq', x_range=(0, 7))
            solution, fitness = optimizer.run_optimization()
            
            # Calculate total error
            total_error = optimizer.calculate_total_error(solution)
            
            # Store results
            results['filename'] = fits_file
            results['total_error'] = total_error
            
            # Process each element
            for idx, element in enumerate(['Fe', 'Al', 'Mg', 'Si', 'Ca', 'Ti', 'O']):
                results[f'{element}_amplitude'] = max(solution[idx], 0)
                results[f'{element}_sigma'] = solution[idx + 7]
                
            results['scale'] = solution[-1]
            return results
        except Exception as e:
            print(f"Error processing {fits_file}: {str(e)}")
            return None

    # Read the full list of files
    ca_al_list = pd.read_csv('/home/ubuntu/ch2-abundance/utils/ca-al-detection/high_confidence_ca_al.csv')
    all_files = ca_al_list['filename'].tolist()
    
    output_path = '/home/ubuntu/ch2-abundance/xrf_fit/gaussian-fit/optimization_results.csv'
    
    # Initialize progress bar for total files
    pbar = tqdm(total=len(all_files), desc="Processing files")
    
    def update_progress(*a):
        """Callback function to update progress bar"""
        pbar.update()

    # Process files using multiprocessing
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Map process_file to all files with progress tracking
        results = []
        for i, result in enumerate(pool.imap_unordered(process_file, all_files)):
            if result is not None:
                results.append(result)
                # Save intermediate results periodically (every 100 files)
                if len(results) % 3000 == 0:
                    pd.DataFrame(results).to_csv(f"{output_path}_{i}", index=False)
            update_progress()
    
    # Save final results
    final_df = pd.DataFrame(results)
    final_df.to_csv(output_path, index=False)
    
    pbar.close()

    # # Plot first 10 files from the high confidence list
    # ca_al_list = pd.read_csv('/home/ubuntu/ch2-abundance/utils/ca-al-detection/high_confidence_ca_al.csv')
    
    # for i in tqdm(range(10), desc="Plotting files", position=0, leave=True):
    #     fits_file = ca_al_list['filename'][i]
        
    #     # Create optimizer instance
    #     optimizer = ScipyOptimizer(DataHandler(), fits_file, x_range=(0, 7))
        
    #     # Run optimization
    #     solution, fitness = optimizer.run_optimization()
        
    #     # Create figure for plotting
    #     fig, ax = plt.subplots(figsize=(12, 6))
        
    #     # Plot original data
    #     ax.plot(optimizer.energies, optimizer.counts, 'k-', label='Data', alpha=0.7)
        
    #     # Plot fitted model
    #     model = optimizer.calculate_model_intensity(solution)
    #     ax.plot(optimizer.energies, model, 'r--', label='Fitted Model', alpha=0.7)
        
    #     # Customize plot
    #     ax.set_xlabel('Energy (KeV)')
    #     ax.set_ylabel('Counts')
    #     ax.set_title(f'XRF Fit - {os.path.basename(fits_file)}')
    #     ax.legend()
    #     ax.grid(True)
        
    #     # Save plot
    #     plot_dir = 'fit_plots'
    #     os.makedirs(plot_dir, exist_ok=True)
    #     plt.savefig(os.path.join(plot_dir, f'fit_{i:03d}.png'), dpi=300, bbox_inches='tight')
    #     plt.close()
