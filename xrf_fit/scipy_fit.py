import numpy as np
from scipy import optimize
from data_handler import DataHandler
import matplotlib.pyplot as plt

class ScipyOptimizer:
    def __init__(self, data_handler, fits_path, bkg_path, x_range=None, method='leastsq'):
        """
        Initialize optimizer using scipy.optimize methods.
        
        Args:
            data_handler (DataHandler): Instance of DataHandler class
            fits_path (str): Path to the FITS file to fit
            bkg_path (str): Path to background file
            x_range (tuple, optional): Energy range (min_kev, max_kev) to fit
            method (str): Optimization method ('leastsq', 'levenberg', 'trust-ncg', etc.)
        """
        self.data_handler = data_handler
        self.fits_path = fits_path
        self.bkg_path = bkg_path
        self.x_range = x_range
        self.method = method
        self.energies, self.counts = self.data_handler.get_fits_data(fits_path)
        
        # Number of parameters to optimize
        self.n_elements = len(self.data_handler.elements)
        # For each element: one amplitude and one sigma, plus global scale
        self.n_params = self.n_elements * 2 + 1
        
        # Initialize bounds and initial guess
        self._setup_optimization_params()

    def _setup_optimization_params(self):
        """Setup optimization parameters, bounds, and initial guess."""
        # Initialize bounds
        amp_bounds = [(self.data_handler.bounds[elem][0], self.data_handler.bounds[elem][1]) 
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
        """Calculate residuals for optimization."""
        model = self.calculate_model_intensity(params)
        return self.counts - model

    def calculate_model_intensity(self, params):
        """Calculate model intensity for given parameters."""
        amplitudes = params[:self.n_elements]
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
        self.data_handler.add_background(self.bkg_path, self.energies, model_intensity)
        return model_intensity

    def calculate_jacobian(self, params):
        """Calculate the Jacobian (gradient) for optimization.""" 
        model = self.calculate_model_intensity(params)
        jacobian = np.zeros((len(self.counts), self.n_params))
        
        # Calculate the Jacobian for each parameter
        for i in range(self.n_elements):
            # Partial derivative with respect to amplitude
            jacobian[:, i] = -self.data_handler.gaussian_models[self.data_handler.elements[i]](self.energies)
        
        for i in range(self.n_elements):
            # Partial derivative with respect to sigma
            jacobian[:, self.n_elements + i] = -self.data_handler.element_amplitudes[self.data_handler.elements[i]] * \
                self.data_handler.gaussian_models[self.data_handler.elements[i]].derivative(self.energies)
        
        # Partial derivative with respect to scale
        jacobian[:, -1] = model
        
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
                verbose=2
            )
        elif self.method == 'levenberg':
            result = optimize.root(
                self.calculate_residuals,
                self.initial_guess,
                method='lm',
                options={'maxiter': 500}
            )
        else:
            result = optimize.minimize(
                lambda x: np.sum(self.calculate_residuals(x)**2),
                self.initial_guess,
                method=self.method,
                bounds=self.bounds,
                jac=self.calculate_jacobian,  # Add Jacobian here
                hessp=self.calculate_hessian_vector_product,  # Add Hessian-vector product here

                options={'maxiter': 500}
            )

        self.result = result
        return result.x, result.fun if hasattr(result, 'fun') else None

    def plot_result(self,model_name=""):
        """Plot the optimized fit result."""
        fig, (ax1, ax2) = self.data_handler.plot_combined_spectrum(
            self.fits_path,
            x_range=self.x_range,
            name=f"scipy_solver_{model_name}",
            normalize=False
        )

        return fig, (ax1, ax2)

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

def test_optimization(fits_path, bkg_path, method='leastsq', x_range=(0, 27)):
    """Test function to demonstrate the optimization."""
    handler = DataHandler(bkg_path=bkg_path)
    optimizer = ScipyOptimizer(handler, fits_path, bkg_path, x_range, method)
    
    solution, fitness = optimizer.run_optimization()
    
    print("\nOptimization Results:")
    print(f"Method: {method}")
    print(f"Final objective value: {fitness}")
    print("\nOptimized Parameters:")
    for i, element in enumerate(handler.elements):
        print(f"{element}:")
        print(f"  Amplitude: {solution[i]:.3f}")
        print(f"  Sigma: {solution[i + len(handler.elements)]:.3f}")
    print(f"Scale: {solution[-1]:.3f}")
    
    fig, (ax1, ax2) = optimizer.plot_result(method)
    
    plt.show()
    plt.savefig(f"Fit_{method}.png")
    return optimizer

if __name__ == "__main__":
    fits_file = "fits_20/ch2_cla_l1_20240123T033030609_20240123T033038609.fits"
    # bkg_file = "data/ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits"
    
    # Test different optimization methods
    methods = ['leastsq', 'levenberg']
    for method in methods:
        print(f"\nTesting {method} method...")
        test_optimization(fits_file, bkg_file, method=method) 