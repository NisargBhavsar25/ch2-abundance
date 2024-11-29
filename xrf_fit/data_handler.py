from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from gauss import GaussianSum
import os

class DataHandler:
    def __init__(self, std_dev=0.1):
        """
        Initialize DataHandler with a list of elements and their amplitudes.
        
        Args:
            std_dev (float): Standard deviation for Gaussian peaks
        """
        self.elements = ["Fe", "Al", "Mg", "Si", "Ca", "Ti"]
        # Initial amplitudes - can be adjusted based on expected abundances
        self.element_amplitudes = {
            "Fe": 1.0,
            "Al": 0.8,
            "Mg": 0.6,
            "Si": 1,
            "Ca": 0.7,
            "Ti": 0.5
        }
        self.bounds={
            "Fe":[0,1],
            "Al":[0,1],
            "Mg":[0,1],
            "Si":[0,1],
            "Ca":[0,1],
            "Ti":[0,1],
                     }
        self.std_dev = std_dev
        self.gaussian_models = {}
        
        # Create GaussianSum objects for each element
        for element in self.elements:
            try:
                self.gaussian_models[element] = GaussianSum(element, std_dev)
            except Exception as e:
                print(f"Warning: Could not create model for {element}: {str(e)}")
        
        # Create plots directory if it doesn't exist
        self.plot_dir = os.path.join(os.path.dirname(__file__), 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
    
    def set_amplitude(self, element, amplitude):
        """
        Set the amplitude for a specific element.
        
        Args:
            element (str): Element symbol
            amplitude (float): New amplitude value
        """
        if element in self.element_amplitudes:
            self.element_amplitudes[element] = amplitude
        else:
            raise ValueError(f"Element {element} not in model list")

    def display_fits(self, fits_path, x_range=None):
        """
        Display FITS file data with Gaussian models overlay.
        
        Args:
            fits_path (str): Path to the FITS file
            x_range (tuple, optional): (min_energy, max_energy) in KeV to plot
        """
        # Read FITS file
        try:
            with fits.open(fits_path) as hdul:
                data = hdul[1].data
                header = hdul[1].header
                
                # Extract energy calibration from header if available
                # This is a placeholder - adjust according to your FITS file structure
                num_channels = len(data)
                energy_start = 0
                energy_step = 0.0277
                print(energy_start,energy_step,num_channels)
                energies = np.linspace(energy_start, 
                                     energy_start + energy_step * num_channels,
                                     num_channels)
                
        except Exception as e:
            raise ValueError(f"Error reading FITS file: {str(e)}")

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot FITS data
        ax.plot(data, 'k-', label='Data', alpha=0.5)
        
        # Plot Gaussian models for each element
        for element, model in self.gaussian_models.items():
            y = model(energies)
            ax.plot(energies, y, '--', label=f'{element} lines', alpha=0.7)
        
        # Set plot limits if specified
        if x_range is not None:
            ax.set_xlim(x_range)
        
        ax.set_xlabel('Energy (KeV)')
        ax.set_ylabel('Intensity')
        ax.set_title('XRF Spectrum with Emission Line Models')
        ax.legend()
        ax.grid(True)
        
        plt.show()
        
        return fig, ax
    def get_fits_data(self,fits_path):
         with fits.open(fits_path) as hdul:
                data = hdul[1].data  # Convert to flat numpy array
                header = hdul[1].header
                 # Handle structured array data
                # if isinstance(data, np.recarray) or data.dtype.names is not None:
                    # Extract channels and counts from structured array
                channels = data['CHANNEL']
                counts = data['COUNTS']
                # else:
                #     # If it's a simple array, use it directly
                #     counts = np.array(data).flatten()
                #     channels = np.arange(len(counts))
                
                # Create energy axis
                num_channels = len(data)
                energy_start = 0
                energy_step = 0.0277
                print(energy_start,energy_step,num_channels)
                energies = np.linspace(energy_start, 
                                     energy_start + energy_step * num_channels,
                                     num_channels)
                return energies,counts
    def plot_fits_data(self, fits_path, x_range=None):
        """
        Plot the FITS file data as a line curve.
        
        Args:
            fits_path (str): Path to the FITS file
            x_range (tuple, optional): (min_energy, max_energy) in KeV to plot
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        # try:
        with fits.open(fits_path) as hdul:
                data = hdul[1].data  # Convert to flat numpy array
                header = hdul[1].header
                 # Handle structured array data
                # if isinstance(data, np.recarray) or data.dtype.names is not None:
                    # Extract channels and counts from structured array
                channels = data['CHANNEL']
                counts = data['COUNTS']
                # else:
                #     # If it's a simple array, use it directly
                #     counts = np.array(data).flatten()
                #     channels = np.arange(len(counts))
                
                # Create energy axis
                num_channels = len(data)
                energy_start = 0
                energy_step = 0.0277
                print(energy_start,energy_step,num_channels)
                energies = np.linspace(energy_start, 
                                     energy_start + energy_step * num_channels,
                                     num_channels)
                
                # Create plot
                fig, ax = plt.subplots(figsize=(12, 6))
                # plt.figure(figsize=(10,6))
                # plt.plot(counts)
                # plt.title("Line Plot of FITS Image Data")
                # plt.xlabel("Pixel Index")
                # plt.ylabel("Intensity (COUNTS)")
                # plt.grid(True)  # Add grid for clarity
                # plt.show()
                # Plot the data
                ax.plot(energies,counts, '-', color='black', 
                       label='XRF Data', linewidth=1.0)
                # Set plot limits if specified
                # if x_range is not None:
                #     ax.set_xlim(x_range)
                
                # Customize plot
                # ax.set_xticks(np.arange(x_range[0],x_range[1]))  # Use numpy's arange instead
                # ax.set_xticklabels([str(x) for x in range(x_range[0],x_range[1])])  # Convert labels to strings
                
                ax.set_xlabel('Energy (KeV)')
                ax.set_ylabel('Intensity (counts)')
                ax.set_title('XRF Spectrum')
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                # Save plot
                fits_name = os.path.splitext(os.path.basename(fits_path))[0]
                range_str = f'_{x_range[0]}-{x_range[1]}keV' if x_range else ''
                self.save_plot(fig, f'fits_spectrum_{fits_name}{range_str}.png')
                
                return fig, ax, energies, data
                
        # except Exception as e:
        #     raise ValueError(f"Error reading or plotting FITS file: {str(e)}")

    def plot_combined_spectrum(self, fits_path, x_range=None, normalize=True):
        """
        Modified version of combined spectrum plot using the new FITS plotting function.
        """
        # First get the FITS data
        fig, ax1, energies, data = self.plot_fits_data(fits_path, x_range)
        ax2 = ax1.twinx()
        
        # Get max intensity for normalization if requested
        # max_intensity = np.max(data) if normalize else 1.0
        # plt.show()
        # Plot Gaussian models
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.elements)))
        for element, model, color in zip(self.elements, self.gaussian_models.values(), colors):
            amplitude = self.element_amplitudes[element]
            y = model(energies) * amplitude
            # if normalize:
            #     y = y * (max_intensity / np.max(y))
            ax2.plot(energies, y, '--', 
                    label=f'{element} (amp={amplitude:.2f})', 
                    color=color, alpha=0.7)
            
            # Add peak annotations
            for label, energy in model.means.items():
                if x_range is None or (x_range[0] <= energy <= x_range[1]):
                    height = model.single_gaussian(energy, energy, model.std_devs[0], 
                                                model.amplitudes[0]) * amplitude
                    # if normalize:
                    #     height = height * (max_intensity / np.max(y))
                    ax2.annotate(f'{element}-{label}',
                               xy=(energy, height),
                               xytext=(0, 10), textcoords='offset points',
                               ha='center', va='bottom',
                               fontsize=8, color=color,
                               rotation=45)
        
        # Customize plot
        ax1.set_xlabel('Energy (KeV)', fontsize=12)
        ax1.set_ylabel('XRF Intensity (counts)', fontsize=12, color='black')
        ax2.set_ylabel('Model Intensity (a.u.)', fontsize=12, color='gray')
        plt.title('XRF Spectrum with Emission Line Models', fontsize=14, pad=20)
        
        # Customize ticks
        ax1.tick_params(axis='y', labelcolor='black')
        ax2.tick_params(axis='y', labelcolor='gray')
        
        # Add legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, 
                  loc='upper right', bbox_to_anchor=(1.15, 1.0))
        
        # Save plot
        fits_name = os.path.splitext(os.path.basename(fits_path))[0]
        range_str = f'_{x_range[0]}-{x_range[1]}keV' if x_range else ''
        self.save_plot(fig, f'combined_spectrum_{fits_name}{range_str}.png')
        
        return fig, (ax1, ax2)

    def generate_spectrum(self, energy_range=(0, 10), num_bins=1000):
        """
        Generate a complete spectrum by summing all weighted Gaussian models.
        
        Args:
            energy_range (tuple): (min_energy, max_energy) in KeV
            num_bins (int): Number of bins for the energy axis
            
        Returns:
            tuple: (energies, intensities) arrays for the combined spectrum
        """
        # Create energy axis
        energies = np.linspace(energy_range[0], energy_range[1], num_bins)
        
        # Initialize intensity array
        total_intensity = np.zeros_like(energies)
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Sum weighted contributions from all elements
        for element, model in self.gaussian_models.items():
            amplitude = self.element_amplitudes[element]
            element_intensity = model(energies) * amplitude
            total_intensity += element_intensity
            
            # Plot individual weighted contributions
            ax2.plot(energies, element_intensity, '--', 
                    label=f'{element} (amp={amplitude:.2f})', alpha=0.7)
        
        # Plot combined spectrum
        ax1.plot(energies, total_intensity, 'k-', label='Combined Spectrum', linewidth=2)
        ax1.set_title('Combined Weighted Spectrum')
        ax1.set_xlabel('Energy (KeV)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.grid(True)
        ax1.legend()
        
        ax2.set_title('Individual Element Contributions (Weighted)')
        ax2.set_xlabel('Energy (KeV)')
        ax2.set_ylabel('Intensity (a.u.)')
        ax2.grid(True)
        ax2.legend()
        
        plt.tight_layout()
        
        # Save plot instead of showing
        self.save_plot(fig, f'spectrum_{energy_range[0]}-{energy_range[1]}keV.png')
        
        return energies, total_intensity

    def save_plot(self, fig, filename):
        """
        Helper method to save plots with proper naming.
        
        Args:
            fig: matplotlib figure object
            filename (str): Base name for the plot file
        """
        filepath = os.path.join(self.plot_dir, filename)
        fig.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close(fig)

def test_weighted_spectrum(handler, energy_range=(0, 27)):
    """
    Test function to demonstrate spectrum generation with weights.
    
    Args:
        handler (DataHandler): Instance of DataHandler class
        energy_range (tuple): Energy range (min_kev, max_kev) to analyze
    """
    # Set some custom amplitudes
    handler.set_amplitude("Fe", 2.0)
    handler.set_amplitude("Si", 1.5)
    
    # Generate spectrum with custom parameters
    energy_range=(0, 2048*0.0135)
    energies, intensities = handler.generate_spectrum(
        energy_range=energy_range,
        num_bins=1024
    )
    
    # Create and save summary plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(energies, intensities, 'k-', label='Combined Spectrum')
    ax.set_title('Weighted Spectrum Summary')
    ax.set_xlabel('Energy (KeV)')
    ax.set_ylabel('Intensity (a.u.)')
    ax.grid(True)
    
    # Add amplitude information to plot
    amp_text = "Element Amplitudes:\n"
    for element, amp in handler.element_amplitudes.items():
        amp_text += f"{element}: {amp:.2f}\n"
    ax.text(0.95, 0.95, amp_text,
            transform=ax.transAxes,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    handler.save_plot(fig, f'test_weighted_spectrum_{energy_range[0]}-{energy_range[1]}keV.png')
    
    # Print statistics
    print("\nSpectrum Statistics:")
    print(f"Energy range: {energies[0]:.2f} - {energies[-1]:.2f} KeV")
    print(f"Number of bins: {len(energies)}")
    print(f"Maximum intensity: {np.max(intensities):.2e}")
    print(f"Total intensity: {np.sum(intensities):.2e}")
    print("\nElement Amplitudes:")
    for element, amp in handler.element_amplitudes.items():
        print(f"{element}: {amp:.2f}")
    
    print(f"\nPlots saved to: {handler.plot_dir}")

def test_combined_plot(handler, fits_path, x_range=None):
    """
    Test function to demonstrate the combined plotting functionality.
    
    Args:
        handler (DataHandler): Instance of DataHandler class
        fits_path (str): Path to FITS file
        x_range (tuple, optional): Energy range to plot (min_kev, max_kev)
    """
    fig, (ax1, ax2) = handler.plot_combined_spectrum(
        fits_path, 
        x_range=x_range,
        normalize=True
    )

def test_data_handler(fits_path, x_range=None):
    """
    Main test function that runs all tests.
    
    Args:
        fits_path (str): Path to FITS file
        x_range (tuple, optional): Energy range to plot (min_kev, max_kev)
    """
    # Create handler
    handler = DataHandler()
    
    # Run individual tests
    test_weighted_spectrum(handler, energy_range=x_range if x_range else (0, 27))
    test_combined_plot(handler, fits_path, x_range)

if __name__ == "__main__":
    # Example usage
    test_data_handler("data\\ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits", x_range=(0, 27)) 