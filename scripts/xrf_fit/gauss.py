import numpy as np
from scipy.stats import norm
import xraylib
import matplotlib.pyplot as plt

class GaussianSum:
    def __init__(self, element, std_devs):
        """
        Initialize a sum of Gaussian functions for X-ray emission lines.
        
        Args:
            element (str or int): Element symbol (e.g., 'Fe') or atomic number
            std_devs (array-like): Standard deviations for each Gaussian
        """
        # Convert element symbol to atomic number if needed
        self.Z = element if isinstance(element, int) else xraylib.SymbolToAtomicNumber(element)
        
        # Get K and L emission lines energies and probabilities
        means_dict = {}
        amplitudes = {}
        
        # K lines
        k_lines = [
            (xraylib.KA1_LINE, "Ka1"),
            (xraylib.KA2_LINE, "Ka2"),
            (xraylib.KB1_LINE, "Kb1"),
            (xraylib.KB2_LINE, "Kb2")
        ]
        
        # Add K lines to means and amplitudes
        for line, label in k_lines:
            try:
                energy = xraylib.LineEnergy(self.Z, line)
                prob = xraylib.RadRate(self.Z, line)
                if energy > 0 and prob > 0:
                    means_dict[f"{label}"] = energy
                    amplitudes[f"{label}"] = prob
            except:
                continue
        
        # L lines (you can add more if needed)
        l_lines = [
            (xraylib.LA1_LINE, "La1"),
            (xraylib.LB1_LINE, "Lb1"),
            (xraylib.LB2_LINE, "Lb2")
        ]
        
        # Add L lines to means and amplitudes
        for line, label in l_lines:
            try:
                energy = xraylib.LineEnergy(self.Z, line)
                prob = xraylib.RadRate(self.Z, line)
                if energy > 0 and prob > 0:
                    means_dict[f"{label}"] = energy
                    amplitudes[f"{label}"] = prob
            except:
                continue

        self.means = means_dict
        self.std_devs = np.array([std_devs] * len(means_dict))
        self.amplitudes = np.array(list(amplitudes.values()))
        
        # Validate inputs
        if len(self.std_devs) != len(self.amplitudes):
            raise ValueError("Number of standard deviations must match number of amplitudes")
        if len(self.std_devs) != len(means_dict):
            raise ValueError("Number of parameters must match number of means in dictionary")

    def single_gaussian(self, x, mean, std_dev, amplitude):
        """Calculate a single Gaussian function using scipy.stats.norm."""
        if std_dev==0:
            return 0
        return amplitude * norm.pdf(x, mean, std_dev)

    def __call__(self, x):
        """
        Calculate the sum of all Gaussian functions at point(s) x.
        
        Args:
            x (array-like): Points at which to evaluate the sum of Gaussians
            
        Returns:
            array-like: Sum of all Gaussian functions evaluated at x
        """
        result = np.zeros_like(x, dtype=float)
        
        for (key, mean), std_dev, amp in zip(self.means.items(), self.std_devs, self.amplitudes):
            result += self.single_gaussian(x, mean, std_dev, amp)
            
        return result
    @staticmethod
    def calculate_gaussian_sum(x,means, std_devs, amplitudes):
        """
        Calculate the sum of Gaussian functions with explicit parameters.
        
        Args:
            x (array-like): Points at which to evaluate the sum of Gaussians
            means (array-like): Mean values for each Gaussian
            std_devs (array-like): Standard deviations for each Gaussian
            amplitudes (array-like): Amplitude values for each Gaussian
            
        Returns:
            array-like: Sum of all Gaussian functions evaluated at x
        """
        result = np.zeros_like(x, dtype=float)
        
        for mean, std_dev, amp in zip(means, std_devs, amplitudes):
            result += amp * norm.pdf(x, mean, std_dev)
            
        return result
    def plot(self, x_range=(0, 10), num_points=1000):
        """
        Plot the individual Gaussians and their sum versus energy in KeV.
        
        Args:
            x_range (tuple): (min_energy, max_energy) in KeV to plot
            num_points (int): Number of points to evaluate the functions
            
        Returns:
            tuple: (fig, ax) matplotlib figure and axes objects
        """
        
        x = np.linspace(x_range[0], x_range[1], num_points)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot individual Gaussians
        for (label, mean), std_dev, amp in zip(self.means.items(), self.std_devs, self.amplitudes):
            y = self.single_gaussian(x, mean, std_dev, amp)
            ax.plot(x, y, '--', alpha=0.5, label=f'{label} ({mean:.2f} KeV)')
        
        # Plot sum of all Gaussians
        ax.plot(x, self.__call__(x), 'k-', label='Sum', linewidth=2)
        
        ax.set_xlabel('Energy (KeV)')
        ax.set_ylabel('Intensity (a.u.)')
        ax.set_title(f'X-ray Emission Lines for Z={self.Z}')
        ax.legend()
        ax.grid(True)
        
        return fig, ax

def test_gaussian_sum(element='Al', std_dev=0.1, x_range=(0, 8)):
    """
    Test function to demonstrate the GaussianSum class.
    
    Args:
        element (str or int): Element symbol or atomic number
        std_dev (float): Standard deviation for the Gaussian peaks
        x_range (tuple): (min_energy, max_energy) in KeV to plot
    """
    # Create GaussianSum instance
    gs = GaussianSum(element, std_dev)
    
    # Create and show the plot
    fig, ax = gs.plot(x_range=x_range)
    plt.show()
    
    # Print emission line information
    print(f"\nEmission lines for {element} (Z={gs.Z}):")
    for label, energy in gs.means.items():
        print(f"{label}: {energy:.2f} KeV")

if __name__ == "__main__":
    test_gaussian_sum()
