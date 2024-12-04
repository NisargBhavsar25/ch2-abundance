import numpy as np
from element_model import ElementModel
from scipy.stats import norm
import matplotlib.pyplot as plt
import os

class ElementHandler:
    def __init__(self, std_dev=0.1, concentrations=None, verbose=False, num_channels=2048):
        """
        Initialize ElementHandler with a list of elements and their amplitudes.
        
        Args:
            std_dev (float): Standard deviation for Gaussian peaks
            concentrations (dict): Dictionary of concentrations for each element
            verbose (bool): Flag to enable verbose output
            num_channels (int): Number of energy channels
        """
        self.verbose = verbose
        self.elements = ["Fe", "Al", "Mg", "Si", "Ca", "Ti", "O"]
        # self.elements = ["Al", "Mg", "Si"]

        self.conc = {
            "Fe": 0.3,
            "Al": 0.6,
            "Mg": 0.53,
            "Si": 1,
            "Ca": 0.00047,
            "Ti": 0.0001,
            "O": 2.4
        }
        self.std_dev = {
            "Fe": std_dev,
            "Al": std_dev,
            "Mg": std_dev,
            "Si": std_dev,
            "Ca": std_dev,
            "Ti": std_dev,
            "O": std_dev
        }
        self.beta = 1.0
        self.scale=0.001
        self.num_channels = num_channels
        self.energy_factor = 0.0135 if num_channels == 2048 else 0.0277
        # Initialize ElementModel for each element
        self.element_models = {
            element: ElementModel(element, self.conc[element], self.std_dev[element]) 
            for element in self.elements
        }
    def set_beta(self, el1, v1, el2, v2):
        """
        Set the beta (secondary fluorescence) parameters for a pair of elements.
        
        Args:
            el1 (str): First element symbol
            v1 (float): Beta value for el1 -> el2 interaction
            el2 (str): Second element symbol
            v2 (float): Beta value for el2 -> el1 interaction
        """
        if el1 not in self.elements or el2 not in self.elements:
            raise ValueError(f"Elements must be from the list: {self.elements}")
        
        self.beta[f"{el1}_{el2}"] = v1
        self.beta[f"{el2}_{el1}"] = v2
        
        if self.verbose:
            print(f"Set beta {el1}->{el2} = {v1}, {el2}->{el1} = {v2}")

    def get_beta(self):
        """
        Get the beta parameter for the interaction between two elements.
        Now returns a single beta value regardless of element pair.
        
        Args:
            el1 (str): First element symbol
            el2 (str): Second element symbol
            
        Returns:
            float: Beta value for the interaction
        """
        # Return the single beta value stored for all interactions
        return self.beta

    def set_conc(self, element, conc):
        """
        Set the amplitude for a specific element.
        
        Args:
            element (str): Element 
            amplitude (float): New amplitude value
        """
        if element in self.element_amplitudes:
            self.element_amplitudes[element] = conc
        else:
            raise ValueError(f"Element {element} not in model list")
        
        
    def calculate_secondary_intensity(self, ex, pline):
        """
        Calculate secondary fluorescence intensity with beta parameters.
        
        Args:
            ex (str): Excited element
            pline (str): Spectral line
            
        Returns:
            float: Secondary intensity
        """
        si = 0
        for ey in self.elements:
            if ey == ex:
                continue
            for line in self.element_models[ey].lines:
                iy = self.element_models[ey].calculate_mass_absorption_coefficient(ey, line=line)
                ux_line = self.element_models[ex].calculate_mass_absorption_coefficient(ey, line)
                kx = self.element_models[ex].calulate_elemental_const(pline, self.element_models[ey].energy_dict[line[:2]][line])
                
                # Use beta parameter for secondary fluorescence
                beta = self.get_beta()
                si += iy * ux_line * kx * beta
                
                if self.verbose and iy * ux_line * kx != 0:
                    print(f"Secondary fluorescence: {ey}->{ex}, beta={beta:.3f}, contribution={iy*ux_line*kx*beta:.3e}")
        
        return si
    
    
    def gaussian(self, x,mean,std_dev):
        """
        Calculate the Gaussian function value at the specified energy.
        
        Args:
            energy (float): The energy value in keV.
        
        Returns:
            float: The Gaussian value at the specified energy.
        """
        if std_dev==0:
            return 0
        return norm.pdf(x, mean, std_dev)
    

    def calc_absolute_intensity(self,el,line):
        pi=self.element_models[el].calculate_mass_absorption_coefficient(el,line)
        
        si=self.calculate_secondary_intensity(el,line)

        fulle=self.conc[el]*(pi+si)
        if self.verbose:
            print(f"Element: {el}\n Primary Intensity: {pi} Secondary Intensity: {si}\n Energy:{self.element_models[el].energy_dict[line[:2]][line]}")
        return fulle

    def calculate_folded_intensity(self, energies):
        """
        Calculate the combined intensity from all elements at specified energies.
        
        Args:
            energies (np.array): Energy values for the spectrum
            
        Returns:
            np.array: Combined intensity values
        """
        total_intensity = np.zeros_like(energies)
        for element in self.elements:
            if self.verbose:
                print(f"\nProcessing element: {element}")
                print(f"Line groups: {self.element_models[element].line_div}")
            
            for line_group in self.element_models[element].line_div.values():
                for line in line_group:
                    if self.verbose:
                        print(f"  Processing line: {line}")
                        print(f"  Energy mean: {self.element_models[element].energy_dict[line[:2]]['mean']}")
                        print(f"  Intensity: {self.calc_absolute_intensity(element, line)}")
                    
                    # Define the number of bins based on the standard deviation
                    num_bins = int(2 * self.std_dev[element] / self.energy_factor) + 1 
                    energy_mean=self.element_models[element].energy_dict[line[:2]]["mean"]
                    se = self.energy_factor * int((energy_mean - self.std_dev[element]) / self.energy_factor)
                    te = self.energy_factor * int((energy_mean + self.std_dev[element]) / self.energy_factor)
            #         # Create the energies array centered around energy_mean
                    energies = np.linspace(se, 
                                            te, 
                                            num_bins)
                    fulle=np.zeros(2048)
                    binstart = max(0,int((energy_mean - self.std_dev[element]) / self.energy_factor))
                    binend = min(2048,int((energy_mean + self.std_dev[element]) / self.energy_factor))
                    it=self.calc_absolute_intensity(element,line)
                    # print(element,": ",line,": ",it,energies.shape)
                    # print(binstart,binend,"Bins")
                    gauss=self.gaussian(energies,energy_mean,self.std_dev[element])
                    
                    # Ensure gauss has the same shape as the bins we want to fill
                    if len(gauss) < (binend - binstart):
                        # If gauss is smaller, pad it with zeros
                        gauss = np.pad(gauss, (0, (binend - binstart) - len(gauss)), 'constant')
                    elif len(gauss) > (binend - binstart):
                        # If gauss is larger, truncate it
                        gauss = gauss[:(binend - binstart)]
                    
                    total_intensity[binstart:binend]+=it*gauss
            
        return total_intensity
    
    def plot_intensity(self, energies, intensities, title='Intensity Plot', x_label='Energy (KeV)', y_label='Intensity (a.u.)', filename="plots/phyFit/plot_"):
        """
        Plot the calculated intensity against energy values.
        """
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        # Create an array of x values corresponding to the bin centers
        bin_width = self.energy_factor * 1000 / self.num_channels  # Calculate the width of each bin
        x_values = np.linspace(bin_width / 2, self.energy_factor * 1000 - bin_width / 2, self.num_channels)

        plt.plot(x_values, intensities, label='Calculated Intensity', color='blue')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        
        # Set x-axis ticks at every integer value
        max_energy = int(np.max(x_values))
        plt.xticks(np.arange(0, max_energy + 1, 1))
        
        plt.grid(True)
        plt.legend()
        plt.savefig(filename + "test.png")
        plt.close()

if __name__ == "__main__":
    handler = ElementHandler(verbose=True)
    energies = np.linspace(0, 27, 2048)  # Example energy range
    folded = handler.calculate_folded_intensity(energies)
    print("Combined Intensity:", folded.shape)
    handler.plot_intensity(np.arange(len(folded)), folded, filename="plots/phyFit/folded_intensity_")