import numpy as np
import xraylib
from scipy.stats import norm

class ElementModel:
    def __init__(self, element, conc, std_dev):
        """
        Initialize the ElementModel with the specified element, conc, and standard deviation.
        
        Args:
            element (str): The symbol of the element (e.g., 'Fe').
            conc (float): The conc of the Gaussian.
            std_dev (float): The standard deviation of the Gaussian.
        """
        self.element = element
        self.conc = conc
        self.std_dev = std_dev
        self.Z = xraylib.SymbolToAtomicNumber(element)  # Atomic number

        # Convert element symbol to atomic number if needed
        self.Z = element if isinstance(element, int) else xraylib.SymbolToAtomicNumber(element)
        
        # Get K and L emission lines energies and probabilities
        energy_dict = {}
        radrate = {}
        
        lines=[]
        # K lines
        self.lines_map={}
        ka_lines = [
            (xraylib.KA1_LINE, "ka1"),
            # (xraylib.KA2_LINE, "ka2"),
        ]
        kb_lines= [(xraylib.KB1_LINE, "kb1"),
            # (xraylib.KB2_LINE, "kb2")
            ]
        # Add K lines to means and radrate
        peak_energies={}
        peak_rad_rates={}
        for line, label in ka_lines:
            try:
                energy = xraylib.LineEnergy(self.Z, line)
                prob = xraylib.RadRate(self.Z, line)
                if energy > 0 and prob > 0:
                    self.lines_map[label]=line        
                    peak_energies[f"{label}"] = energy
                    peak_rad_rates[f"{label}"] = prob
                    lines.append(label)

            except:
                continue
        energy_dict["ka"]=peak_energies
        radrate["ka"]=peak_rad_rates
        peak_energies={}
        peak_rad_rates={}
        for line, label in kb_lines:
            try:
                energy = xraylib.LineEnergy(self.Z, line)
                prob = xraylib.RadRate(self.Z, line)
             
                if energy > 0 and prob > 0:
                    self.lines_map[label]=line        
                    peak_energies[f"{label}"] = energy
                    peak_rad_rates[f"{label}"] = prob
                    lines.append(label)

            except:
                continue
        energy_dict["kb"]=peak_energies
        radrate["kb"]=peak_rad_rates
        
        # L lines (you can add more if needed)
 # Add L lines to means and radrate
         # L lines
        la_lines = [
            # (xraylib.LA1_LINE, "la1"),
        ]
        lb_lines = [  
            #  (xraylib.LB1_LINE, "lb1"),
            # (xraylib.LB2_LINE, "lb2")
            ]
        peak_energies_la = {}
        peak_rad_rates_la = {}
        for line, label in la_lines:
            try:
                energy = xraylib.LineEnergy(self.Z, line)
                prob = xraylib.RadRate(self.Z, line)
                if energy > 0 and prob > 0:
                    self.lines_map[label]=line 
       
                    peak_energies_la[label] = energy
                    peak_rad_rates_la[label] = prob
                    lines.append(label)

            except:
                continue
        
        energy_dict["la"] = peak_energies_la
        radrate["la"] = peak_rad_rates_la
        peak_energies_lb = {}
        peak_rad_rates_lb = {}
        for line, label in la_lines:
            try:
                energy = xraylib.LineEnergy(self.Z, line)
                prob = xraylib.RadRate(self.Z, line)
                if energy > 0 and prob > 0:
                    self.lines_map[label]=line        
                    peak_energies_lb[label] = energy
                    peak_rad_rates_lb[label] = prob
                    lines.append(label)
            except:
                continue
        
        energy_dict["lb"] = peak_energies_lb
        radrate["lb"] = peak_rad_rates_lb
        means={}
        for key,levels in energy_dict.items():
            means[key]=np.mean(list(levels.values()))
        
        for key in energy_dict.keys():
            energy_dict[key]["mean"]=means[key]

        for key,levels in radrate.items():
            means[key]=np.mean(list(levels.values()))
        
        for key in radrate.keys():
            radrate[key]["mean"]=means[key]

        self.energy_dict = energy_dict
        self.radrates=radrate
        self.lines=lines
        self.line_div={}
        for line in lines:
            if line not in self.line_div.keys():
                self.line_div[line[:2]]=[]
            self.line_div[line[:2]].append(line)
        self.std_dev={line[:2]:0.01 for line in self.energy_dict.keys()}
        # self.std_devs = np.array(std_dev * len(energy_dict))
        # self.radrate = np.array(list(radrate.values()))

    def calculate_mass_absorption_coefficient(self, energy):
        """
        Calculate the mass absorption coefficient for the element at the given energy.
        
        Args:
            energy (float): The energy value in keV.
        
        Returns:
            float: The mass absorption coefficient in cm^2/g.
        """
        return xraylib.CS_Total(self.Z, energy)  # Total cross-section as a proxy
    
    def calculate_mass_absorption_coefficient(self, element,line):
        """
        Calculate the mass absorption coefficient for the element at the given energy.
        
        Args:
            energy (float): The energy value in keV.
        
        Returns:
            float: The mass absorption coefficient in cm^2/g.
        """
        Z = xraylib.SymbolToAtomicNumber(element)  # Atomic number
        if line not in self.lines:
            return 0
        line_num=self.lines_map[line]
        # energy= xraylib.LineEnergy(Z,line_num )
        energy=self.energy_dict[line[:2]][line]
        return xraylib.CS_Total(self.Z, energy)  # Total cross-section as a proxy
    def calulate_elemental_const(self,line,ey):
        if "ka" in line:
            ltype="ka"
            line_energy=self.energy_dict["ka"]["mean"]
            rk = xraylib.JumpFactor(self.Z, 0)  # Get jump ratio for the specific line
        if "kb" in line:
            ltype="kb"
            line_energy=self.energy_dict["kb"]["mean"]
            rk = xraylib.JumpFactor(self.Z,0)  # Get jump ratio for the specific line
        if "la" in line:
            ltype="la"
            line_energy=self.energy_dict["la"]["mean"]
            rk = xraylib.JumpFactor(self.Z,1)  # Get jump ratio for the specific line
        if "lb" in line:
            ltype="lb"
            line_energy=self.energy_dict["lb"]["mean"]
            rk = xraylib.JumpFactor(self.Z,1)  # Get jump ratio for the specific line
        try:
            fluor_yield=xraylib.FluorYield(self.Z,0 if "k" in line else 1)
        except Exception as e:
            print(e,self.element,line_energy,"KeV")
            return 0
        c= (1-1/(rk+1e-9))*fluor_yield*self.radrates[ltype]["mean"]
        return c

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
        return self.conc * norm.pdf(x, mean, std_dev)
    def primary_intensity(self, line):
        """
        Calculate the weighted Gaussian function value at the specified energy.
        
        Args:
            energy (float): The energy value in keV.
        
        Returns:
            float: The weighted Gaussian value at the specified energy.
        """
        energy_mean=self.energy_dict[line[:2]]["mean"]
        mass_absorption_coeff = self.calculate_mass_absorption_coefficient(energy_mean)
        # # Define the number of bins based on the standard deviation
        # num_bins = int(2 * self.std_dev[line] / 0.0277) + 1
        # # Create the energies array centered around energy_mean
        # energies = np.linspace(energy_mean - self.std_dev[line], 
        #                         energy_mean + self.std_dev[line], 
        #                         num_bins)
        # fulle=np.zeros(2048)
        # binstart = int((energies[0] - 0) / 0.0277)  # Assuming 0 is the starting energy
        # binend = int((energies[-1] - 0) / 0.0277)    # Assuming 0 is the starting energy
        
        # fulle[binstart:binend]=self.gaussian(energies, self.energy_dict[line[:2]]["mean"],std_dev=self.std_dev[line[:2]]) * mass_absorption_coeff
        return mass_absorption_coeff*self.conc
    def jump_ratio_factor(self, shell):
        """
        Calculate the jump ratio rk for the given energy using xraylib.
        
        Args:
            energy (float): The energy value in keV.
        
        Returns:
            float: The jump ratio rk at the specified energy.
        """
        rk=xraylib.JumpFactor(self.Z, shell)  # Assuming JumpFactor returns rk
        if rk == 0:
            raise ValueError("rk must not be zero to avoid division by zero.")
        return 1- 1 / rk
        
# Example usage:
if __name__ == "__main__":
    element = 'Fe'  # Iron
    energy = 6.4  # Energy in keV
    conc = 1.0  # conc
    std_dev = 0.1  # Standard deviation
    
    model = ElementModel(element, conc, std_dev)
    result = model.primary_intensity(energy)
    print(f"Weighted Gaussian value for {element} at {energy} keV: {result}") 