import numpy as np
from typing import Dict, Optional

class XRFConcentrationSolver:
    def __init__(self):
        """Initialize the XRF concentration solver with predefined elements and coefficients"""
        # Define elements in the same order as the table
        self.elements = ['O', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'K', 'Ca', 'Ti', 'Cr', 'Mn', 'Fe']
        self.n_elements = len(self.elements)
        
        # Create element index mapping for easy lookup
        self.element_indices = {elem: idx for idx, elem in enumerate(self.elements)}
        
        # Initialize coefficient matrices from Table 1
        self._initialize_coefficients()
    
    def _initialize_coefficients(self):
        """Initialize the coefficient matrices using complete data from Table 1"""
        n = self.n_elements
        
        # Initialize primary influence coefficients (aij)
        self.aij = np.zeros((n, n))
        # Fill in known values from first row of table
        self.aij[self.element_indices['Fe']] = [
            0.0000, -0.5588, -0.4261, -0.2820, -0.1143, 0.0741, 0.2772, 
            0.9789, 1.2330, 1.8293, 2.5307, -0.1026, 0.0000
        ]
        
        # Initialize secondary influence coefficients (aijj)
        self.aijj = np.zeros((n, n))
        # Fill in known values from second row of table
        self.aijj[self.element_indices['Fe']] = [
            0.0000, -0.0361, -0.0497, -0.0642, -0.0804, -0.0979, -0.1155,
            -0.1687, -0.1852, -0.2196, -0.2515, 0.0000, 0.0000
        ]
        
        # Initialize ternary influence coefficients (aijk)
        self.aijk = np.zeros((n, n, n))
        
        # Fill in the known aijk values from the table
        # Note: these are the values from the table for element interactions
        aijk_data = {
            ('O', 'Na'): 0.0000,
            ('O', 'Mg'): 0.0000,
            ('O', 'Al'): 0.0000,
            ('O', 'Si'): 0.0000,
            ('O', 'P'): 0.0000,
            ('O', 'S'): 0.0000,
            ('O', 'K'): 0.0000,
            ('O', 'Ca'): 0.0000,
            ('O', 'Ti'): 0.0000,
            ('O', 'Cr'): 0.0000,
            ('O', 'Mn'): 0.0000,
            ('Na', 'Mg'): 0.0005,
            ('Na', 'Al'): 0.0018,
            ('Na', 'Si'): 0.0038,
            ('Na', 'P'): 0.0063,
            ('Na', 'S'): 0.0089,
            ('Na', 'K'): 0.0171,
            ('Na', 'Ca'): 0.0196,
            ('Na', 'Ti'): 0.0245,
            ('Na', 'Cr'): 0.0291,
            ('Na', 'Mn'): 0.0361,
            ('Mg', 'Al'): 0.0004,
            ('Mg', 'Si'): 0.0015,
            ('Mg', 'P'): 0.0032,
            ('Mg', 'S'): 0.0053,
            ('Mg', 'K'): 0.0120,
            ('Mg', 'Ca'): 0.0142,
            ('Mg', 'Ti'): 0.0186,
            ('Mg', 'Cr'): 0.0227,
            ('Mg', 'Mn'): 0.0497,
            ('Al', 'Si'): 0.0004,
            ('Al', 'P'): 0.0014,
            ('Al', 'S'): 0.0028,
            ('Al', 'K'): 0.0082,
            ('Al', 'Ca'): 0.0100,
            ('Al', 'Ti'): 0.0139,
            ('Al', 'Cr'): 0.0176,
            ('Al', 'Mn'): 0.0638,
            ('Si', 'P'): 0.0003,
            ('Si', 'S'): 0.0011,
            ('Si', 'K'): 0.0052,
            ('Si', 'Ca'): 0.0067,
            ('Si', 'Ti'): 0.0100,
            ('Si', 'Cr'): 0.0133,
            ('Si', 'Mn'): 0.0792,
            ('P', 'S'): 0.0003,
            ('P', 'K'): 0.0030,
            ('P', 'Ca'): 0.0042,
            ('P', 'Ti'): 0.0069,
            ('P', 'Cr'): 0.0097,
            ('P', 'Mn'): 0.0952,
            ('S', 'K'): 0.0015,
            ('S', 'Ca'): 0.0024,
            ('S', 'Ti'): 0.0046,
            ('S', 'Cr'): 0.0070,
            ('S', 'Mn'): 0.1107,
            ('K', 'Ca'): 0.0001,
            ('K', 'Ti'): 0.0009,
            ('K', 'Cr'): 0.0021,
            ('K', 'Mn'): 0.1537,
            ('Ca', 'Ti'): 0.0004,
            ('Ca', 'Cr'): 0.0013,
            ('Ca', 'Mn'): 0.1661,
            ('Ti', 'Cr'): 0.0003,
            ('Ti', 'Mn'): 0.1902,
            ('Cr', 'Mn'): 0.2123
        }
        
        # Fill the aijk tensor with the data
        for (elem1, elem2), value in aijk_data.items():
            i1, i2 = self.element_indices[elem1], self.element_indices[elem2]
            self.aijk[i1, i2, :] = value
            self.aijk[i2, i1, :] = value  # Symmetric property
    
    def analyze_sample(self, 
                      intensities: Dict[str, float], 
                      max_iterations: int = 50,
                      tolerance: float = 1e-6) -> Dict[str, float]:
        """
        Analyze a sample to determine all element concentrations
        
        Args:
            intensities: Dictionary mapping element symbols to their measured intensities
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            
        Returns:
            Dictionary of calculated concentrations for all elements
        """
        # Convert intensities dict to array
        intensity_array = np.array([intensities[elem] for elem in self.elements])
        
        # Get initial estimates using Claisse-Quintin
        concentrations = self.claisse_quintin_estimate(
            intensity_array,
            max_iterations=max_iterations,
            tolerance=tolerance
        )
        
        # Convert results back to dictionary
        return {elem: conc for elem, conc in zip(self.elements, concentrations)}
    
    def claisse_quintin_estimate(self, 
                               intensities: np.ndarray,
                               max_iterations: int = 50,
                               tolerance: float = 1e-6) -> np.ndarray:
        """Calculate concentrations using the Claisse-Quintin equation"""
        # Initial guess - equal concentrations summing to 1
        concentrations = np.ones(self.n_elements) / self.n_elements
        
        for iteration in range(max_iterations):
            prev_concentrations = concentrations.copy()
            
            for i in range(self.n_elements):
                # Calculate matrix effects term
                matrix_effect = 1.0
                C_M = np.sum(prev_concentrations)
                
                for j in range(self.n_elements):
                    if j != i:
                        Cj = prev_concentrations[j]
                        # Primary and secondary effects
                        matrix_effect += Cj * (self.aij[i,j] + self.aijj[i,j] * C_M)
                        
                        # Ternary effects
                        for k in range(self.n_elements):
                            if k != i and k != j:
                                Ck = prev_concentrations[k]
                                matrix_effect += Cj * Ck * self.aijk[i,j,k]
                
                # Update concentration
                concentrations[i] = intensities[i] * matrix_effect
            
            # Normalize to sum to 1
            concentrations = concentrations / np.sum(concentrations)
            
            # Check convergence
            if np.max(np.abs(concentrations - prev_concentrations)) < tolerance:
                break
        
        return concentrations

# Example usage:
if __name__ == "__main__":
    # Create solver instance
    solver = XRFConcentrationSolver()
    
    # Example intensities (these should be replaced with real measurements)
    test_intensities = {
        'O': 0.0, 'Na': 0.1, 'Mg': 0.2, 'Al': 0.3, 'Si': 0.4,
        'P': 0.1, 'S': 0.1, 'K': 0.2, 'Ca': 0.3,
        'Ti': 0.2, 'Cr': 0.1, 'Mn': 0.1, 'Fe': 0.4
    }
    
    # Analyze sample
    results = solver.analyze_sample(test_intensities)
    
    # Print results
    print("\nCalculated concentrations:")
    for element, concentration in results.items():
        print(f"{element}: {concentration:.4f}")