import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

class XRFSimulator:
    def __init__(self, fano_factor, incident_angle, detector_angle):
        self.E_pair = 0.00385  # Energy to create electron-hole pair in Si (keV)
        self.electronic_noise = 0.1  # Electronic noise value
        self.fano = fano_factor
        self.angle_incident = np.radians(incident_angle)  # Convert to radians
        self.angle_detector = np.radians(detector_angle)  # Convert to radians

    def fundamental_parameters(self, element_conc, geom_factor, absorption_coeff, excitation_factor, mass_thickness):
        """
        Calculate primary fluorescence intensity using FP equation
        Using angles provided in the input
        """
        mu_lambda = absorption_coeff['mu_lambda']
        mu_i = absorption_coeff['mu_i']
        tau = absorption_coeff['tau']
        
        sin_psi1 = np.sin(self.angle_incident)
        mu_prime_lambda = mu_lambda / sin_psi1
        mu_prime_i = mu_i / np.sin(self.angle_detector)
        
        exp_term = 1 - np.exp(-(mu_prime_lambda + mu_prime_i) * mass_thickness)
        intensity = (geom_factor / sin_psi1) * element_conc * excitation_factor * tau
        intensity *= exp_term / (mu_prime_lambda + mu_prime_i)
        
        return intensity

    def apply_gaussian_broadening(self, energies, intensities, energy_points=1000):
        """
        Apply Gaussian broadening to X-ray lines using Fano factor
        """
        energy_scale = np.linspace(min(energies)-1, max(energies)+1, energy_points)
        broadened_spectrum = np.zeros_like(energy_scale)
        
        for energy, intensity in zip(energies, intensities):
            variance = (self.electronic_noise/2.3548)**2 + self.E_pair * self.fano * energy
            sigma = np.sqrt(variance)
            
            gaussian = stats.norm.pdf(energy_scale, energy, sigma)
            broadened_spectrum += intensity * gaussian
            
        return energy_scale, broadened_spectrum

def simulate_and_plot_xrf_spectrum(fano_factor, incident_angle, detector_angle):
    simulator = XRFSimulator(fano_factor, incident_angle, detector_angle)
    
    # Parameters common for all elements
    base_params = {
        'geom_factor': 0.1,
        'absorption_coeff': {
            'mu_lambda': 100,
            'mu_i': 80,
            'tau': 0.5
        },
        'excitation_factor': 0.8,
        'mass_thickness': 0.01
    }
    
    # Element concentrations and characteristic lines
    elements = {
        'Si': {'concentration': 0.20, 'lines': [(1.74, 1.0), (1.83, 0.1)]},  # Kα, Kβ
        'Mg': {'concentration': 0.19, 'lines': [(1.25, 1.0), (1.30, 0.1)]},
        'Fe': {'concentration': 0.10, 'lines': [(6.40, 1.0), (7.06, 0.13)]},
        'Ca': {'concentration': 0.03, 'lines': [(3.69, 1.0), (4.01, 0.13)]},
        'Al': {'concentration': 0.03, 'lines': [(1.49, 1.0), (1.55, 0.1)]}
    }
    
    # Calculate intensities for all elements
    energies = []
    intensities = []
    peak_labels = []
    raw_energies = []  # Raw energies for the peaks before broadening
    raw_intensities = []  # Raw intensities for the peaks before broadening
    
    for element, info in elements.items():
        params = base_params.copy()
        raw_intensity = simulator.fundamental_parameters(
            element_conc=info['concentration'],
            **params
        )
        
        for (energy, relative_intensity), line_type in zip(info['lines'], ['Kα', 'Kβ']):
            raw_energies.append(energy)
            raw_intensities.append(raw_intensity * relative_intensity)
            peak_labels.append(f'{element} {line_type}')
    
    # Apply Gaussian broadening
    energy_scale, spectrum = simulator.apply_gaussian_broadening(raw_energies, raw_intensities)
    
    # Create plot
    plt.figure(figsize=(12, 6))
    
    # Modify the stem plot to have the red lines reach the blue line
    spectrum_interpolated = np.interp(raw_energies, energy_scale, spectrum)  # Interpolated values at raw_energies

    plt.stem(raw_energies, spectrum_interpolated, 'r', label='Raw X-ray lines', linefmt='r-', markerfmt='ro', basefmt=' ')

    # Plot broadened spectrum
    plt.plot(energy_scale, spectrum, 'b-', linewidth=2, label='XRF Spectrum')
    
    # Add peak labels with black triangles and yellow arrows pointing to each red point
    for energy, intensity, label in zip(raw_energies, spectrum_interpolated, peak_labels):
        plt.annotate(label, 
                    xy=(energy, intensity),
                    xytext=(10, 10), textcoords='offset points',
                    ha='left', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        plt.plot(energy, intensity, marker='v', color='black', markersize=8)  # Black triangle

    plt.xlabel('Energy (keV)', fontsize=12)
    plt.ylabel('Intensity (a.u.)', fontsize=12)
    title = f'Simulated XRF Spectrum\nIncident Angle: {incident_angle}°, Detector Angle: {detector_angle}°'
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.margins(x=0.1)
    
    return energy_scale, spectrum, plt.gcf()

# Run simulation and create plot
if __name__ == "__main__":
    # Define input parameters
    fano_factor = 0.17770213  # Example value    # Minimum = 0.15940477    # Maximum = 0.19984151    # Average = 0.17770213
    incident_angle = 45  # Example value in degrees
    detector_angle = 30  # Example value in degrees
    
    energy_scale, spectrum, fig = simulate_and_plot_xrf_spectrum(fano_factor, incident_angle, detector_angle)
    plt.show()
    
    # Print results
    print("Energy range:", energy_scale[0], "to", energy_scale[-1], "keV")
    print("Maximum intensity:", np.max(spectrum))