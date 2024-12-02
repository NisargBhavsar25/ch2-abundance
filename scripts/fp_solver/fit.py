import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from typing import Dict
import os
import sys

sys.path.append('E:\ISRO\ch2-abundance\\')

# from scripts.fp_solver.claisse_quintin import XRFConcentrationSolver
from scripts.fp_solver.claisse_quintin import XRFConcentrationSolver
from scripts.fp_solver.intensity_finder import XRFSpectrumAnalyzer

class XRFAnalyzer:
    def __init__(self):
        """Initialize both intensity and concentration analyzers"""
        self.intensity_analyzer = XRFSpectrumAnalyzer()
        self.concentration_solver = XRFConcentrationSolver()
        
    def analyze_sample(self, 
                      sample_fits: str = r'scripts\fp_solver\ch2_cla_l1_20240221T230106660_20240221T230114659.fits', 
                      background_fits: str = r'scripts\fp_solver\ch2_cla_l1_20230902T064630474_20230902T064638474_BKG.pha', 
                      use_y: bool = False,
                      y_file: np.ndarray = None,
                      use_background: bool = True,
                      plot_results: bool = False,
                      verbose: int = 1) -> Dict[str, float]:
        """
        Perform complete XRF analysis on a sample
        
        Args:
            sample_fits: Path to sample spectrum FITS file
            background_fits: Path to background spectrum FITS/PHA file
            plot_results: Whether to plot spectral analysis results
            
        Returns:
            Dictionary of element concentrations
        """
        # Step 1: Calculate intensities
        if verbose == 1:
            print("Calculating intensities...")
        intensities = self.intensity_analyzer.analyze_spectrum(
            sample_fits,
            background_fits,
            plot_results=plot_results,
            use_background=use_background,
            use_y=use_y,
            y_file=y_file,
            verbose=verbose
        )
        if verbose == 1:
            print("\nCalculated intensities:")
            for element, intensity in intensities.items():
                print(f"{element}: {intensity:.4f}")

            # Step 2: Calculate concentrations
            print("\nCalculating concentrations...")
        concentrations = self.concentration_solver.analyze_sample(intensities)
        
        if verbose == 1:
            print("\nCalculated concentrations (%):")
            for element, concentration in concentrations.items():
                print(f"{element}: {concentration*100:.4f}%")
            
        if plot_results:
            self.plot_results(intensities, concentrations)
            
        return intensities, concentrations
    
    def plot_results(self, intensities: Dict[str, float], concentrations: Dict[str, float]):
        """
        Plot analysis results
        
        Args:
            intensities: Dictionary of calculated intensities
            concentrations: Dictionary of calculated concentrations
        """
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot intensities
        elements = list(intensities.keys())
        intensity_values = [intensities[elem] for elem in elements]
        
        ax1.bar(elements, intensity_values)
        ax1.set_title('XRF Intensities')
        ax1.set_xlabel('Elements')
        ax1.set_ylabel('Intensity')
        ax1.tick_params(axis='x', rotation=45)
        
        # Plot concentrations
        concentration_values = [concentrations[elem]*100 for elem in elements]
        
        ax2.bar(elements, concentration_values)
        ax2.set_title('Element Concentrations')
        ax2.set_xlabel('Elements')
        ax2.set_ylabel('Concentration (%)')
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

def main():
    # Create analyzer instance
    analyzer = XRFAnalyzer()
    
    # Set file paths
    # sample_file = input("Enter path to sample FITS file: ")
    # sample_file = r'scripts\fp_solver\ch2_cla_l1_20240221T230106660_20240221T230114659.fits'
    sample_file = r'scripts\fp_solver\ch2_cla_l1_20240222T062907831_20240222T062915831.fits'
    # sample_file = r'scripts\fp_solver\ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits'
    # background_file = r'scripts\fp_solver\ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits'
    background_file = r'scripts\fp_solver\ch2_cla_l1_20230902T064630474_20230902T064638474_BKG.pha'
    # background_file = input("Enter path to background FITS/PHA file: ")
    
    # Verify files exist
    if not os.path.exists(sample_file):
        raise FileNotFoundError(f"Sample file not found: {sample_file}")
    if not os.path.exists(background_file):
        raise FileNotFoundError(f"Background file not found: {background_file}")
    
    try:
        # Perform analysis
        print("\nStarting XRF analysis...")
        intensities, concentrations = analyzer.analyze_sample(
            sample_file,
            background_file,
            plot_results=True,
            use_background = True
        )
        
        # Save results to file
        output_file = "xrf_analysis_results.txt"
        print(f"\nSaving results to {output_file}")
        
        with open(output_file, 'w') as f:
            f.write("XRF Analysis Results\n")
            f.write("===================\n\n")
            f.write(f"Sample file: {sample_file}\n")
            f.write(f"Background file: {background_file}\n\n")
            f.write("Element Concentrations:\n")
            f.write("---------------------\n")
            for element, concentration in concentrations.items():
                f.write(f"{element}: {concentration*100:.4f}%\n")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()