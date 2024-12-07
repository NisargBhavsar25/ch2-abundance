import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from typing import Dict
import os
import sys
import argparse

from scripts.fp_solver.claisse_quintin import XRFConcentrationSolver
from scripts.fp_solver.intensity_finder import XRFSpectrumAnalyzer

class XRFAnalyzer:
    def __init__(self):
        """Initialize both intensity and concentration analyzers"""
        self.intensity_analyzer = XRFSpectrumAnalyzer()
        self.concentration_solver = XRFConcentrationSolver()
        
    def analyze_sample(self, 
                      sample_fits: str,
                      background_fits: str,
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
            Tuple of dictionaries containing intensities, concentrations, and uncertainties
        """
        # Step 1: Calculate intensities
        if verbose == 1:
            print("\nAnalyzing spectrum...")
        intensities, uncertainties = self.intensity_analyzer.analyze_spectrum(
            sample_fits,
            background_fits,
            plot_results=plot_results,
            use_background=use_background,
            use_y=use_y,
            y_file=y_file,
            verbose=verbose
        )
        
        # Clean up negative or invalid intensities
        for key in intensities.keys():
            if intensities[key] < 0 or np.isnan(intensities[key]) or np.isinf(intensities[key]):
                intensities[key] = 0
                uncertainties[key] = 0

        # Step 2: Calculate concentrations
        if verbose == 1:
            print("\nCalculating concentrations...")
        concentrations = self.concentration_solver.analyze_sample(intensities)
        
        # Calculate concentration uncertainties
        concentration_uncertainties = {}
        for element in concentrations.keys():
            if intensities[element] > 0:
                concentration_uncertainties[element] = uncertainties[element] * concentrations[element]
            else:
                concentration_uncertainties[element] = 0

        intensities_uncertainties = {}
        for element in intensities.keys():
            if intensities[element] > 0:
                intensities_uncertainties[element] = uncertainties[element] * intensities[element]
            else:
                intensities_uncertainties[element] = 0
            
        if plot_results:
            self.plot_results(intensities, concentrations)
            
        return intensities, concentrations, intensities_uncertainties, concentration_uncertainties
    
    def plot_results(self, intensities: Dict[str, float], concentrations: Dict[str, float]):
        """Plot analysis results with two subplots showing intensities and concentrations"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        elements = list(intensities.keys())
        
        # Plot intensities
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

def print_results(sample_file: str, 
                 background_file: str, 
                 intensities: Dict[str, float], 
                 concentrations: Dict[str, float],
                 intensity_uncertainties: Dict[str, float],
                 concentration_uncertainties: Dict[str, float]):
    """Print analysis results in a formatted table"""
    print("\n" + "="*80)
    print("XRF Analysis Results".center(80))
    print("="*80)
    print(f"\nSample file: {sample_file}")
    print(f"Background file: {background_file}")
    
    print("\n" + "-"*80)
    print(f"{'Element':^10} | {'Intensity':^15} ± {'Uncertainty':^15} | {'Concentration (%)':^15} ± {'Uncertainty (%)':^15}")
    print("-"*80)
    
    for element in intensities.keys():
        intensity = intensities[element]
        concentration = concentrations[element] * 100  # Convert to percentage
        intensity_unc = intensity_uncertainties[element]
        concentration_unc = concentration_uncertainties[element] * 100  # Convert to percentage
        
        print(f"{element:^10} | {intensity:^15.4f} ± {intensity_unc:^15.4f} | {concentration:^15.4f} ± {concentration_unc:^15.4f}")
    
    print("-"*80)

def main():
    # Set up argument parser with default values
    parser = argparse.ArgumentParser(description='XRF Analysis Tool')
    parser.add_argument('--sample_file', 
                       default=r'scripts\fp_solver\ch2_cla_l1_20240221T230106660_20240221T230114659.fits',
                       help='Path to FITS file')
    parser.add_argument('--background_file', 
                       default=r'scripts\fp_solver\ch2_cla_l1_20230902T064630474_20230902T064638474_BKG.pha',
                       help='Path to background FITS/PHA file')
    parser.add_argument('--plot', action='store_true', help='Plot the results')
    parser.add_argument('--output', '-o', help='Output file path for results')
    
    args = parser.parse_args()
    
    # Verify files exist
    if not os.path.exists(args.sample_file):
        raise FileNotFoundError(f"Sample file not found: {args.sample_file}")
    if not os.path.exists(args.background_file):
        raise FileNotFoundError(f"Background file not found: {args.background_file}")
    
    try:
        # Create analyzer instance and perform analysis
        analyzer = XRFAnalyzer()
        intensities, concentrations, intensity_uncertainties, concentration_uncertainties = analyzer.analyze_sample(
            args.sample_file,
            args.background_file,
            plot_results=args.plot,
            use_background=True
        )
        
        # Print results to console
        print_results(
            args.sample_file,
            args.background_file,
            intensities,
            concentrations,
            intensity_uncertainties,
            concentration_uncertainties
        )
        
        # Save results to file if specified
        if args.output:
            with open(args.output, 'w') as f:
                f.write("XRF Analysis Results\n")
                f.write("===================\n\n")
                f.write(f"Sample file: {args.sample_file}\n")
                f.write(f"Background file: {args.background_file}\n\n")
                f.write(f"{'Element':^10} | {'Intensity':^15} ± {'Uncertainty':^15} | {'Concentration (%)':^15} ± {'Uncertainty (%)':^15}\n")
                f.write("-"*80 + "\n")
                
                for element in intensities.keys():
                    intensity = intensities[element]
                    concentration = concentrations[element] * 100
                    intensity_unc = intensity_uncertainties[element]
                    concentration_unc = concentration_uncertainties[element] * 100
                    
                    f.write(f"{element:^10} | {intensity:^15.4f} ± {intensity_unc:^15.4f} | {concentration:^15.4f} ± {concentration_unc:^15.4f}\n")
                
                print(f"\nResults saved to {args.output}")
        
    except Exception as e:
        print(f"\nError during analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()