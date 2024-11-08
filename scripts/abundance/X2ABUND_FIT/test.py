# Import necessary modules
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.io import fits
#from spectrafit.plugins.notebook import SpectraFitNotebook
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.stats import chisquare

# Lorentzian function
def lorentzian(x, amplitude, center, width):
    return amplitude * width / ((x - center)**2 + width**2)

# Multi-Lorentzian model
def multi_lorentzian(x, *params):
    result = np.zeros_like(x)
    for i in range(0, len(params), 3):
        amplitude, center, width = params[i:i+3]
        result += lorentzian(x, amplitude, center, width)
    return result

# Fit multiple Lorentzians to data with Savitzky-Golay filtering
def fit_multi_lorentzian(list_, lower_limit, upper_limit, elements, plot_semilog=True, wid_init=0.01, plot=True):
    conversion_factor = 0.0135

    # Load datax
    if len(list_) == 1:
        fits_data = list_[0]
        with fits.open(fits_data) as hdul:
            record_array = hdul[1].data.view(np.recarray)
            x = record_array.CHANNEL
            y = record_array.COUNTS
            x1 = x * conversion_factor
    else:
        x1, y = list_

    # Select range for fitting
    x = x1[int(lower_limit / conversion_factor): int(upper_limit / conversion_factor)]
    y = y[int(lower_limit / conversion_factor): int(upper_limit / conversion_factor)]
    y = np.where(y == 0, 0.00001, y)

    # Plot original data
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='Original Data')
        plt.title('Original Data')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()

    # Apply Savitzky-Golay filter for smoothing
    y_smooth = savgol_filter(y, window_length=11, polyorder=4)

    # Plot smoothed data
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='Original Data', alpha=0.5)
        plt.plot(x, y_smooth, label='Smoothed Data', color='red')
        plt.title('Data after Savitzky-Golay Filtering')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()

    # Define initial guesses for Lorentzian peaks
    amp_init = 1.5
    element_peak = {'Mg': 1.25, 'Al': 1.49, 'Si': 1.74}
    initial_guess = []
    bounds_lower = []
    bounds_upper = []

    for element in elements:
        initial_guess.append(amp_init)
        initial_guess.append(element_peak[element])
        initial_guess.append(wid_init)

        # Set bounds for each parameter
        bounds_lower += [0, element_peak[element] - 0.1, 0.01]
        bounds_upper += [np.inf, element_peak[element] + 0.1, 0.2]

    # Perform the fit on the smoothed data with bounds
    popt, pcov = curve_fit(multi_lorentzian, x, y_smooth, p0=initial_guess, bounds=(bounds_lower, bounds_upper))

    # Generate fitted curve
    y_fit = multi_lorentzian(x, *popt)

    # Plot the fitted data
    if plot:
        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label='Original Data', alpha=0.5)
        plt.plot(x, y_smooth, label='Smoothed Data', color='red')
        plt.plot(x, y_fit, label='Fitted Data', color='green')
        plt.title('Multi-Lorentzian Fit')
        plt.xlabel('Energy (keV)')
        plt.ylabel('Counts')
        plt.legend()
        plt.show()

    # Restrict y and y_fit to the range 0 to 2 keV for chi-square calculation
    idx_range = np.where((x >= 1) & (x <= 2))
    y_in_range = y[idx_range]
    y_fit_in_range = y_fit[idx_range]

    # Normalize y and y_fit to the same sum for chi-square
    y_sum = np.sum(y_in_range)
    y_fit_sum = np.sum(y_fit_in_range)
    y_normalized = y_in_range * (y_fit_sum / y_sum)
    y_fit_normalized = y_fit_in_range

    # Perform chi-square test
    chi2, pval = chisquare(y_normalized, f_exp=y_fit_normalized)

    print("Chi-squared (0 to 2 keV):", chi2)
    print("p-value (0 to 2 keV):", pval)

    return x, y_fit

# Fit synthetic data with SpectraFit and calculate area ratios

# Example usage0T203137197.fits
fits_file = '/home/manasj/isro/code/combined_fits.fits'
x, y_fit = fit_multi_lorentzian([fits_file], 1, 2, ['Mg', 'Al', 'Si'], plot_semilog=True, wid_init=0.01, plot=True)