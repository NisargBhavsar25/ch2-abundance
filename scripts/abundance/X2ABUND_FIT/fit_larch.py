"""
;====================================================================================================
;                              X2ABUNDANCE
;
; Package for determining elemental weight percentages from XRF line fluxes
;
; Algorithm developed by P. S. Athiray (Athiray et al. 2015)
; Codes in IDL written by Netra S Pillai
; Codes for XSPEC localmodel developed by Ashish Jacob Sam and Netra S Pillai
;
; Developed at Space Astronomy Group, U.R.Rao Satellite Centre, Indian Space Research Organisation
;
;====================================================================================================

This is the local model defined for fitting CLASS data using PyXspec

"""
# Importing necessary modules
import numpy as np
import xraylib
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V3 import xrf_comp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import scipy.stats as stats
import larch
from larch import Group
from larch.fitting import minimize, Parameter, Parameters
# import param_group
from larch.fitting import param_group
# import fit_report
from larch.fitting import fit_report
#import guess
from larch.fitting import guess
import os

# Getting the static parameters for the local model
static_parameter_file = "static_par_localmodel.txt"
fid = open(static_parameter_file, "r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

import astropy.io.fits as fits

fits_filename = "test/ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits"

# load fits
with fits.open(fits_filename) as hdul:
    # get the data and header
    data = hdul[1].data.view(np.recarray)
    x, y = data.CHANNEL, data.COUNTS
# print(x)
print(len(x), len(y))
# duplicate y to double the size. so duplicate ith index to its adjecent index
original_intensities = []

for i in range(len(y)):
    original_intensities.append(y[i])
    original_intensities.append(y[i])
    

print(len(x))
print(len(original_intensities))
        
original_intensities = np.float64(original_intensities)


energy = np.arange(0, 2049, dtype=float)
energy *= 0.0135


# Defining the model function
def phy_model(energy, parameters, original_intensities):
    flux = np.zeros(np.size(energy) - 1)
    # Defining proper energy axis    
    energy_mid = np.zeros(np.size(energy) - 1)
    for i in np.arange(np.size(energy) - 1):
        energy_mid[i] = 0.5 * (energy[i + 1] + energy[i])
        
    # Defining some input parameters required for x2abund xrf computation modules
    at_no = np.array([26, 22, 20, 14, 13, 12, 11, 8])
    
    weight = list(parameters)
    
    i_angle = 90.0 - solar_zenith_angle
    e_angle = 90.0 - emiss_angle
    (energy_solar, tmp1_solar, counts_solar) = readcol(solar_file, format='F,F,F')
        
    # Computing the XRF line intensities
    k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE, xraylib.KM2_LINE, xraylib.KM3_LINE, xraylib.KM4_LINE, xraylib.KM5_LINE])
    l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE, xraylib.L1M3_LINE, xraylib.L1M4_LINE, xraylib.L1M5_LINE, xraylib.L1N1_LINE, xraylib.L1N2_LINE, xraylib.L1N3_LINE, xraylib.L1N4_LINE, xraylib.L1N5_LINE, xraylib.L1N6_LINE, xraylib.L1N7_LINE])
    l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE, xraylib.L2M4_LINE, xraylib.L2M5_LINE, xraylib.L2N1_LINE, xraylib.L2N2_LINE, xraylib.L2N3_LINE, xraylib.L2N4_LINE, xraylib.L2N5_LINE, xraylib.L2N6_LINE, xraylib.L2N7_LINE])
    l3_lines = [xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE, xraylib.L3M4_LINE, xraylib.L3M5_LINE, xraylib.L3N1_LINE, xraylib.L3N2_LINE, xraylib.L3N3_LINE, xraylib.L3N4_LINE, xraylib.L3N5_LINE, xraylib.L3N6_LINE, xraylib.L3N7_LINE]
    xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
    const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
    xrf_struc = xrf_comp(energy_solar, counts_solar, i_angle, e_angle, at_no, weight, xrf_lines, const_xrf)
    # Generating XRF spectrum
    bin_size = energy[1] - energy[0]
    ebin_left = energy_mid - 0.5 * bin_size
    ebin_right = energy_mid + 0.5 * bin_size
    no_elements = (np.shape(xrf_lines.lineenergy))[0]
    n_lines = (np.shape(xrf_lines.lineenergy))[1]
    n_ebins = np.size(energy_mid)

    spectrum_xrf = np.zeros(n_ebins)
    for i in range(0, no_elements):
        for j in range(0, n_lines):
            line_energy = xrf_lines.lineenergy[i, j]
            bin_index = np.where((ebin_left <= line_energy) & (ebin_right >= line_energy))
            spectrum_xrf[bin_index] = spectrum_xrf[bin_index] + xrf_struc.total_xrf[i, j]
    
    # Defining the flux array required for XSPEC
    scaling_factor = (12.5 * 1e4 * 12.5 * (round(exposure / 8.0) + 1) * 1e4) / (exposure * 4 * np.pi * (altitude * 1e4) ** 2)
    spectrum_xrf_scaled = scaling_factor * spectrum_xrf
    
    for i in range(0, n_ebins):
        flux[i] = spectrum_xrf_scaled[i]
        
    return flux

# Create a Larch dataset group
dataset = Group(
    energy=energy[1:],  # Remove first point to match data length
    data=original_intensities,
    model=None
)

# Define the fitting model function for Larch
def fit_model(pars, data=None, model=None):
    """Model function for Larch fitting"""
    if data is None:
        data = dataset.data
    
    # Extract weights from parameters
    weights = [
        pars.weight_fe.value,
        pars.weight_ti.value,
        pars.weight_ca.value,
        pars.weight_si.value,
        pars.weight_al.value,
        pars.weight_mg.value,
        pars.weight_na.value,
        pars.weight_o.value
    ]
    
    # Calculate model using phy_model
    model = phy_model(energy, weights, original_intensities)
    
    # Return residuals array
    residuals = []
    for i in range(len(model)):
        if data[i] > 0:  # Only include non-zero data points
            residuals.append((model[i] - data[i]) / np.sqrt(data[i]))
    
    return np.array(residuals)

# Define parameters with constraints
params = param_group(
    weight_fe=Parameter(name='weight_fe', value=5.0, min=3, max=7),
    weight_ti=Parameter(name='weight_ti', value=1.0, min=0, max=0.3),
    weight_ca=Parameter(name='weight_ca', value=9.0, min=8, max=12),
    weight_si=Parameter(name='weight_si', value=21.0, min=19, max=22),
    weight_al=Parameter(name='weight_al', value=14.0, min=13, max=15),
    weight_mg=Parameter(name='weight_mg', value=5.0, min=4, max=6),
    weight_na=Parameter(name='weight_na', value=0.5, min=0, max=1),
    weight_o=Parameter(name='weight_o', value=44.5, min=40, max=50)  # Give oxygen some reasonable bounds
)

# params = param_group(
#     weight_fe=guess(5.0, min = 3, max =7),
#     weight_ti=guess(1.0, min = 0, max =0.3),
#     weight_o=guess(44.5, min = 40,max = 50),
#     weight_ca=guess(9.0, min = 8, max =12),
#     weight_si=guess(21.0, min = 19, max =22),
#     weight_al=guess(14.0, min = 13, max =15),
#     weight_mg=guess(5.0, min = 4, max =6),
#     weight_na=guess(0.5, min = 0, max =1)
# )
# Perform the fit
method = 'leastsq'
# options = {'maxiter': 10}
result = minimize(fit_model, params, args=(dataset.data, ), method=method,
                #   options=options
                ) 


print(result.params)
print(fit_report(result))
# Extract best-fit parameters
# best_params = [
#     result.params.weight_fe.value,
#     result.params.weight_ti.value,
#     result.params.weight_ca.value,
#     result.params.weight_si.value,
#     result.params.weight_al.value,
#     result.params.weight_mg.value,
#     result.params.weight_na.value,
#     result.params.weight_o.value
# ]

# # Normalize parameters to sum to 100
# sum_weights = sum(best_params)
# best_params = [w / sum_weights * 100 for w in best_params]

# # Print results
# elements = ["Fe", "Ti", "Ca", "Si", "Al", "Mg", "Na", "O"]
# print("\nFitted elemental compositions:")
# for element, weight in zip(elements, best_params):
#     print(f"{element}: {weight:.2f}%")

# # Calculate best fit model
# best_flux = phy_model(energy, best_params, original_intensities)

# # Plot results
# plt.figure(figsize=(12, 6))
# plt.plot(energy[1:], original_intensities, label='Original Data', alpha=0.7)
# plt.plot(energy[1:], best_flux, label='Best Fit', alpha=0.7)
# plt.xlabel('Energy (keV)')
# plt.ylabel('Intensity')
# plt.yscale('log')
# plt.xlim(0, 10)
# plt.legend()
# plt.title('XRF Spectrum Fitting Results')
# plt.grid(True, which='both', linestyle='--', alpha=0.3)
# plt.show()

# # Calculate fit statistics
# chi_square = result.chisqr
# dof = result.nfree
# reduced_chi_square = result.redchi

# print(f"\nFit Statistics:")
# print(f"Chi-square: {chi_square:.2f}")
# print(f"Degrees of freedom: {dof}")
# print(f"Reduced chi-square: {reduced_chi_square:.2f}")
# print(f"Fit success: {result.success}")
# print(f"Number of function evaluations: {result.nfev}")
# print(f"Number of variables: {result.nvarys}")
