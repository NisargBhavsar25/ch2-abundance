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
# from xspec import *
import xraylib
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V3 import xrf_comp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import numpy as np
import scipy.stats as stats
import optuna

import os
# Getting the static parameters for the local model
static_parameter_file = "static_par_localmodel.txt"
# check if the file exists
# if not os.pat`h.exists(static_parameter_file):
    # print("File does not exist")
    # exit()
fid = open(static_parameter_file,"r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])




# Defining the model function
def phy_model(energy, parameters, original_intensities):
    flux = np.zeros(np.size(energy) -1)
    # Defining proper energy axis    
    energy_mid = np.zeros(np.size(energy)-1)
    for i in np.arange(np.size(energy)-1):
        energy_mid[i] = 0.5*(energy[i+1] + energy[i])
        
    # Defining some input parameters required for x2abund xrf computation modules
    at_no = np.array([26,22,20,14,13,12,11,8])
    
    weight = list(parameters)
    
    i_angle = 90.0 - solar_zenith_angle
    e_angle = 90.0 - emiss_angle
    (energy_solar,tmp1_solar,counts_solar) = readcol(solar_file,format='F,F,F')
        
    # Computing the XRF line intensities
    k_lines = np.array([xraylib.KL1_LINE, xraylib.KL2_LINE, xraylib.KL3_LINE, xraylib.KM1_LINE, xraylib.KM2_LINE, xraylib.KM3_LINE, xraylib.KM4_LINE, xraylib.KM5_LINE])
    l1_lines = np.array([xraylib.L1L2_LINE, xraylib.L1L3_LINE, xraylib.L1M1_LINE, xraylib.L1M2_LINE, xraylib.L1M3_LINE, xraylib.L1M4_LINE, xraylib.L1M5_LINE, xraylib.L1N1_LINE, xraylib.L1N2_LINE, xraylib.L1N3_LINE, xraylib.L1N4_LINE, xraylib.L1N5_LINE, xraylib.L1N6_LINE, xraylib.L1N7_LINE])
    l2_lines = np.array([xraylib.L2L3_LINE, xraylib.L2M1_LINE, xraylib.L2M2_LINE, xraylib.L2M3_LINE, xraylib.L2M4_LINE, xraylib.L2M5_LINE, xraylib.L2N1_LINE, xraylib.L2N2_LINE, xraylib.L2N3_LINE, xraylib.L2N4_LINE, xraylib.L2N5_LINE, xraylib.L2N6_LINE, xraylib.L2N7_LINE])
    l3_lines = [xraylib.L3M1_LINE, xraylib.L3M2_LINE, xraylib.L3M3_LINE, xraylib.L3M4_LINE, xraylib.L3M5_LINE, xraylib.L3N1_LINE,xraylib.L3N2_LINE, xraylib.L3N3_LINE, xraylib.L3N4_LINE, xraylib.L3N5_LINE, xraylib.L3N6_LINE, xraylib.L3N7_LINE]
    xrf_lines = get_xrf_lines(at_no, xraylib.K_SHELL, k_lines, xraylib.L1_SHELL, l1_lines, xraylib.L2_SHELL, l2_lines, xraylib.L3_SHELL, l3_lines)
    const_xrf = get_constants_xrf(energy_solar, at_no, weight, xrf_lines)
    xrf_struc = xrf_comp(energy_solar,counts_solar,i_angle,e_angle,at_no,weight,xrf_lines,const_xrf)
    # Generating XRF spectrum
    bin_size = energy[1] - energy[0]
    ebin_left = energy_mid - 0.5*bin_size
    ebin_right = energy_mid + 0.5*bin_size
    no_elements = (np.shape(xrf_lines.lineenergy))[0]
    n_lines = (np.shape(xrf_lines.lineenergy))[1]
    n_ebins = np.size(energy_mid)

    spectrum_xrf = dblarr(n_ebins)
    for i in range(0, no_elements):
        for j in range(0, n_lines):
            line_energy = xrf_lines.lineenergy[i,j]
            bin_index = np.where((ebin_left <= line_energy) & (ebin_right >= line_energy))
            spectrum_xrf[bin_index] = spectrum_xrf[bin_index] + xrf_struc.total_xrf[i,j]
            # print(bin_index)
            # if bin_index[0] < 30:
            #     print(i, bin_index)
    
    # Defining the flux array required for XSPEC
    scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)
    spectrum_xrf_scaled = scaling_factor*spectrum_xrf
    
    for i in range(0, n_ebins):
        flux[i] = spectrum_xrf_scaled[i]
        
    return flux

import optuna

fits_filename = "test/ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits"


import astropy.io.fits as fits

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
print(energy)
# energy = np.zeros() #TODO: Define the energy array
# original_intensities = np.zeros() #TODO: Read from XSpec
# initial_guess = [5, 1, 9, 21, 14, 5, 0.5, 45]

def objective(trial):
    # weight = [trial.suggest_float(f"weight_{i}", -3, 3, step=0.1) for i in range(8)]
    step = 0.2
    weight_fe = trial.suggest_float("weight_fe", 3, 7, step=step)
    weight_ti = trial.suggest_float("weight_ti", 0, 0.3, step=step)
    weight_ca = trial.suggest_float("weight_ca", 8, 12, step=step)
    weight_si = trial.suggest_float("weight_si", 18, 22, step=step)
    weight_al = trial.suggest_float("weight_al", 13, 15, step=step)
    weight_mg = trial.suggest_float("weight_mg", 3, 9, step=step)
    weight_na = trial.suggest_float("weight_na", 0, 1, step=step)
    # weight_fe = trial.suggest_float("weight_fe", 0, 100, step=step)
    # weight_ti = trial.suggest_float("weight_ti", 0, 100, step=step)
    # weight_ca = trial.suggest_float("weight_ca", 0, 100, step=step)
    # weight_si = trial.suggest_float("weight_si", 0, 100, step=step)
    # weight_al = trial.suggest_float("weight_al", 0, 100, step=step)
    # weight_mg = trial.suggest_float("weight_mg", 0, 100, step=step)
    # weight_na = trial.suggest_float("weight_na", 0, 100, step=step)
    weight_o = max(0,100 - weight_fe - weight_ti - weight_ca - weight_si - weight_al - weight_mg - weight_na)
    # weight_o = trial.suggest_float("weight_o", -2, 2, step=0.1)
    
    weight = [weight_fe, weight_ti, weight_ca, weight_si, weight_al, weight_mg, weight_na, weight_o]
    
    
    # weight = [max(0, initial_guess[i] + weight[i]) for i in range(8)]

    total_weight = sum(weight)
    for i in range(8):
        # if i != 2:
        weight[i] = weight[i] / total_weight * 100
    
    # weight[3] = 21
    # weight_sum = sum(weight)
    # weight = [w / weight_sum * 100 for w in weight]
    
    flux = phy_model(energy, weight, original_intensities)
    flux[flux<1e-5] = 0
    chi2 = 0
    
    nonzeroidx = np.nonzero(flux)
    flux2 = flux[nonzeroidx]
    original_intensities2 = original_intensities[nonzeroidx]
    
    # for i in range(len(flux)):
    #     if original_intensities[i] != 0:
    #         chi2 += (flux[i] - original_intensities[i])**2 / (original_intensities[i])
    
    for i in range(len(flux2)):
        if original_intensities2[i] != 0:
            chi2 += (flux2[i] - original_intensities2[i])**2 / (original_intensities2[i])
    
    # chi2 = chi2/len(flux)        
        
    
    # for wi, wg in zip(initial_guess, weight):
    #     # print("ADDING", (wg-wi)**16)
    #     chi2 += (wg-wi)**2
    print("chi2", chi2)
    return chi2


# Define the Optuna study

study = optuna.create_study(direction="minimize", sampler=optuna.samplers.NSGAIIISampler())
# study.enqueue_trial({"weight_Fe": 5, "weight_Ti": 1, "weight_Ca": 9, "weight_Si": 21, "weight_Al": 14, "weight_Mg": 5, "weight_Na": 0.5, "weight_O": 45})
# Run the optimization
study.optimize(objective, n_trials=300)

# Get the best parameters
best_params = study.best_params

best_params = list(best_params.values())
best_params.append(100 - sum(best_params))
# add initial guess
# best_params = [max(0,initial_guess[i] + best_params[i]) for i in range(len(best_params))]
# make everything add up to 100
sum = sum(best_params)
best_params = [best_params[i]/sum*100 for i in range(len(best_params))]


elements = ["Fe", "Ti", "Ca", "Si", "Al", "Mg", "Na", "O"]
for i in range(len(best_params)):
    print(elements[i],": ", best_params[i])

mg_si_ratio = 28*best_params[5]/(24*best_params[3])
print("mg/si ratio", mg_si_ratio)

# get flux using best_params
best_flux = phy_model(energy, best_params, original_intensities)

# threshold to zero below 1e-3

# best_flux[best_flux < 1e-3] = 0

# initial_guess = [1.06, 1.11, 11.45, 7.36, 8.44, 0.63, 0.39, 30.75]
# best_flux = phy_model(energy, initial_guess, original_intensities)

# fit a curve through the best flux taking x, y values corresponding to non zero y
# non_zero_indices = np.where(best_flux != 0)
# fys = best_flux[non_zero_indices]

# fit a splined curve through the best flux
# best_flux = np.interp(energy[1:], energy[non_zero_indices], fys)




# chi2 = 0
# for i in range(len(best_flux)):
#     if original_intensities[i] != 0:
#         chi2 += (best_flux[i] - original_intensities[i])**2 / (original_intensities[i])

# print("chi2", chi2)


# plot both original and best flux in fig side by side
fig, ax1 = plt.subplots(1,1, figsize=(12, 6))

# Plot original and best flux as curves
ax1.plot(energy[1:], original_intensities, label="Original")
ax1.plot(energy[1:], best_flux, label="Best")
ax1.set_yscale('log')
ax1.set_xlabel("Energy (keV)")
ax1.set_ylabel("Intensity")
ax1.legend()
ax1.set_xlim([0, 10])  # Set x-axis limits
plt.show()

print(f"Best parameters: {best_params}")
