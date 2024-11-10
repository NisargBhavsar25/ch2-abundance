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

elements=["fe", "ca", "si", "al", "mg", "na", "o"]
# elements=["fe","ti","ca","si","al","mg","na","o"]
element_range={"fe":(3,12),"ti":(0,0.3),"ca":(8,12),"si":(19,22),"mg":(4,6),"na":(0,20),"al":(5,20),"o":(0,40)}
# element_range={"fe":(0,50),"ti":(0,0.3),"ca":(0,50),"si":(0,50),"mg":(0,50),"na":(0,50),"al":(0,50)}

elem_bins={key:[] for key in elements}
# Defining the model function
centers=[]
bin_size=1
def phy_model(energy, parameters, original_intensities):
    global centers,bin_size
    flux = np.zeros(np.size(energy) -1)
    # print(len(energy),energy[0],energy[1])
    # Defining proper energy axis    
    energy_mid = np.zeros(np.size(energy)-1)
    for i in np.arange(np.size(energy)-1):
        energy_mid[i] = 0.5*(energy[i+1] + energy[i])
    centers=energy_mid
    # Defining some input parameters required for x2abund xrf computation modules
    at_no = np.array([26,20,14,13,12,11,8])
    # [26,22,20,14,13,12,11,8]
    
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
        elem_bins[elements[i]]=[]
        for j in range(0, n_lines):
            line_energy = xrf_lines.lineenergy[i,j]
            bin_index = np.where((ebin_left <= line_energy) & (ebin_right >= line_energy))
            spectrum_xrf[bin_index] = spectrum_xrf[bin_index] + xrf_struc.total_xrf[i,j]
            if(xrf_struc.total_xrf[i,j]!=0):
                elem_bins[elements[i]].append(bin_index[0])
                
            # print(bin_index)
            # if bin_index[0] < 30:
            #     print(i, bin_index)
    
    # Defining the flux array required for XSPEC
    scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)
    # scaling_factor=1
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
    weight_fe = trial.suggest_float("weight_fe", 0, 7, step=step)
    weight_ti = trial.suggest_float("weight_ti", 0, 0.3, step=step)
    weight_ca = trial.suggest_float("weight_ca", 4, 12, step=step)
    weight_si = trial.suggest_float("weight_si", 10, 22, step=step)
    weight_al = trial.suggest_float("weight_al", 8, 15, step=step)
    weight_mg = trial.suggest_float("weight_mg", 0, 9, step=step)
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

# study = optuna.create_study(direction="minimize", sampler=optuna.samplers.NSGAIIISampler())

studies = {el:optuna.create_study(direction="minimize", sampler=optuna.samplers.CmaEsSampler()) for el in elements if el!="o"}
# study.enqueue_trial({"weight_Fe": 5, "weight_Ti": 1, "weight_Ca": 9, "weight_Si": 21, "weight_Al": 14, "weight_Mg": 5, "weight_Na": 0.5, "weight_O": 45})
# Run the optimization
# study.optimize(objective, n_trials=50)
# Customize the optimization loop
n_trials = 50  # Define the number of trials
weights={}
default_sigma=0.8
sigmas={}
amps={}
# Define the Gaussian function
def gaussian(x, amplitude, std_dev, x_center):
    return amplitude * np.exp(-((x - x_center) ** 2) / (2 * std_dev ** 2))
def expand_list(values, before=50, after=50):
    expanded_list = []
    for i in range(len(values)):
        # Get the range of indices for 50 elements before and after
        start = max(0, i - before)
        end = min(len(values), i + after + 1)
        
        # Extend the expanded_list with values from the range
        expanded_list.extend(values[start:end])
    
    return expanded_list


def area_under_bin(x_bin, y_bin):
    """
    Calculate the area under a curve for a specific bin.

    Parameters:
    - x: array-like, x-coordinates
    - y: array-like, y-coordinates
    - bin_start: int, starting index of the bin in x
    - bin_end: int, ending index of the bin in x (inclusive)

    Returns:
    - area: float, the area under the curve in the specified bin
    """
    # Extract the relevant section of x and y for the specified bin

    
    # Calculate the area under the curve within this bin using the trapezoidal rule
    area = np.trapz(y_bin, x_bin)
    
    return area

for idx in range(n_trials):
    trials={}
    # Generate a new trial
    for key,study in studies.items():
        # print(key,study)
        trial = study.ask()
        trials[key]=trial
        try:
        # Evaluate the trial
            # Run the objective function with the trial's parameters
            step = 0.2
            # print(element_range[key][0],element_range[key][1])
            weight_el = trial.suggest_float(f"weight_{key}", element_range[key][0], element_range[key][1], step=step)
            weights[key]=weight_el
            # weight_o = max(0,100 - weight_fe - weight_ti - weight_ca - weight_si - weight_al - weight_mg - weight_na)
            # weight_o = trial.suggest_float("weight_o", -2, 2, step=0.1)
            sigmas[key]=trial.suggest_float(f"sigma_{key}", 0, default_sigma, step=0.001)
            # amps[key]=trial.suggest_float(f"amps_{key}", 0, 5, step=0.5)
            
            # weight = [weight_fe, weight_ti, weight_ca, weight_si, weight_al, weight_mg, weight_na, weight_o]
            
            
            # weight = [max(0, initial_guess[i] + weight[i]) for i in range(8)]
        except Exception as e:
            # If something goes wrong in evaluation, tell the study about the failure
            print(f"Trial failed with error: {e}")
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            trials[key]=None
    # for el in elements:
    weights["o"]=max(0,100 - sum(weights.values()))
    total_weight = sum(weights.values())
    weight_cal=weights
    # print(weight_cal)
    # print(weight_cal)
    areas={}
    normalized_area={}
    for i in elements:
        # if i != 2:
        if i not in weight_cal.keys():
            weight_cal[i]=0
        weight_cal[i] = (weight_cal[i] / total_weight) * 100
    # print("Energy",list(energy),weight_cal)
    flux = phy_model(energy, weight_cal.values(), original_intensities)
    for i in elements:
    
        key=i
        bin=elem_bins[key]
        bin=np.array(bin).flatten()
        bin=list(set(bin))
        # if key=="ti":
        #     print("Ti......\n\n",key,bin)
        ebin=sorted(list(set(expand_list(bin,10,10))))
        i0 = original_intensities[ebin]
        area=area_under_bin(ebin,i0)
        areas[i]=area

        # print("area:",area,key,bin)
    tot_area=sum(areas.values())
    print("Area",areas,"-----------------\n\n")
    normalized_area={key:value/tot_area for key,value in areas.items()}
    # print(weight_cal)
    flux[flux<1e-5] = 0
    chi2 = 0
    nonzeroidx = np.nonzero(flux)
    flux2 = flux[nonzeroidx]
    # print(trials)
    err=0
    for key,study in studies.items():
        bin=elem_bins[key]
        bin=np.array(bin).flatten()
        bin=list(set(bin))
        # print(key,bin)
        el_flux = flux[bin]
        nflux=np.zeros(shape=len(flux))
        ebin=sorted(list(set(expand_list(bin,10,10))))

        i0 = original_intensities[ebin]
        # print(i0,el_flux)
        # print(normalized_area[key])
        # print("bins:" ,bin,ebin)
        # print("ORIG: ",len(original_intensities),"Sigma: ",sigmas[key])
        for item in bin:

            # width=sigmas[key]/bin_size
            x=np.arange(max(item-50,0), min(item + 50,2048))
            x_center=centers[item]
            # print("Element ceenter",item,key)

            # print("Amp Multiplier",amps[key])
            y_values=gaussian(x, flux[item], sigmas[key]*10, item)
            nflux[x]+=y_values

        # print(ebin)
        
        # Plot the Gaussian and bin points
        plt.plot(range(0,2048), nflux, label='Gaussian')
        # for bin_range, (x_bin, y_bin) in bin_points.items():
        # plt.scatter(x, y_bin, s=10, label=f'Bin {bin_range}')
        plt.xlabel('x')
        plt.ylabel('Gaussian(x)')
        plt.legend()
        plt.savefig(f"images/{key}_{item}.jpg")
        plt.clf()  # Clear the current figure

        chi2=0
        print(f"Final {key}", normalized_area[key],sum(nflux[ebin]))

        
        # print(len(i0),len(el_flux))
        for i in range(len(nflux[ebin])):
            if i0[i] != 0:
                # chi2 += (nflux[i] - i0[i])**2 / (i0[i])
                chi2+=nflux[i]
        chi2=(chi2-normalized_area[key])**2
        if trials[key] is not None:
            study.tell(trials[key],chi2)
        if(key=="ca"):
            print("Element: ",key,"err: ",chi2)
        err+=chi2
    # weight[3] = 21
    # weight_sum = sum(weight)
    # weight = [w / weight_sum * 100 for w in weight]
    print(f"Error in {idx}:",err)
    
    
    # Tell the study that this trial has succeeded and report the value
    


# Get the best parameters
best_params=[]
vars={}
for key,study in studies.items():
    best_params.append(list(study.best_params.values())[0])
    vars[key]=list(study.best_params.values())[1]

# best_params = study.best_params

# best_params = list(best_params.values())
best_params.append(100 - sum(best_params))
# add initial guess
# best_params = [max(0,initial_guess[i] + best_params[i]) for i in range(len(best_params))]
# make everything add up to 100
sum = sum(best_params)
best_params = [best_params[i]/sum*100 for i in range(len(best_params))]
def merge_close_values(arrays, threshold=10):
    # Flatten the arrays and sort the values
    values = np.sort(np.concatenate(arrays))
    
    # Initialize a list to hold merged groups
    merged_values = []
    current_group = [values[0]]
    
    # Iterate through sorted values and group close values
    for i in range(1, len(values)):
        if values[i] - values[i - 1] <= threshold:
            # If the current value is close to the previous, add it to the current group
            current_group.append(values[i])
        else:
            # Otherwise, finalize the current group and start a new one
            merged_values.append(np.mean(current_group))  # Use mean for merged value
            current_group = [values[i]]
    
    # Append the final group
    if current_group:
        merged_values.append(np.mean(current_group))
    
    return merged_values

labels={}
for key in elements:
    if len(elem_bins[key])==0:
        continue
    labels[key]=merge_close_values(elem_bins[key])
    print(f"{key}:", labels[key])

# get flux using best_params
best_flux = phy_model(energy, best_params, original_intensities)
 # width=sigmas[key]/bin_size
bflux=np.zeros(len(best_flux))
tflux=np.zeros(len(original_intensities))
for key in elements:
    if key=="o":
        continue
    bins=np.array(elem_bins[key]).flatten()
    for item in bins:
        x=np.arange(max(item-50,0), min(item + 50,2048))
        x_center=centers[item]
        # print("Element ceenter",item,key)
        # print("Amp Multiplier",amps[key])
        y_values=gaussian(x, best_flux[item], vars[key], item)
        bflux[x]+=y_values
        # tflux[x]+=original_intensities[x]
# bflux[bflux<0.00001]=0
# best_flux=bflux
print(best_flux)
elements = ["Fe", "Ca", "Si", "Al", "Mg", "Na", "O"]

for i in range(len(best_params)):

    print(elements[i],": ", best_params[i])

mg_si_ratio = 28*best_params[4]/(24*best_params[2])
print("mg/si ratio", mg_si_ratio)

# threshold to zero below 1e-3

# best_flux[best_flux < 1e-3] = 0

# initial_guess = [1.06, 1.11, 11.45, 7.36, 8.44, 0.63, 0.39, 30.75]
# best_flux = phy_model(energy, initial_guess, original_intensities)

# fit a curve through the best flux taking x, y values corresponding to non zero y
non_zero_indices = np.where(best_flux != 0)
fys = best_flux[non_zero_indices]

# fit a splined curve through the best flux
best_flux = np.interp(energy[1:], energy[non_zero_indices], fys)

# non_zero_indices = np.where(tflux != 0)
# fys = tflux[non_zero_indices]

# fit a splined curve through the best flux
# tflux = np.interp(tflux, tflux[non_zero_indices], fys)




# chi2 = 0
# for i in range(len(best_flux)):
#     if original_intensities[i] != 0:
#         chi2 += (best_flux[i] - original_intensities[i])**2 / (original_intensities[i])

# print("chi2", chi2)


# plot both original and best flux in fig side by side
fig, ax1 = plt.subplots(1,1, figsize=(12, 6))

# Add element labels at specific x locations from `elem_bins`
for elem, x_position in labels.items():
    # y_position = np.interp(x_position, energy[1:], best_flux)  # Interpolate y-value at x_position
    for i in range(len(x_position)):
        # print(elem,x_position[i],y_position[i])
        ax1.text(x_position[i]*10/2048, i*100 * 1.2, f"{elem}_{i}", ha='center', color="black")  # Adjust y offset if needed
print(len(energy[1:]),len(tflux))
# Plot original and best flux as curves
ax1.plot(energy[1:], original_intensities, label="Original")
# ax1.plot(energy[1:], tflux, label="Original")
ax1.plot(energy[1:], best_flux, label="Best")
ax1.set_yscale('log')
ax1.set_xlabel("Energy (keV)")
ax1.set_ylabel("Intensity")
ax1.legend()
ax1.set_xlim([0, 10])  # Set x-axis limits
plt.savefig("images/final.jpg")

print(f"Best parameters: {best_params}")
