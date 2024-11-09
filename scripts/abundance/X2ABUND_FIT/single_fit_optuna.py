import numpy as np
import xraylib
from common_modules import *
from get_xrf_lines_V1 import get_xrf_lines
from get_constants_xrf_new_V2 import get_constants_xrf
from xrf_comp_new_V3 import xrf_comp
import matplotlib.pyplot as plt
import optuna
import astropy.io.fits as fits

# Constants
ELEMENT_TO_FIT = "Fe"  # Iron
PEAK_ALPHA_ENERGY = 6.40
PEAK_WIDTH = 0.5  # Width around peak to consider in keV

# Keep the static parameters
static_parameter_file = "static_par_localmodel.txt"
fid = open(static_parameter_file,"r")
finfo_full = fid.read()
finfo_split = finfo_full.split('\n')
solar_file = finfo_split[0]
solar_zenith_angle = float(finfo_split[1])
emiss_angle = float(finfo_split[2])
altitude = float(finfo_split[3])
exposure = float(finfo_split[4])

def phy_model(energy, parameters, original_intensities):
    flux = np.zeros(np.size(energy) -1)
    energy_mid = np.zeros(np.size(energy)-1)
    for i in np.arange(np.size(energy)-1):
        energy_mid[i] = 0.5*(energy[i+1] + energy[i])
        
    at_no = np.array([26, 22, 20, 14, 13, 12, 11, 8])
    # at_no = np.array([26])
    
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
            # energy_of_bin = energy[bin_index]
            # if spectrum_xrf[bin_index] > 1e-2:
                # print("ELEMENT", i+1, "LINE", j+1, "ENERGY", line_energy, "ENERGY OF BIN", energy_of_bin)
    
    scaling_factor = (12.5*1e4*12.5*(round(exposure/8.0)+1)*1e4)/(exposure*4*np.pi*(altitude*1e4)**2)
    spectrum_xrf_scaled = scaling_factor*spectrum_xrf
    
    for i in range(0, n_ebins):
        flux[i] = spectrum_xrf_scaled[i]
        
    return flux

def objective(trial):
    # Only optimize Fe weight, set others to 0
    weight_fe = trial.suggest_float("weight_fe", 0, 10, step=0.1)
    
    # Set other elements to 100-weiht/7
    other_w = (100 - weight_fe)/7
    # weight = [weight_fe, other_w, other_w, other_w, other_w, other_w, other_w, other_w]
    weight = [weight_fe, other_w, other_w, other_w, other_w, other_w, other_w, other_w]
    
    flux = phy_model(energy, weight, original_intensities)
    flux[flux<1e-5] = 0
    
    # Only consider the region around the peak
    peak_region = np.where((energy[1:] >= PEAK_ALPHA_ENERGY - PEAK_WIDTH) & 
                          (energy[1:] <= PEAK_ALPHA_ENERGY + PEAK_WIDTH))
    
    # peak_region = (energy[1:] >= PEAK_ALPHA_ENERGY - 20) & (energy[1:] <= PEAK_ALPHA_ENERGY + 20)
    
    flux_peak = flux[peak_region]
    original_peak = original_intensities[peak_region]
    
    # Calculate chi-square only for non-zero values in the peak region
    nonzero_mask = original_peak != 0
    chi2 = np.sum((flux_peak[nonzero_mask] - original_peak[nonzero_mask])**2 / 
                  original_peak[nonzero_mask])
    
    print(f"Trial with Fe weight: {weight_fe:.2f}, Chi2: {chi2:.2f}")
    return chi2

# Load and prepare data
fits_filename = "test/ch2_cla_l1_20210827T210316000_20210827T210332000_1024.fits"
with fits.open(fits_filename) as hdul:
    data = hdul[1].data.view(np.recarray)
    x, y = data.CHANNEL, data.COUNTS

original_intensities = []
for i in range(len(y)):
    original_intensities.append(y[i])
    original_intensities.append(y[i])
original_intensities = np.float64(original_intensities)

energy = np.arange(0, 2049, dtype=float)
energy *= 0.0135

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Get best results
best_params = study.best_params
best_fe_weight = best_params['weight_fe']
best_weights = [best_fe_weight, 0, 0, 0, 0, 0, 0, 100-best_fe_weight]

print(f"\nBest fit results:")
print(f"Fe concentration: {best_fe_weight:.2f}%")
print(f"O concentration: {100-best_fe_weight:.2f}%")

# Calculate and plot results
best_flux = phy_model(energy, best_weights, original_intensities)

# Plot results
fig, ax1 = plt.subplots(1, 1, figsize=(12, 10))

# Full spectrum plot
# Plot original and best flux as curves
ax1.plot(energy[1:], original_intensities, label="Original", alpha=0.7)
ax1.plot(energy[1:], best_flux, label="Fitted", alpha=0.7)



ax1.axvline(x=PEAK_ALPHA_ENERGY, color='black', linestyle='--', label=f"Peak at {PEAK_ALPHA_ENERGY} keV")
ax1.axvline(x=PEAK_ALPHA_ENERGY-PEAK_WIDTH, color='black', linestyle='--', alpha=0.5)
ax1.axvline(x=PEAK_ALPHA_ENERGY+PEAK_WIDTH, color='black', linestyle='--', alpha=0.5) 

ax1.set_yscale('log')
ax1.set_xlabel("Energy (keV)")
ax1.set_ylabel("Intensity")
ax1.set_title("Full Spectrum")
ax1.legend()
ax1.set_xlim([0, 10])
# Zoomed plot around the peak
# peak_mask = (energy[1:] >= PEAK_ALPHA_ENERGY - PEAK_WIDTH) & (energy[1:] <= PEAK_ALPHA_ENERGY + PEAK_WIDTH)
# ax2.plot(energy[1:][peak_mask], original_intensities[peak_mask], label="Original", alpha=0.7)
# ax2.plot(energy[1:][peak_mask], best_flux[peak_mask], label="Fitted", alpha=0.7)
# ax2.set_xlabel("Energy (keV)")
# ax2.set_ylabel("Intensity")
# ax2.set_title(f"Zoom around {ELEMENT_TO_FIT} Kα peak ({PEAK_ALPHA_ENERGY} keV)")
# ax2.legend()

plt.tight_layout()
plt.show()