import os
import numpy as np
import pandas as pd
from xspec import *
from astropy.io import fits
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
# from define_xrf_localmodel import loadm
import pickle
import time

from tqdm import tqdm
from fits_utils import convert_to_1024_channels
from define_xrf_localmodel import xrf_localmodel,xrf_localmodel_ParInfo,finfo_full,loadm
# Load models and data
# loadm()
# # Creating the local model in PyXspec
# AllModels.addPyMod(xrf_localmodel, xrf_localmodel_ParInfo, 'add')
# def loadm():
#     print("loading models...")
#     AllModels.addPyMod(xrf_localmodel, xrf_localmodel_ParInfo, 'add')
#     return  


solar_dict = pickle.load(open("solar.pkl", "rb"))
# catalog = pd.read_parquet("merged_data.parquet")
catalog = pd.read_parquet("test.parquet")
catalog=catalog[catalog["flare_class_with_bg"].str.startswith("X")|catalog["flare_class_with_bg"].str.startswith("M")|catalog["flare_class_with_bg"].str.startswith("C")]
# Constants
base_path = '/home/heasoft/ch2-abundance/scripts/abundance/pipeline/test'
data_folder = "/home/heasoft/ch2-abundance/scripts/abundance/test/"
solar_path = "/home/heasoft/ch2-abundance/scripts/flare_ops/scripts/spectrum"
ignore_erange = ["0.9", "4.2"]
ignore_string = f'0.0-{ignore_erange[0]} {ignore_erange[1]}-**'
results=[]
# Define atomic numbers and corresponding model parameter indices
elements_atomic_numbers = np.array([26, 22, 20, 14, 13, 12, 11, 8])  # Fe, Ti, Ca, Si, Al, Mg, Na, O
model_param_indices = [3, 4, 5, 7, 8, 9, 10, 11]  # Adjusted according to your model parameters

# Process counter
counter = 0

# Iterate over each row in catalog
for index, row in tqdm(catalog.iterrows(), total=len(catalog)):
    start_time = time.perf_counter()
    tst = row['class_file_name']  # Obtain tst from class_file_name
    dt = tst[:8]  # Extract date as dt from tst
    flare_type = row['flare_class_with_bg'][0]  # Get the flare_type from the catalog
    
    # Check flare_type in solar_dict and get relevant dates list
    if flare_type in solar_dict:
        date_list = solar_dict[flare_type]
        selected_dt = dt if dt in date_list else date_list[0]  # Use dt if available; otherwise, use the first date in list
    else:
        print(f"Flare type {flare_type} not found in solar_dict. Skipping.")
        continue
    
    # Define file paths
    class_l1_data = os.path.join(data_folder, f'ch2_cla_l1_{tst}.fits')
    bkg_file = os.path.join(base_path, 'ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits')
    # bkg_file = "/home/heasoft/ch2-abundance/scripts/abundance/pipeline/BKG/ch2_cla_l1_20200529T104508257_20200529T104516257.fits"
    # bkg_root="/home/heasoft/ch2-abundance/scripts/abundance/pipeline/BKG/modified"
    scatter_atable = os.path.join(solar_path, "table", f"t{selected_dt}.fits")
    solar_model_file = os.path.join(solar_path, f'ch2_xsm_{selected_dt}_l1_vvapec.txt')
    
    static_par_file = os.path.join(base_path, '../static_par_localmodel_{tst}.txt')
    xspec_log_file = os.path.join(base_path, f'log_x2abund_{tst}.txt')
    xspec_xcm_file = os.path.join(base_path, f'xcm_x2abund_{tst}.xcm')
    plot_file = os.path.join(base_path, f'plots_x2abund_{tst}.pdf')
    print(class_l1_data,bkg_file,scatter_atable,solar_model_file)
    # continue    # Open FITS file and extract header information
    hdu_data = fits.open(class_l1_data)
    hdu_header = hdu_data[1].header
    solar_zenith_angle = hdu_header['SOLARANG']
    emiss_angle = hdu_header['EMISNANG']
    sat_alt = hdu_header['SAT_ALT']
    tint = hdu_header['EXPOSURE']
    with open(static_par_file,"w") as f:
        f.write(str(solar_model_file)+"\n")
        f.write(str(solar_zenith_angle)+"\n")
        f.write(str(emiss_angle)+"\n")
        f.write(str(sat_alt)+"\n")
        f.write(str(tint)+"\n")
    hdu_data.close()

    # solar_file =
    # solar_zenith_angle = float(finfo_split[1])
    # emiss_angle = float(finfo_split[2])
    # altitude = float(finfo_split[3])
    # exposure = float(finfo_split[4])
    

    tmp_class_l1_data="/home/heasoft/ch2-abundance/scripts/abundance/test/converted/"+f'ch2_cla_l1_{tst}.fits'
    # if not os.path.exists(tmp_class_l1_data):
    convert_to_1024_channels(class_l1_data,tmp_class_l1_data)
    # tmp_class_l1_data=class_l1_data
    # Initialize PyXspec
    Xset.openLog(xspec_log_file)
    Xset.allowNewAttributes = True
    Xset.parallel.xrf_localmodel = 8
    AllData.clear()
    AllModels.clear()
    
    # Load spectrum and background data
    # tmp_class_l1_data="/home/heasoft/ch2-abundance/scripts/abundance/pipeline/test/ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits"
    spec_data = Spectrum(tmp_class_l1_data)
    spec_data.background = bkg_file
    spec_data.ignore(ignore_string)

    loadm(static_par_file)
    # Set response gain
    spec_data.response.gain.slope = '1.0043000'
    spec_data.response.gain.offset = '0.0316000'
    spec_data.response.gain.slope.frozen = True
    spec_data.response.gain.offset.frozen = True
    # Define the model and fit
    full_model = f'atable{{{scatter_atable}}} + xrf_localmodel'
    mo = Model(full_model)
    mo(10).values = "45.0"
    mo(10).frozen = True
    mo(1).frozen = True
    mo(6).link = '100 - (3+4+5+7+8+9+10)'

    # Parameter 3 (Fe)
    mo(3).values = [5.0, 0.1, 0.0, 3.0, 13.0, 13.0]  # initial, delta, min, bot, top, max

    # Parameter 4 (Ti)
    mo(4).values = [1.0, 0.01, 0.0, 0.0, 3, 3]  # initial, delta, min, bot, top, max

    # Parameter 5 (Ca)
    mo(5).values = [9.0, 0.1, 6.0, 8.0, 12.0, 12.0]  # initial, delta, min, bot, top, max

    # Parameter 6 (Si)
    mo(6).values = [21.0, 0.1, 18.0, 18.0, 23.0, 23.0]  # initial, delta, min, bot, top, max

    # Parameter 7 (Al)
    mo(7).values = [14.0, 0.1, 13.0, 13.0, 15.0, 15.0]  # initial, delta, min, bot, top, max

    # Parameter 8 (Mg)
    mo(8).values = [5.0, 0.1, 3.0, 3.0, 9.0, 9.0]  # initial, delta, min, bot, top, max

    # Parameter 9 (Na)
    mo(9).values = [0.5, 0.01, 0.0, 0.0, 1.0, 1.0]  # initial, delta, min, bot, top, max

    # weight_fe = trial.suggest_float("weight_fe", 3, 7, step=step)
    # weight_ti = trial.suggest_float("weight_ti", 0, 0.3, step=step)
    # weight_ca = trial.suggest_float("weight_ca", 8, 12, step=step)
    # weight_si = trial.suggest_float("weight_si", 18, 22, step=step)
    # weight_al = trial.suggest_float("weight_al", 13, 15, step=step)
    # weight_mg = trial.suggest_float("weight_mg", 3, 9, step=step)
    # weight_na = trial.suggest_float("weight_na", 0, 1, step=step)


    Fit.nIterations = 5
    print("Fitting now...\n\n")
    start_fit = time.perf_counter()
    Fit.delta = 0.01
    Fit.query = "no"
    Fit.perform()
    end_fit = time.perf_counter()
    print("Time for fit:",end_fit-start_fit)

    # Extract chi-squared value
    chi_sq = Fit.statistic
    degrees_of_freedom = Fit.dof
    red_chi_sq = chi_sq / degrees_of_freedom if degrees_of_freedom != 0 else np.nan

    # Extract abundances and errors
    abundances = {}
    for elem_atomic_num, param_index in zip(elements_atomic_numbers, model_param_indices):
        param = mo(param_index)
        abundances[f'Element_{elem_atomic_num}'] = param.values[0]
        abundances[f'Element_{elem_atomic_num}_err'] = param.sigma

    # Collect results
    result = {
        'tst': tst,
        'dt': dt,
        'flare_type': flare_type,
        'chi_sq': chi_sq,
        'red_chi_sq': red_chi_sq,
        'solar_zenith_angle': solar_zenith_angle,
        'emiss_angle': emiss_angle,
        'sat_alt': sat_alt,
        'tint': tint,
        **abundances  # Merge abundances dictionary into result
    }
    results.append(result)
    counter += 1


    # Plotting the fit outputs
    # pdf_plot = PdfPages(plot_file)
    data_energy_tmp = spec_data.energies
    data_countspersec = spec_data.values
    data_background = spec_data.background.values
    data_backrem = np.array(data_countspersec) - np.array(data_background)
    data_energy = np.array([(energy[0]) for energy in data_energy_tmp])

    folded_flux = mo.folded(1)
    delchi = (data_backrem - folded_flux) / np.sqrt(folded_flux)

    # fig, (axis1, axis2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 1]})
    # fig.suptitle('Data Model Comparison')

    # Plot data and model
    # axis1.plot(data_energy, data_backrem, label='Data')
    # axis1.plot(data_energy, folded_flux, label='Model')
    # axis1.set_yscale("log")
    # axis1.set_xlabel('Energy (keV)')
    # axis1.set_ylabel('Counts/s')
    # axis1.set_xlim([float(ignore_erange[0]), float(ignore_erange[1])])
    # axis1.legend()

    # Plot residuals
    # axis2.plot(data_energy, delchi)
    # axis2.set_xlabel('Energy (keV)')
    # axis2.set_ylabel('Delchi')
    # axis2.set_xlim([float(ignore_erange[0]), float(ignore_erange[1])])

    # Save and close the plot
    # pdf_plot.savefig(fig, bbox_inches='tight', dpi=300)
    # pdf_plot.close()
    # plt.close(fig)

    # Save Xspec output and close log
    # Xset.save(xspec_xcm_file)
    # Xset.closeLog()

    print(f"Processed file {tst} with selected date {selected_dt}")
    finfo_full=None
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
    # exit()
     # Save to CSV every 1000 files
    if counter % 1000 == 0:
        df_results = pd.DataFrame(results)
        csv_filename = f'abundances_results_{counter // 1000}_{Fit.nIterations}.csv'
        df_results.to_csv(csv_filename, index=False)
        print(f"Saved results to {csv_filename}")
        # Clear results to save memory
        results.clear()
# After processing all files, save any remaining results
if results:
    df_results = pd.DataFrame(results)
    csv_filename = f'abundances_results_final.csv'
    df_results.to_csv(csv_filename, index=False)
    print(f"Saved final results to {csv_filename}")