from multiprocessing import Pool
import os
import numpy as np
import pandas as pd
from astropy.io import fits
import pickle
import time
from tqdm import tqdm
from fits_utils import convert_to_1024_channels
import gpufit
from xspec import *

# Load solar_dict and catalog outside of the function to avoid reloading in each process
solar_dict = pickle.load(open("solar.pkl", "rb"))
catalog = pd.read_parquet("test.parquet")
catalog = catalog[catalog["flare_class_with_bg"].str.startswith("X") | 
                  catalog["flare_class_with_bg"].str.startswith("M") | 
                  catalog["flare_class_with_bg"].str.startswith("C")]

# Constants
base_path = os.path.join(os.path.dirname(__file__), 'test')
data_folder = os.path.join(os.path.dirname(__file__), '../test/')
solar_path = os.path.join(os.path.dirname(__file__), '../../flare_ops/scripts/spectrum')
ignore_erange = ["0.9", "4.2"]
ignore_string = f'0.0-{ignore_erange[0]} {ignore_erange[1]}-**'

def process_row(row):
    # Extract necessary variables from the row
    tst = row['class_file_name']
    dt = tst[:8]
    flare_type = row['flare_class_with_bg'][0]
    
    # Check flare_type in solar_dict and get relevant dates list
    if flare_type in solar_dict:
        date_list = solar_dict[flare_type]
        selected_dt = dt if dt in date_list else date_list[0]
    else:
        print(f"Flare type {flare_type} not found in solar_dict. Skipping.")
        return None  # Skip this row

    # Define file paths
    class_l1_data = os.path.join(data_folder, f'ch2_cla_l1_{tst}.fits')
    bkg_file = os.path.join(base_path, 'ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits')
    scatter_atable = os.path.join(solar_path, "table", f"t{selected_dt}.fits")
    
    # Open FITS file and extract header information
    hdu_data = fits.open(class_l1_data)
    hdu_header = hdu_data[1].header
    solar_zenith_angle = hdu_header['SOLARANG']
    emiss_angle = hdu_header['EMISNANG']
    sat_alt = hdu_header['SAT_ALT']
    tint = hdu_header['EXPOSURE']
    hdu_data.close()

    # Load spectrum data
    spec_data = Spectrum(class_l1_data)  # Assuming you have a way to load this
    data_countspersec = spec_data.values  # Replace with actual data extraction
    data_energy = spec_data.energies  # Replace with actual energy extraction

    # Prepare data for gpufit
    model = "gauss"  # Example model, replace with your actual model
    num_parameters = 3  # Number of parameters for the model
    num_data_points = len(data_countspersec)

    # Initial parameter guesses
    initial_params = np.array([1.0, 1.0, 1.0], dtype=np.float32)  # Example initial parameters
    fit_results = np.zeros((num_parameters,), dtype=np.float32)
    fit_errors = np.zeros((num_parameters,), dtype=np.float32)
    chi_squared = np.zeros((1,), dtype=np.float32)

    # Call gpufit
    gpufit.gpufit(data_countspersec, num_data_points, model, initial_params, num_parameters, fit_results, fit_errors, chi_squared)

    # Collect results
    result = {
        'tst': tst,
        'dt': dt,
        'flare_type': flare_type,
        'solar_zenith_angle': solar_zenith_angle,
        'emiss_angle': emiss_angle,
        'sat_alt': sat_alt,
        'tint': tint,
        'fit_results': fit_results.tolist(),
        'fit_errors': fit_errors.tolist(),
        'chi_squared': chi_squared[0],
    }
    return result

# Main processing
if __name__ == '__main__':
    with Pool(processes=os.cpu_count()) as pool:
        results = pool.map(process_row, [row for index, row in catalog.iterrows()])

    # Filter out None results
    results = [result for result in results if result is not None]

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('abundances_results_gpufit.csv', index=False)
    print("Saved results to abundances_results_gpufit.csv")