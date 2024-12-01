import os
import numpy as np
import pandas as pd
from astropy.io import fits
from jax import random, jit
from jaxspec import Spectrum, Model, Fit
from tqdm import tqdm

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct paths using the current directory
base_path = os.path.join(current_dir, 'test')  # Path to 'test' directory
data_folder = os.path.join(current_dir, 'test')  # Path to 'test' directory
solar_path = os.path.join(current_dir, '../../flare_ops/scripts/spectrum')  # Path to 'flare_ops/scripts/spectrum'
ignore_erange = ["0.9", "4.2"]
ignore_string = f'0.0-{ignore_erange[0]} {ignore_erange[1]}-**'
results = []

# Define atomic numbers and corresponding model parameter indices
elements_atomic_numbers = np.array([26, 22, 20, 14, 13, 12, 11, 8])  # Fe, Ti, Ca, Si, Al, Mg, Na, O
model_param_indices = [3, 4, 5, 7, 8, 9, 10, 11]  # Adjusted according to your model parameters

# Define a Gaussian model function
def gaussian_model(x, params):
    """Gaussian model: A * exp(-((x - mu) ** 2) / (2 * sigma ** 2))"""
    A = params[0]  # Amplitude
    mu = params[1]  # Mean
    sigma = params[2]  # Standard deviation
    return A * jnp.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Create a model object
model = Model(gaussian_model, n_params=3)  # Gaussian model with 3 parameters

def process_row(row):
    tst = row['class_file_name']
    dt = tst[:8]
    flare_type = row['flare_class_with_bg'][0]

    # Define file paths
    class_l1_data = os.path.join(data_folder, f'ch2_cla_l1_{tst}.fits')
    bkg_file = os.path.join(base_path, 'ch2_cla_l1_20210826T220355000_20210826T223335000_1024.fits')
    scatter_atable = os.path.join(solar_path, f"t{dt}.fits")
    
    # Open FITS file and extract header information
    with fits.open(class_l1_data) as hdu_data:
        hdu_header = hdu_data[1].header
        solar_zenith_angle = hdu_header['SOLARANG']
        emiss_angle = hdu_header['EMISNANG']
        sat_alt = hdu_header['SAT_ALT']
        tint = hdu_header['EXPOSURE']
    
    # Load spectrum data
    spec_data = Spectrum(class_l1_data)  # Load the spectrum using jaxspec
    spec_data.background = bkg_file
    spec_data.ignore(ignore_string)

    # Prepare data for fitting
    energy = spec_data.energies  # Get energy values from the spectrum
    counts = spec_data.values  # Get counts from the spectrum

    # Set initial parameters for the Gaussian model
    initial_params = jnp.array([1.0, np.mean(energy), 1.0])  # Example initial parameters: A=1.0, mu=mean energy, sigma=1.0
    model.set_parameters(initial_params)  # Set initial parameters for the model

    # Fit the model to the data
    fit = Fit(model, energy, counts)
    fit.perform()  # Perform the fitting

    # Collect results
    result = {
        'tst': tst,
        'dt': dt,
        'flare_type': flare_type,
        'solar_zenith_angle': solar_zenith_angle,
        'emiss_angle': emiss_angle,
        'sat_alt': sat_alt,
        'tint': tint,
        'fit_results': fit.parameters.tolist(),
        'chi_squared': fit.chi_squared,
    }
    return result

# Main processing
if __name__ == '__main__':
    catalog = pd.read_parquet(os.path.join(data_folder, "test.parquet"))
    catalog = catalog[catalog["flare_class_with_bg"].str.startswith("X") | 
                      catalog["flare_class_with_bg"].str.startswith("M") | 
                      catalog["flare_class_with_bg"].str.startswith("C")]

    for index, row in tqdm(catalog.iterrows(), total=len(catalog)):
        result = process_row(row)
        results.append(result)

    # Save results to CSV
    df_results = pd.DataFrame(results)
    df_results.to_csv('abundances_results_jaxspec.csv', index=False)
    print("Saved results to abundances_results_jaxspec.csv")