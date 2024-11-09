import os
from astropy.io import fits
import numpy as np

def combine_first_12_fits(folder_path, output_fits_file, method="stack"):
    """
    Combines the first 12 FITS files from a specified folder into one FITS file.

    Parameters:
    folder_path (str): Path to the folder containing FITS files.
    output_fits_file (str): Name of the output combined FITS file.
    method (str): Combination method - 'stack' for stacking data or 'extension' for multi-extension.
    """
    # Get list of all .fits files in the folder, and limit to first 12
    file_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".fits")]
    file_list = file_list[:12]  # Take only the first 12 files

    if len(file_list) < 12:
        print("Warning: Less than 12 FITS files found in the folder.")

    if method == "stack":
        # Stacking Data (Row-wise)
        all_energy = []
        all_counts = []

        # Loop through each file, read data, and append to lists
        for file_name in file_list:
            with fits.open(file_name) as hdul:
                data = hdul[1].data  # Assuming data is in the first extension (index 1)
                all_energy.extend(data['CHANNEL'])
                all_counts.extend(data['COUNTS'])

        # Convert lists to numpy arrays
        all_energy = np.array(all_energy)
        all_counts = np.array(all_counts)
        # Create new FITS columns from the combined data
        col1 = fits.Column(name='CHANNEL', array=all_energy, format='E', unit='keV')
        col2 = fits.Column(name='COUNTS', array=all_counts, format='E', unit='counts')

        # Create a new FITS table with the combined columns
        hdu = fits.BinTableHDU.from_columns([col1, col2])

        # Write to the output FITS file
        hdu.writeto(output_fits_file, overwrite=True)
        print(f"Stacked data FITS file '{output_fits_file}' created successfully.")

    elif method == "extension":
        # Merging as Separate Extensions (Multi-extension FITS)
        hdulist = [fits.PrimaryHDU()]  # Start with a primary HDU

        # Loop through each file and add its data as a new extension
        for i, file_name in enumerate(file_list):
            with fits.open(file_name) as hdul:
                # Add the data (first extension) from each FITS file as a new HDU
                new_hdu = fits.BinTableHDU(data=hdul[1].data, name=f"SPECTRUM_{i+1}")
                hdulist.append(new_hdu)

        # Create an HDU list and write to the output file
        hdul = fits.HDUList(hdulist)
        hdul.writeto(output_fits_file, overwrite=True)
        print(f"Multi-extension FITS file '{output_fits_file}' created successfully.")
    else:
        print("Invalid method specified. Use 'stack' or 'extension'.")

# Example usage:
folder_path = "/home/manasj/isro/code/30"  # Replace with your folder path
output_fits_file = "/home/manasj/isro/code/combined_fits.fits"  # Desired output file name
combine_first_12_fits(folder_path, output_fits_file, method="extension")  # Use "extension" for multi-extension