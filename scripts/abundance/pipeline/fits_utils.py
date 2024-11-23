from astropy.io import fits
import os
import numpy as np

from astropy.io import fits
import numpy as np

def convert_to_1024_channels(input_file,output_file):
    """
    Converts a FITS file with 2048 channels to one with 1024 channels
    by aggregating every two adjacent channels.
    
    Parameters:
        input_file (str): Path to the input FITS file.
        output_file (str): Path to the output FITS file with 1024 channels.
    """
    # Open the input FITS file
    with fits.open(input_file) as hdul:
           # Extract the headers and data from the first extension (HDU 1)
        hdu1 = hdul[1]
        header = hdu1.header
        data = hdu1.data

        # Assuming this is within a function where the header is being modified
        current_dir = os.path.dirname(os.path.abspath(__file__))  # Get the absolute path of the current directory
        test_dir = os.path.join(current_dir, 'test')  # Construct the path to the 'test' directory

        # Update RESPFILE and ANCRFILE values
        header['RESPFILE'] = os.path.join(test_dir, 'class_rmf_v1.rmf')
        header['ANCRFILE'] = os.path.join(test_dir, 'class_arf_v1.arf')
        print(header['RESPFILE'])
        header['DETCHANS'] = 1024
        header['TLMIN1'] = 0
        header['TLMAX1'] = 1023
        header['TFORM2'] = '1D'
        header['NAXIS1'] = 10
        header['GAIN'] = 27
        # Reduce the data to 1024 channels by summing every two adjacent counts
        new_counts = np.add.reduceat(data['COUNTS'], np.arange(0, len(data['COUNTS']), 2))
        new_channels = np.arange(0, 1024)  # New channel numbers from 1 to 1024
        
        # Create new FITS columns for the reduced data
        col1 = fits.Column(name='CHANNEL', array=new_channels, format='1I')
        col2 = fits.Column(name='COUNTS', array=new_counts, format='1E', unit='count')
        new_cols = fits.ColDefs([col1, col2])

        # Create a new HDU with the same header and updated data
        new_hdu = fits.BinTableHDU.from_columns(new_cols, header=header)

        # Replace the old HDU with the new one in a copy of the HDUL
        hdul[1] = new_hdu
        
        # Save the result to the output file
        hdul.writeto(output_file, overwrite=True)

        # Save the result to the output file
