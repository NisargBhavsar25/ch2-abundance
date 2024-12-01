from astropy.io import fits

"""
This script converts a FITS file to a PHA file.
"""

def convert_to_pha(input_fits, output_pha, txtfile):
    # Read the input FITS file
    with fits.open(input_fits) as hdul:
        # Get the spectrum data from HDU 1 (which is index 1)
        spectrum_data = hdul[1].data
        orig_header = hdul[1].header
        
        # Create a new header with PHA required keywords
        new_header = fits.Header()
        
        # Copy all original header keywords
        for key in orig_header:
            new_header[key] = orig_header[key]
            
        # Add required PHA headers if they don't exist
        pha_headers = {
            'HDUCLASS': 'OGIP',
            'HDUCLAS1': 'SPECTRUM',
            'TELESCOP': 'UNKNOWN',  # Replace with actual telescope name if known
            'INSTRUME': 'UNKNOWN',  # Replace with actual instrument name if known
            'EXPOSURE': 1.0,        # Replace with actual exposure time if known
            'BACKFILE': 'NONE',
            'RESPFILE': 'NONE',
            'ANCRFILE': 'NONE'
        }
        
        for key, value in pha_headers.items():
            if key not in new_header:
                new_header[key] = value
        
        # Create primary HDU (empty)
        primary_hdu = fits.PrimaryHDU()
        
        # Create spectrum HDU
        spectrum_hdu = fits.BinTableHDU(data=spectrum_data, header=new_header)
        
        # Create new FITS file with both HDUs
        hdul_new = fits.HDUList([primary_hdu, spectrum_hdu])
        
        # Write to output file
        hdul_new.writeto(output_pha, overwrite=True)
        
        # Append the output file name to the text file
        with open(txtfile, 'a') as file:
            file.write(f'{output_pha}\n')

# Usage example
"""input_fits = "/root/code/ch2-abundance/scripts/abundance/test/ch2_cla_l1_20200529T104516257_20200529T104524257.fits"    # Replace with your input FITS filename
output_pha = "output2.pha"         # Replace with desired output filename

convert_to_pha(input_fits, output_pha)"""
