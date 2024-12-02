from astropy.io import fits
import numpy as np
#import matplotlib.pyplot as plt

"""
This script groups the counts in a FITS file into bins of a given factor.
"""

# Load the FITS file
fits_file = '/root/code/ch2-abundance/scripts/abundance/test/ch2_cla_l1_20200529T104548257_20200529T104556257.fits'

# Open the FITS file
with fits.open(fits_file) as hdul:
    # Extract the data from the SPECTRUM HDU (HDU 2)
    spectrum_data = hdul[1].data
    
    # Extract the 'CHANNEL' and 'COUNTS' columns
    channels = spectrum_data['CHANNEL']
    counts = spectrum_data['COUNTS']

# Display the first few rows to check the data
print(channels[84:86], counts[84:86])

# Define the binning factor
bin_factor = 2

# Group the counts into bins
binned_counts = np.add.reduceat(counts, np.arange(0, len(counts), bin_factor))

# Adjust the channels to match the new bins (taking the first channel in each bin)
binned_channels = channels[::bin_factor]

binned_channels = binned_channels/2
print(len(binned_channels), len(binned_counts))
print(binned_channels[42], binned_counts[42])

# Create a new FITS file with the binned data
new_fits_file = 'ch2_cla_l1_20200529T104548257_20200529T104556257_1024.fits'

# Create a new Binary Table with the binned data
col1 = fits.Column(name='CHANNEL', format='1I', array=binned_channels)
col2 = fits.Column(name='COUNTS', format='1E', array=binned_counts)

# Create a new BinTableHDU for the binned data
new_hdu = fits.BinTableHDU.from_columns([col1, col2])

# Create a new HDU list
hdu_list = fits.HDUList([fits.PrimaryHDU(), new_hdu])

# Write to the new FITS file
hdu_list.writeto(new_fits_file, overwrite=True)


"""plt.figure(figsize=(10, 6))

# Plot the CHANNEL vs COUNTS
plt.plot(channels, counts, label='Counts per Channel', color='b')

# Add labels and title
plt.xlabel('Energy Channel')
plt.ylabel('Counts')
plt.title('Counts per Energy Channel')

# Optional: Adding a grid for better readability
plt.grid(True)

# Show the plot
plt.legend()
plt.show()"""
