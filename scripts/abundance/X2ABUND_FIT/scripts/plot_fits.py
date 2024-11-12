from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
def plot_fits(file_path):

    # Load the FITS file
      # Replace with your file path
    fits_file = fits.open(file_path)
    data = fits_file[1].data  # Access the data from the primary HDU
    print(data)
    data = np.array(data, dtype=[('index', 'i4'), ('value', 'f4')])

    # Extract indices and values
    indices = data['index']
    values = data['value']

    # Plot the data
    plt.figure(figsize=(10, 8))
    plt.plot(indices, values, marker='o', linestyle='-', color='b')

    # plt.imshow(data, cmap='gray', origin='lower')
    # plt.colorbar()
    plt.title('FITS File Data')
    plt.xlabel('X Pixel')
    plt.ylabel('Y Pixel')
    ss=file_path.split("/")[-1]
    plt.savefig(f"exp_{ss}.jpg")

    # Close the FITS file
    fits_file.close()
