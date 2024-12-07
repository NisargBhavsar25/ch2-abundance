import os
import rasterio
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_hyperspectral_data(file_name):
    # Define the directory containing the files
    base_dir = '/content/drive/MyDrive/Data/IIRS_sample/ch2_iir_nci_20240616T1338294007_d_img_d18/data/calibrated/20240616/'

    # Construct the full paths for the .hdr and .qub files
    hdr_file = os.path.join(base_dir, f'{file_name}.hdr')
    qub_file = os.path.join(base_dir, f'{file_name}.qub')

    # Check if the .qub file exists in the directory
    if not os.path.exists(qub_file):
        print(f"Error: The file {qub_file} does not exist in the specified directory.")
        return

    # Open the hyperspectral file using rasterio (reading .qub file)
    with rasterio.open(qub_file) as src:
        # Read the hyperspectral data as a 3D array (bands, lines, samples)
        data = src.read()  # shape (bands, lines, samples)

        # Get metadata for understanding the data dimensions
        print(f"Shape of the data: {data.shape}")  # (bands, lines, samples)
        print(f"Data type: {data.dtype}")

        # Get the coordinate reference system and georeferencing information (if available)
        print(f"CRS: {src.crs}")
        print(f"Transform: {src.transform}")

    # Now let's create a DataFrame from the data (flattening it into 2D for easier inspection)
    # Reshape the data to have (lines * samples) rows and bands columns
    reshaped_data = data.reshape((-1, data.shape[0]))  # Flatten samples and lines
    df = pd.DataFrame(reshaped_data, columns=[f'Band_{i+1}' for i in range(data.shape[0])])

    # Show the first few rows of the DataFrame
    print("First few rows of the data (DataFrame):")
    print(df.head())

    # Choose valid pixels for plotting (you can modify these indices as per your requirement)
    pixels_to_plot = [(100, 100), (10, 10), (69,69), (96, 96), (13, 10), (20, 2004), (249, 13000), (200, 10000), (50, 50)]  # Example valid pixel positions

    # Plot spectral lines for valid pixel positions
    plt.figure(figsize=(10, 6))  # Set plot size for clarity
    for pixel_pos in pixels_to_plot:
        # Access the spectral data for each pixel across all bands (bands, lines, samples)
        pixel_data = data[:, pixel_pos[1], pixel_pos[0]]  # Notice order: [bands, lines, samples]

        # Plot the spectral line
        plt.plot(np.arange(1, data.shape[0] + 1), pixel_data, label=f'Pixel {pixel_pos}')

    # Set up the plot
    plt.title("Spectral Lines for Different Pixels")
    plt.xlabel("Spectral Bands")
    plt.ylabel("Reflectance / Radiance")
    plt.grid(True)
    plt.legend()
    plt.show()


