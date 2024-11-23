import numpy as np
import matplotlib.pyplot as plt

# from google.colab import drive
# drive.mount('/content/drive')

# Link to Data: https://pgda.gsfc.nasa.gov/products/54

def parse_lbl(lbl_file):
    """
    Parses the .LBL file and extracts metadata into a dictionary.
    """
    metadata = {}
    with open(lbl_file, 'r') as f:
        for line in f:
            if "=" in line:
                key, value = line.split("=")
                metadata[key.strip()] = value.strip().replace('"', '').replace(';', '')
    return metadata

def plot_elevation_map(height_data, extent):
    """
    Plots the elevation data on a 2D map.
    """
    plt.figure(figsize=(12, 6))
    plt.imshow(height_data, cmap="viridis", extent=extent, aspect='auto')
    plt.colorbar(label="Height (km)")
    plt.title("Lunar Elevation Map")
    plt.xlabel("Longitude (°)")
    plt.ylabel("Latitude (°)")
    plt.show()

def get_height_at_point(height_data, extent, latitude, longitude):
    """
    Returns the height at a specific latitude and longitude.
    """
    min_lon, max_lon, min_lat, max_lat = extent
    rows, cols = height_data.shape
    
    # Normalize latitude and longitude to indices
    row = int((max_lat - latitude) / (max_lat - min_lat) * (rows - 1))
    col = int((longitude - min_lon) / (max_lon - min_lon) * (cols - 1))
    
    return height_data[row, col]

# Paths to files
lbl_path = "/content/drive/MyDrive/Data/Elevation Mapping IMG/SLDEM2015_256_60S_60N_000_360_FLOAT.LBL"
img_path = "/content/drive/MyDrive/Data/Elevation Mapping IMG/SLDEM2015_256_60S_60N_000_360_FLOAT.IMG"

# Parse the .LBL file
metadata = parse_lbl(lbl_path)

# Extract variables from metadata
rows = int(metadata['LINES'])
cols = int(metadata['LINE_SAMPLES'])
data_type = np.float32 if metadata['SAMPLE_TYPE'] == "PC_REAL" else None
scaling_factor = float(metadata['SCALING_FACTOR'])
offset = float(metadata['OFFSET'])
min_lat = float(metadata['MINIMUM_LATITUDE'].split()[0])
max_lat = float(metadata['MAXIMUM_LATITUDE'].split()[0])
min_lon = float(metadata['WESTERNMOST_LONGITUDE'].split()[0])
max_lon = float(metadata['EASTERNMOST_LONGITUDE'].split()[0])

# Load the .IMG file
data = np.fromfile(img_path, dtype=data_type).reshape((rows, cols))

# Apply scaling and offset
height_data = data * scaling_factor + offset

# Define extent (longitude and latitude)
extent = [min_lon, max_lon, min_lat, max_lat]

# Plot the elevation map
plot_elevation_map(height_data, extent)

# Example: Find height at a specific latitude and longitude
latitude = 30  # Example latitude
longitude = 60  # Example longitude
height = get_height_at_point(height_data, extent, latitude, longitude)
print(f"Height at latitude {latitude}° and longitude {longitude}°: {height} km")
