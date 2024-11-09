import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
import os

data_path = r'scripts\validation\TIF_DATA'

def decode_viridis(rgba):
    """
    Decode RGBA values from the viridis colormap to the original scalar value.
    """
    # Normalize RGB values (ignore the alpha channel)
    rgb_normalized = np.array(rgba[:3]) / 255.0

    # Get the viridis colormap
    viridis = cm.get_cmap('viridis')

    # Find the closest data value (scalar) for the given RGB values
    scalar_values = np.linspace(0, 1, 256)
    colormap_colors = viridis(scalar_values)[:, :3]  # Extract RGB values
    distances = np.linalg.norm(colormap_colors - rgb_normalized, axis=1)  # Compute distances
    scalar_index = np.argmin(distances)  # Find the closest match

    return scalar_values[scalar_index]  # Return the corresponding scalar value

def get_abundance(lat, lon):
    """
    Retrieve and decode elemental abundance values for a given latitude and longitude.
    """
    ops = {}
    tif_paths = os.listdir(data_path)

    for tif_path in tif_paths:
        if 'v2' in tif_path:
            element = tif_path[-9:-7]
            image = Image.open(os.path.join(data_path, tif_path))

            # Extract image data
            image_data = np.array(image)

            # Define longitude and latitude ranges
            lon_start, lon_end = -180, 180
            lat_start, lat_end = -90, 90

            # Calculate step sizes based on image dimensions
            image_shape = image_data.shape
            lon_step = (lon_end - lon_start) / image_shape[1]  # 360 / width
            lat_step = (lat_end - lat_start) / image_shape[0]  # 180 / height

            if lat < lat_start or lat > lat_end or lon < lon_start or lon > lon_end:
                return "Latitude or Longitude is out of range."

            # Calculate pixel indices
            lat_idx = int((lat - lat_start) / lat_step)
            lon_idx = int((lon - lon_start) / lon_step)

            # Ensure indices are within bounds
            lat_idx = min(max(lat_idx, 0), image_shape[0] - 1)
            lon_idx = min(max(lon_idx, 0), image_shape[1] - 1)

            # Get the RGBA abundance value
            rgba_value = image_data[lat_idx, lon_idx]

            # Decode the RGBA value using the viridis colormap
            decoded_value = decode_viridis(rgba_value)
            if (element == 'mg'):
                decoded_value *= 12
            elif (element == 'fe'):
                decoded_value *= 25
            elif (element == 'al'):
                decoded_value *= 25
            elif (element == 'si'):
                decoded_value *=30
            else:
                decoded_value *=12
                
            ops[element] = decoded_value

    return ops

# Example usage
latitude = -85.0
longitude = 170.0
abundance_value = get_abundance(latitude, longitude)

print(f"Decoded abundance values: {abundance_value}")
