import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data_path = r'scripts\validation\TIF_DATA'

def get_abundance(lat, lon):
    ops = {}
    tif_paths = os.listdir(data_path)
    # print(tif_paths)
    for tif_path in tif_paths:
        if('v2' in tif_path):
            element = tif_path[-9:-7]
            image = Image.open(os.path.join(data_path,tif_path))

            # Extract image data
            image_data = np.array(image)

            # Define longitude and latitude ranges
            lon_start, lon_end = -180, 180
            lat_start, lat_end = -90, 90

            # Calculate step sizes based on image dimensions
            image_shape = image_data.shape
            lon_step = (lon_end - lon_start) / image_shape[1]  # 360 / width
            lat_step = (lat_end - lat_start) / image_shape[0]  # 180 / height

            # Create longitude and latitude grids
            longitudes = np.linspace(lon_start, lon_end, image_shape[1])
            latitudes = np.linspace(lat_start, lat_end, image_shape[0])
            if lat < lat_start or lat > lat_end or lon < lon_start or lon > lon_end:
                return "Latitude or Longitude is out of range."

            # Calculate pixel indices
            lat_idx = int((lat - lat_start) / lat_step)
            lon_idx = int((lon - lon_start) / lon_step)

            # Ensure indices are within bounds
            lat_idx = min(max(lat_idx, 0), image_shape[0] - 1)
            lon_idx = min(max(lon_idx, 0), image_shape[1] - 1)

            # Get the abundance value
            ops[element] = image_data[lat_idx, lon_idx]

    return ops



# Example usage:
# # Load the uploaded image
# tif_file = '/content/drive/MyDrive/Data/TIF DATA/ch2_cla_l2_map_al_v1.tif' # Replace with the path to your .tif file
# # Plot the abundance data mapped to the lunar surface
# plt.figure(figsize=(12, 6))
# plt.imshow(abundance, extent=[lon_start, lon_end, lat_start, lat_end], cmap="viridis", origin="lower")
# plt.colorbar(label="Elemental Abundance")
# plt.title("Elemental Abundance Mapped to Lunar Surface")
# plt.xlabel("Longitude (°)")
# plt.ylabel("Latitude (°)")
# plt.show()

latitude = -85.0
longitude = 170.0
abundance_value = get_abundance(latitude, longitude)

print(abundance_value)