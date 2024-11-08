import rasterio
import numpy as np
import matplotlib.pyplot as plt

# Path to the .tif file
tif_file = '/home/manasj/Downloads/ch2_cla_l2_map_si_v2.tif'

# Open the .tif file using rasterio
with rasterio.open(tif_file) as src:
    # Read the image data as a NumPy array
    image = src.read()
    # Read the metadata
    metadata = src.meta
    
    # Print metadata information
    print("Metadata:", metadata)


# Display metadata
for key, value in metadata.items():
    print(f"{key}: {value}")


# Display the first band of the image
plt.imshow(image[0], cmap='gray')
plt.title("First Band of TIF Image")
plt.axis('off')
plt.show()
