import rasterio
import matplotlib.pyplot as plt

def plot_raster_with_point(latitude, longitude, tif_file_path):
    # Open the raster file
    with rasterio.open(tif_file_path) as dataset:
        # Read the first band of raster
        data = dataset.read(1)
        
        # Transform geographic coordinates to image coordinates
        row, col = dataset.index(longitude, latitude)
        elevation = data[row, col]

        # Plotting the raster data
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap='terrain', extent=(dataset.bounds.left, dataset.bounds.right, dataset.bounds.bottom, dataset.bounds.top))
        plt.colorbar(label="Elevation (meters)")
        plt.scatter(longitude, latitude, color='red', marker='x', label=f"Point: ({latitude}, {longitude})")
        plt.title("Elevation Map")
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.legend()
        
        # Annotate the elevation value at the point
        plt.annotate(f"{elevation} m", (longitude, latitude), textcoords="offset points",
                     xytext=(10, 10), ha='center', color='white', backgroundcolor='black')
        
        plt.show()

def get_elevation(lat, lon, dataset):
    # Get pixel coordinates from latitude and longitude
    x = (lon + 180) * (dataset.width / 360.0)
    col = int(x)
    y = (90 - lat) * (dataset.height / 180.0)
    row = int(y)
    
    # Read the elevation value at the pixel coordinates
    elevation = dataset.read(1)[row, col]
    return elevation

# File path to the TIFF file
file_path = '/content/drive/MyDrive/Data/Elevation Mapping TIFF/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif'

# Coordinates for which to find elevation
latitude = 30
longitude = 30

# Open the TIFF file and extract elevation
with rasterio.open(file_path) as dataset:
    elevation = get_elevation(latitude, longitude, dataset)
    print(f"The elevation at latitude {latitude}° and longitude {longitude}° is {elevation} meters.")
    # plot_raster_with_point(latitude, longitude, file_path)

