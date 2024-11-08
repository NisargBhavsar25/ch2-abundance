import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class GaussianArray:
    def __init__(self, grid_size=64):
        # Initialize a 3D array with a data layer and a count layer
        self.arr = np.zeros((grid_size, grid_size, 2))
        self.grid_size = grid_size

    def in_block_or_not(self, img_lat, img_lon, block_lat, block_lon):
        return (block_lat[0] <= min(img_lat) <= block_lat[2] and
                block_lat[0] <= max(img_lat) <= block_lat[2] and
                block_lon[0] <= min(img_lon) <= block_lon[2] and
                block_lon[0] <= max(img_lon) <= block_lon[2])

    def convert_coords_to_indices(self, lat, lon, block_lat, block_lon):
        lat_scale = (self.grid_size - 1) / (block_lat[2] - block_lat[0])
        lon_scale = (self.grid_size - 1) / (block_lon[2] - block_lon[0])

        lat_indices = [(lat[0] - block_lat[0]) * lat_scale, (lat[1] - block_lat[0]) * lat_scale,
                       (lat[2] - block_lat[0]) * lat_scale, (lat[3] - block_lat[0]) * lat_scale]
        lon_indices = [(lon[0] - block_lon[0]) * lon_scale, (lon[1] - block_lon[0]) * lon_scale,
                       (lon[2] - block_lon[0]) * lon_scale, (lon[3] - block_lon[0]) * lon_scale]

        lat_indices = list(map(int, lat_indices))
        lon_indices = list(map(int, lon_indices))

        return lat_indices, lon_indices

    def generate_gaussian_distribution(self, shape, center, sigma):
        x = np.arange(0, shape[0], 1, float)
        y = np.arange(0, shape[1], 1, float)
        x, y = np.meshgrid(x, y)
        gauss = (np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2)))/(2 * np.pi * sigma ** 2)
        return gauss / np.max(gauss)  # Normalize to max value of 1

    def fill_up_the_array(self, img_lat, img_lon, block_lat, block_lon, max_value, sigma=0):
        if self.in_block_or_not(img_lat, img_lon, block_lat, block_lon):
            img_lat_indices, img_lon_indices = self.convert_coords_to_indices(img_lat, img_lon, block_lat, block_lon)

            height = img_lat_indices[2] - img_lat_indices[0] + 1
            width = img_lon_indices[2] - img_lon_indices[0] + 1
            center = (height // 2, width // 2)

            gaussian_values = self.generate_gaussian_distribution((height, width), center, sigma) * max_value

            for i, x in enumerate(range(img_lon_indices[0], img_lon_indices[2] + 1)):
                for j, y in enumerate(range(img_lat_indices[0], img_lat_indices[2] + 1)):
                    value = gaussian_values[j, i]
                    if self.arr[x, y, 1] == 0:  # If this cell hasn't been assigned a value yet
                        self.arr[x, y, 0] = value
                        self.arr[x, y, 1] = 1
                    else:
                        count = self.arr[x, y, 1]
                        self.arr[x, y, 0] = (self.arr[x, y, 0] * count + value) / (count + 1)
                        self.arr[x, y, 1] += 1

    def add_gaussian_box(self, img_lat, img_lon, block_lat, block_lon, max_value, sigma=0, plot=False):
        self.fill_up_the_array(img_lat, img_lon, block_lat, block_lon, max_value, sigma)
        if plot:
            self.visualize_heatmap()

    def visualize_heatmap(self):
        heatmap_data = self.arr[:, :, 0]  # Select the first layer for the heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Heatmap of arr Layer 0 with Gaussian Distributions')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()

# Example usage
gaussian_array = GaussianArray()

# Adding multiple overlapping boxes with varying parameters
gaussian_array.add_gaussian_box([-20, 20, 20, -20], [-20, -20, 20, 20], [-100, 100, 100, -100], [-100, -100, 100, 100], 1, sigma=5, plot=True)
gaussian_array.add_gaussian_box([0, 40, 40, 0], [0, 0, 40, 40], [-100, 100, 100, -100], [-100, -100, 100, 100], 3, sigma=9, plot=True)
gaussian_array.add_gaussian_box([-10, 30, 30, -10], [-10, -10, 30, 30], [-100, 100, 100, -100], [-100, -100, 100, 100], 2, sigma=3, plot=True)
gaussian_array.add_gaussian_box([10, 50, 50, 10], [10, 10, 50, 50], [-100, 100, 100, -100], [-100, -100, 100, 100], 4, sigma=6, plot=True)
gaussian_array.add_gaussian_box([-30, 10, 10, -30], [-30, -30, 10, 10], [-100, 100, 100, -100], [-100, -100, 100, 100], 1.5, sigma=12, plot=True)
