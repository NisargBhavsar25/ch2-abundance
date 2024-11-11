import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from skimage.draw import polygon

class GaussianArray:
    def __init__(self, grid_size=(64, 64)):
        self.grid_size = grid_size
        self.arr = np.zeros((grid_size[0], grid_size[1], 2))  # Initialize array based on grid_size

    def in_block_or_not(self, img_lat, img_lon, block_lat, block_lon):
        return (block_lat[0] <= min(img_lat) <= block_lat[2] and
                block_lat[0] <= max(img_lat) <= block_lat[2] and
                block_lon[0] <= min(img_lon) <= block_lon[2] and
                block_lon[0] <= max(img_lon) <= block_lon[2])

    def convert_coords_to_indices(self, lat, lon, block_lat, block_lon):
        lat_scale = (self.grid_size[0] - 1) / (block_lat[2] - block_lat[0])
        lon_scale = (self.grid_size[1] - 1) / (block_lon[2] - block_lon[0])

        lat_indices = [(lat[i] - block_lat[0]) * lat_scale for i in range(4)]
        lon_indices = [(lon[i] - block_lon[0]) * lon_scale for i in range(4)]

        lat_indices = list(map(int, lat_indices))
        lon_indices = list(map(int, lon_indices))

        return lat_indices, lon_indices

    def calculate_diagonal_length(self, lat_indices, lon_indices):
        diag1 = np.sqrt((lat_indices[2] - lat_indices[0]) ** 2 + (lon_indices[2] - lon_indices[0]) ** 2)
        diag2 = np.sqrt((lat_indices[3] - lat_indices[1]) ** 2 + (lon_indices[3] - lon_indices[1]) ** 2)
        return (diag1 + diag2) / 2

    def generate_gaussian_distribution(self, shape, center, sigma):
        x = np.arange(0, shape[0], 1, float)
        y = np.arange(0, shape[1], 1, float)
        x, y = np.meshgrid(x, y)
        gauss = (np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / (2 * sigma ** 2))) / (2 * np.pi * sigma ** 2)
        return gauss / np.max(gauss)

    def fill_up_the_array(self, img_lat, img_lon, block_lat, block_lon, max_value, target_diagonal=17.625, base_value=2.1739):
        if self.in_block_or_not(img_lat, img_lon, block_lat, block_lon):
            img_lat_indices, img_lon_indices = self.convert_coords_to_indices(img_lat, img_lon, block_lat, block_lon)
            poly_points = np.array([img_lat_indices, img_lon_indices]).T
            min_y, min_x = np.min(poly_points, axis=0)
            max_y, max_x = np.max(poly_points, axis=0)

            avg_diagonal = self.calculate_diagonal_length(img_lat_indices, img_lon_indices)
            scale_factor = target_diagonal / avg_diagonal
            sigma = base_value * scale_factor

            height, width = max_y - min_y + 1, max_x - min_x + 1
            gaussian_values = self.generate_gaussian_distribution((height, width), (height // 2, width // 2), sigma) * max_value

            # Mask the Gaussian to only fit inside the quadrilateral
            rr, cc = polygon(poly_points[:, 0] - min_y, poly_points[:, 1] - min_x, gaussian_values.shape)

            for r, c in zip(rr, cc):
                x, y = r + min_y, c + min_x
                if self.arr[x, y, 1] == 0:  # If cell hasn't been assigned a value yet
                    self.arr[x, y, 0] = gaussian_values[r, c]
                    self.arr[x, y, 1] = 1
                else:
                    count = self.arr[x, y, 1]
                    self.arr[x, y, 0] = (self.arr[x, y, 0] * count + gaussian_values[r, c]) / (count + 1)
                    self.arr[x, y, 1] += 1

    def add_gaussian_box(self, img_lat, img_lon, block_lat, block_lon, max_value, target_diagonal=17.625, base_value=2.1739, plot=False):
        self.fill_up_the_array(img_lat, img_lon, block_lat, block_lon, max_value, target_diagonal, base_value)
        if plot:
            self.visualize_heatmap()

    def visualize_heatmap(self):
        heatmap_data = self.arr[:, :, 0]
        plt.figure(figsize=(10, 8))
        sns.heatmap(heatmap_data, cmap='viridis')
        plt.title('Heatmap of arr Layer 0 with Gaussian Distributions')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.show()
    
    def export_figure_matplotlib(self, f_name, dpi=200, resize_fact=1, plt_show=False):
        arr = self.arr[:, :, 0]
        fig = plt.figure(frameon=False)
        fig.set_size_inches(arr.shape[1]/dpi, arr.shape[0]/dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.imshow(arr)
        plt.savefig(f_name, dpi=(dpi * resize_fact))
        if plt_show:
            plt.show()
        else:
            plt.close()

# Example usage
# gaussian_array = GaussianArray(grid_size=(32,32))
# gaussian_array.add_gaussian_box([-20, 20, 25, -30], [-20, -10, 30, 25], [-100, 100, 100, -100], [-100, -100, 100, 100], 5, plot=True)
# gaussian_array.add_gaussian_box([-20, 20, 25, -30], [-20, -10, 30, 25], [-100, 100, 90, -100], [-100, -80, 70, 100], 4.5, plot=True)
# gaussian_array.add_gaussian_box([-10, 40, 30, -10], [-25, -10, 30, 30], [-100, 100, 140, -100], [-100, -120, 100, 100], 5.5, plot=True)
# gaussian_array.add_gaussian_box([-30, 10, 30, -30], [-30, -40, 10, 10], [-100, 80, 80, -100], [-100, -60, 100, 100], 5.5, plot=True)
