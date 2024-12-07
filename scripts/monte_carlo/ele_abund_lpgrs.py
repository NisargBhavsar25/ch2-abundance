import cv2
import numpy as np
import os

class AbundanceCalculator:
    def __init__(self, directory_path):
        self.images = {}
        self.dict_map = {'fe': [0, 25], 'al': [0, 20], 'ti': [0, 6], 'o': [40, 47], 'mg': [0, 16], 'ca': [2, 18]}
        self._load_images(directory_path)
    
    def _load_images(self, directory_path):
        image_files = os.listdir(directory_path)
        for image_file in image_files:
            element = image_file.split('_')[0].lower()
            image_path = os.path.join(directory_path, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise ValueError(f"Error: Greyscale image could not be loaded. Check path: {image_path}")
            self.images[element] = {
                'image': image,
                'scale_min': np.min(image),
                'scale_max': np.max(image),
                'shape': image.shape,
                'scale_range': self.dict_map[element]  # Changed from i to element
            }
    
    def _lat_lon_to_pixel(self, lat, lon, image_height, image_width):
        row = int((90 - lat) / 180 * (image_height - 1))
        col = int((lon + 180) / 360 * (image_width - 1))
        return row, col
    
    def get_abundances(self, lat, lon):
        abundances = {}
        for element, img_data in self.images.items():
            image = img_data['image']
            el_sc_min, el_sc_max = img_data['scale_range']
            
            latitude_index, longitude_index = self._lat_lon_to_pixel(lat, lon, *img_data['shape'])
            greyscale_value = image[latitude_index, longitude_index]
            
            normalized_value = (greyscale_value - img_data['scale_min']) / (img_data['scale_max'] - img_data['scale_min'])
            wt_value = normalized_value * (el_sc_max - el_sc_min) + el_sc_min
            
            abundances[element] = wt_value
        return abundances
    
# example usage
# from ele_abund_lpgrs import AbundanceCalculator
# image_dir = 'lpgrs_elemental_maps'
# abd_calc = AbundanceCalculator(image_dir)
# abundance = abd_calc.get_abundances(lat, lon)