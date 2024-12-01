def get_abundance_from_lat_lon(lat, lon, image_paths, dict_map):
    """
    Maps a single latitude and longitude to their corresponding abundances (%wt) for multiple images and scales.

    :param lat: Latitude of the point.
    :param lon: Longitude of the point.
    :param image_paths: List of file paths to greyscale images.
    :param dict_map: Dictionary containing scale ranges for each image (key: image index, value: [el_sc_min, el_sc_max]).
    :return: Dictionary with keys as element indices and values as elemental abundances (%wt) at the given point.
    """
    abundances = {}

    for i, image_path in enumerate(image_paths):
        # Load the greyscale image
        greyscale_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Check if image was loaded successfully
        if greyscale_image is None:
            raise ValueError(f"Error: Greyscale image could not be loaded. Check path: {image_path}")

        # Get the scale range from the dictionary
        el_sc_min, el_sc_max = dict_map[i]

        # Get the greyscale image dimensions
        image_height, image_width = greyscale_image.shape

        # Function to map latitude and longitude to pixel indices
        def lat_lon_to_pixel(lat, lon, image_height, image_width):
            row = int((90 - lat) / 180 * (image_height - 1))
            col = int((lon + 180) / 360 * (image_width - 1))
            return row, col

        # Map latitude and longitude to pixel indices
        latitude_index, longitude_index = lat_lon_to_pixel(lat, lon, image_height, image_width)

        # Extract the greyscale value
        greyscale_value = greyscale_image[latitude_index, longitude_index]

        # Normalize the greyscale value to %wt
        scale_min = np.min(greyscale_image)
        scale_max = np.max(greyscale_image)
        normalized_value = (greyscale_value - scale_min) / (scale_max - scale_min)
        wt_value = normalized_value * (el_sc_max - el_sc_min) + el_sc_min

        # Store the abundance in the dictionary with the element index as the key
        abundances[i] = wt_value

    return abundances


# Example usage
lat = 0  # Latitude of the point
lon = 0  # Longitude of the point
image_paths = ['/content/Fe_grey.png', '/content/Al_image.jpeg', '/content/Ti_image.png', '/content/O_grey_scale.jpeg','/content/Mg_image.png','/content/Ca_image.png']  # Example image paths
dict_map = {0: [0, 25], 1: [0, 20], 2: [0, 6], 3: [40, 47], 4: [0, 16], 5: [2, 18]}  # Scale ranges for Fe, Al, Ti, O

# Get the abundances for the given latitude and longitude
abundance_at_point = get_abundance_from_lat_lon(lat, lon, image_paths, dict_map)

# Display the results
abundance_at_point
