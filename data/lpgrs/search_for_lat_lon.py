import pandas as pd

# Sample CSV data loading (assuming the CSV file is named 'data.csv')
df = pd.read_csv('lpgrs_high1_elem_abundance_20deg.csv')

# Function to find the pixel indices for the given lat/lon
def find_pixel_indices(lat, lon, df):
    pixel_indices = []  # Initialize an empty list to collect pixel indices
    
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Check if lat/lon are within the bounds for this row
        if row['MIN_LAT'] <= lat <= row['MAX_LAT'] and row['MIN_LON'] <= lon <= row['MAX_LON']:
            pixel_indices.append(row['PIXEL_INDEX'])  # Append the pixel index to the list
    
    return pixel_indices

# Example usage
lat = 12.34  # example latitude
lon = -56.78  # example longitude

pixel_indices = find_pixel_indices(lat, lon, df)
print("Pixel Indices:", pixel_indices)
