import pandas as pd

# Sample CSV data loading (assuming the CSV file is named 'data.csv')

# Function to find the pixel indices for the given lat/lon
def get_abundance(path, lat, lon):
    pixel_indices = []  # Initialize an empty list to collect pixel indices
    df = pd.read_csv(path)  # Load the CSV file into a pandas dataframe
    elements = ['W_MGO','W_AL2O3','W_SIO2','W_CAO','W_TIO2','W_FEO','W_K','W_TH','W_U']
    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        # Check if lat/lon are within the bounds for this row
        if row['MIN_LAT'] <= lat <= row['MAX_LAT'] and row['MIN_LON'] <= lon <= row['MAX_LON']:
            pixel_indices.append(row['PIXEL_INDEX'])  # Append the pixel index to the list
    
    op = df[df['PIXEL_INDEX'].isin(pixel_indices)][elements].to_dict()
    for key, value in op.items():
        op[key] = value[pixel_indices[0]]
    
    return op

# Example usage
path = r'scripts\validation\LPGRS\lpgrs_high1_elem_abundance_2deg.csv'
lat = 12.34  # example latitude
lon = -56.78  # example longitude

pixel_indices = get_abundance(path, lat, lon)
print("Abundance Values:", pixel_indices)
