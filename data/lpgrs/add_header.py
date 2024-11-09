import pandas as pd

# Define the column names
column_names = [
    "PIXEL_INDEX", "MIN_LAT", "MAX_LAT", "MIN_LON", "MAX_LON", 
    "AM", "NEUTRON_DEN", "W_MGO", "W_AL2O3", "W_SIO2", "W_CAO", 
    "W_TIO2", "W_FEO", "W_K", "W_TH", "W_U", "E[0,0]", "E[0,1]", 
    "E[0,2]", "E[0,3]", "E[0,4]", "E[0,5]", "E[0,6]", "E[0,7]", 
    "E[0,8]", "E[1,1]", "E[1,2]", "E[1,3]", "E[1,4]", "E[1,5]", 
    "E[1,6]", "E[1,7]", "E[1,8]", "E[2,2]", "E[2,3]", "E[2,4]", 
    "E[2,5]", "E[2,6]", "E[2,7]", "E[2,8]", "E[3,3]", "E[3,4]", 
    "E[3,5]", "E[3,6]", "E[3,7]", "E[3,8]", "E[4,4]", "E[4,5]", 
    "E[4,6]", "E[4,7]", "E[4,8]", "E[5,5]", "E[5,6]", "E[5,7]", 
    "E[5,8]", "E[6,6]", "E[6,7]", "E[6,8]", "E[7,7]", "E[7,8]", 
    "E[8,8]"
]

# Read the existing CSV file (without column headers)
df = pd.read_csv('output_file.csv', header=None)

df.columns = column_names

# Save the updated dataframe to a new CSV file
df.to_csv('lpgrs_high1_elem_abundance_20deg.csv', index=False) #change this
