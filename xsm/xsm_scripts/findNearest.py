import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

"""
This script is used to group the locations based on the distance threshold.
"""

# Constants
MOON_RADIUS_KM = 1737.4  # Radius of the Moon in kilometers

df = pd.read_csv('/home/ubuntu/find_nearest/updated_final_output.csv')

longitude_columns = ['V0_LON', 'V1_LON', 'V2_LON', 'V3_LON']
for col in longitude_columns:
    df[col] = df[col].apply(lambda x: x + 360 if x < 0 else x)

latitude_columns = ['V0_LAT', 'V1_LAT', 'V2_LAT', 'V3_LAT']

df['Centroid_LAT'] = df[latitude_columns].mean(axis=1)
df['Centroid_LON'] = df[longitude_columns].mean(axis=1)

df['Centroid_LON'] = df['Centroid_LON'].apply(lambda x: x - 360 if x > 180 else x)

coords = np.radians(df[['Centroid_LAT', 'Centroid_LON']].values)

tree = BallTree(coords, metric='haversine')

distance_threshold_km = 35

# Convert the threshold distance into radians (since BallTree uses radians)
distance_threshold_rad = distance_threshold_km / MOON_RADIUS_KM

# Query the tree for all points within the distance threshold
ind = tree.query_radius(coords, r=distance_threshold_rad)
#dist, ind = tree.query(coords, k=len(coords), distance_upper_bound=distance_threshold_rad)

clusters = {}
group_id = 0
assigned = np.zeros(len(df), dtype=bool)  # To keep track of which locations have been assigned to a group

# Iterate through each location and assign group ids
for i, neighbors in enumerate(ind):
    # Ignore the point itself (i.e., it's not its own neighbor)
    neighbors = neighbors[neighbors != i]
    
    # If the point has neighbors within the threshold and hasn't been assigned a group yet
    if len(neighbors) > 0 and not assigned[i]:
        # Create a new group
        clusters[group_id] = [i] + neighbors.tolist()
        # Mark the point and its neighbors as assigned to a group
        assigned[i] = True
        for neighbor in neighbors:
            assigned[neighbor] = True
        group_id += 1

# Step 7: Assign group_id to the DataFrame based on the clusters
group_ids = np.full(len(df), -1)  # Initialize group IDs with -1 (no group)

# Populate group_ids based on the clusters
for group_id, indices in clusters.items():
    group_ids[indices] = group_id

df['group_id'] = group_ids

# Step 8: Save the grouped data to a new CSV file
output_filename = 'BallTree_sorted.csv'
df_sorted = df.sort_values(by='group_id')  # Sort the DataFrame by group_id for easier inspection
df_sorted.to_csv(output_filename, index=False)

print(f"Grouped locations saved to {output_filename}")
print(df_sorted.head()) 