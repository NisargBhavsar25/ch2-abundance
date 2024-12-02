import pandas as pd
import numpy as np
import pickle
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from tqdm import tqdm
from astropy.io import fits
import os  # Added import for directory management

# Constants
MIN_CA = 3.6
MAX_CA = 4.1
CONV_FACTOR = 0.0135
START_ID = int(MIN_CA / CONV_FACTOR)
END_ID = int(MAX_CA / CONV_FACTOR) + 1

# Load CSV (ensure correct path)
csv_path = 'classnames.csv'  # Adjust the path if needed
df = pd.read_csv(csv_path)
print(f"Number of rows in DataFrame: {len(df)}")

# Load ball tree
pkl_path = 'ball_tree.pkl'  # Ensure correct path
with open(pkl_path, 'rb') as file:
    ball_tree = pickle.load(file)
print('Ball tree loaded')

# Query nearby points
location = np.array([[50, 50]])  # Replace with a function if needed
radius = 0.7
indices = ball_tree.query_radius(location, r=radius)[0].tolist()
surr_df = df.iloc[indices].reset_index(drop=True)  # Reset index to make it a DataFrame
print('DataFrame with surrounding points created')

# Process FITS files
ca_regions = []
y_values = []

for _, row in tqdm(surr_df.iterrows(), total=len(surr_df)):
    file_name = row["class_file_name"]

    year = file_name[:4]
    month = file_name[4:6]
    day = file_name[6:8]

    file_path = f"../../../data/class/fits/cla/data/calibrated/{year}/{month}/{day}/ch2_cla_l1_{file_name}.fits"
    try:
        with fits.open(file_path) as hdul:
            data = hdul[1].data
            y = data['COUNTS']
            ca_region = y[START_ID:END_ID]  # Extract Ca region
            y_values.append(y)
            ca_regions.append(ca_region)
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        y_values.append(None)
        ca_regions.append(None)

# Add processed data to DataFrame
surr_df['y'] = y_values
surr_df['ca_region'] = ca_regions
print(surr_df.columns)

# Standardize the data (important for Spectral Clustering)
ca_data = np.array([region for region in surr_df['ca_region'] if region is not None])
scaler = StandardScaler()
ca_data_scaled = scaler.fit_transform(ca_data)

# Create a directory for all plots
plots_dir = "clustering_plots"
os.makedirs(plots_dir, exist_ok=True)

# Elbow method using Silhouette Score
silhouette_scores = []
k_values = range(2, 11)  # Try different values of n_clusters from 2 to 10

for k in k_values:
    spectral = SpectralClustering(n_clusters=k, affinity='nearest_neighbors', random_state=42)
    spectral.fit(ca_data_scaled)

    # Calculate silhouette score for the current number of clusters
    score = silhouette_score(ca_data_scaled, spectral.labels_)
    silhouette_scores.append(score)
    print(f"Silhouette Score for {k} clusters: {score}")

# Plot the Elbow (Silhouette Score) graph
plt.plot(k_values, silhouette_scores, marker='o')
plt.title('Silhouette Score for Different Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.savefig(os.path.join(plots_dir, 'silhouette_score_elbow.png'))
plt.close()

# Select the best number of clusters based on silhouette score
best_k = k_values[np.argmax(silhouette_scores)]
print(f"The best number of clusters based on Silhouette Score: {best_k}")

# Perform Spectral Clustering with the best number of clusters
spectral = SpectralClustering(n_clusters=best_k, affinity='nearest_neighbors', random_state=42)
spectral.fit(ca_data_scaled)

# Assign cluster labels to the DataFrame
surr_df['cluster'] = spectral.labels_

# Print cluster information
print(f"Cluster labels:\n{surr_df['cluster'].value_counts()}")

# Plot and save data for each cluster
for cluster_num in range(best_k):
    # Create a directory for this cluster
    cluster_dir = os.path.join(plots_dir, f"cluster_{cluster_num}")
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Filter rows belonging to the current cluster
    cluster_data = surr_df[surr_df['cluster'] == cluster_num]

    # Plot the 'y' values for the current cluster
    plt.figure(figsize=(10, 6))
    for i, row in cluster_data.iterrows():
        plt.plot(row['y'], alpha=0.5)  # Plot each 'y' array for the cluster
        plt.title(f"Cluster {cluster_num}")
        plt.xlabel("Data Points")
        plt.ylabel("COUNTS")
        plt.yscale('log')
        plt.grid(True)
        
        # Save in cluster-specific directory
        plt.savefig(os.path.join(cluster_dir, f"spectrum_{i}.png"))
        plt.close()

print("Clustering complete and plots saved in cluster-specific directories.")