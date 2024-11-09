import numpy as np

mn = 0
mx = 1.3
eps=0.001

regions = {
    "Anorthosite": {'mg/si': (mn, mn+eps), 'al/si': (0.9, mx), 'ca/si': (mn, 0.5)},
    "Anorthositic Norite": {'mg/si': (mn, 0.4), 'al/si': (0.6, 0.9), 'ca/si': (mn, 0.45)},
    "Anorthositic Gabbro": {'mg/si': (mn, mn+eps), 'al/si': (0.6, 0.9), 'ca/si': (mn, 0.45)},
    "Anorthositic Troctolite": {'mg/si': (mn, 0.9), 'al/si': (0.6, 0.9), 'ca/si': (mn, 0.45)},
    "Norite": {'mg/si': (mn, 0.9), 'al/si': (0.1, 0.6), 'ca/si': (mn, 0.3)},
    "Gabbro": {'mg/si': (mn, mn+eps), 'al/si': (0.1, 0.6), 'ca/si': (mn, 0.3)},
    "Gabbronorite": {'mg/si': (mn, 0.4), 'al/si': (0.6, 0.9), 'ca/si': (mn, 0.2)},
    "Troctolite": {'mg/si': (0.6, mx), 'al/si': (0.6, 0.9), 'ca/si': (mn, 0.45)}
}

# Function to calculate Euclidean distance from a point to the centroid of a region
def distance_to_region_centroid(new_point, region):
    centroid = np.array([
        (region['mg/si'][0] + region['mg/si'][1]) / 2,
        (region['al/si'][0] + region['al/si'][1]) / 2,
        (region['ca/si'][0] + region['ca/si'][1]) / 2
    ])
    
    distance = np.sqrt(np.sum((new_point - centroid) ** 2))
    
    return distance

# Function to rank regions based on distance to centroid
def rank_regions_by_distance(new_point):
    distances = []
    
    for region_name, region in regions.items():
        distance = distance_to_region_centroid(new_point, region)
        distances.append((region_name, distance))
    
    distances.sort(key=lambda x: x[1])
    
    return distances

# Usage
new_point = np.array([0.0001, 0.95, 0.25])    # [mg/si, al/si, ca/si]
ranked_regions = rank_regions_by_distance(new_point)

# Print the rankings in ascending order of distance
print("Rankings of regions by distance to centroid:")
for rank, (region, distance) in enumerate(ranked_regions, start=1):
    print(f"{rank}. {region} - Distance: {distance}")