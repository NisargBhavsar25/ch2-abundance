qub_file = '/content/drive/MyDrive/Data/IIRS_sample/ch2_iir_nci_20240616T1338294007_d_img_d18/data/calibrated/20240616/ch2_iir_nci_20240616T1338294007_d_img_d18.qub'

def read_qub(qub_file):
    with rasterio.open(qub_file) as dataset:
        data = dataset.read()
    return data

data = read_qub(qub_file)

# FINAL with Balltree

import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import os
import time
import pickle

def get_coordinates(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    namespace = {'isda': 'https://isda.issdc.gov.in/pds4/isda/v1'}
    refined_coordinates = root.find('.//isda:Geometry_Parameters/isda:Refined_Corner_Coordinates', namespace)
    if refined_coordinates is not None:
        upper_left_lat = refined_coordinates.find('isda:upper_left_latitude', namespace).text
        upper_left_lon = refined_coordinates.find('isda:upper_left_longitude', namespace).text
        upper_right_lat = refined_coordinates.find('isda:upper_right_latitude', namespace).text
        upper_right_lon = refined_coordinates.find('isda:upper_right_longitude', namespace).text
        lower_left_lat = refined_coordinates.find('isda:lower_left_latitude', namespace).text
        lower_left_lon = refined_coordinates.find('isda:lower_left_longitude', namespace).text
        lower_right_lat = refined_coordinates.find('isda:lower_right_latitude', namespace).text
        lower_right_lon = refined_coordinates.find('isda:lower_right_longitude', namespace).text

        print(f"Upper Left Corner: Latitude = {upper_left_lat}, Longitude = {upper_left_lon}")
        print(f"Upper Right Corner: Latitude = {upper_right_lat}, Longitude = {upper_right_lon}")
        print(f"Lower Left Corner: Latitude = {lower_left_lat}, Longitude = {lower_left_lon}")
        print(f"Lower Right Corner: Latitude = {lower_right_lat}, Longitude = {lower_right_lon}")
    else:
        print("Refined Corner Coordinates not found in the XML.")

    return upper_left_lat, upper_left_lon, upper_right_lat, upper_right_lon, lower_left_lat, lower_left_lon, lower_right_lat, lower_right_lon

def overlap(data, upper_left_lat, upper_left_lon, upper_right_lat, upper_right_lon,
           lower_left_lat, lower_left_lon, lower_right_lat, lower_right_lon, csv_file, output_dir, bt):

    print(f"\n Starting overlap detection process...")
    df = pd.read_csv(csv_file)

    upper_lat = (float(upper_left_lat) + float(upper_right_lat))/2
    lower_lat = (float(lower_left_lat) + float(lower_right_lat))/2
    left_lon = (float(upper_left_lon) + float(lower_left_lon))/2
    right_lon = (float(upper_right_lon) + float(lower_right_lon))/2
    height = upper_lat - lower_lat
    width = right_lon - left_lon

    class_data = {}
    overlap_count = 0
    pixel_processing_times = []

    search_distance_km = 12 
    search_radius = search_distance_km / 1737.4  # Convert km to radians on the Moon's surface

    # Iterate through each pixel
    for i in range(len(data)):
        for j in range(len(data[0])):
            start_time = time.time()
            current_lat = upper_lat - (height * i/len(data))
            current_lon = left_lon + (width * j/len(data[0]))
            print(f"Analyzing pixel at (i={i}, j={j}) -> (Lat: {current_lat}, Lon: {current_lon})")

            # Query ball tree for nearby points
            nearby_indices = bt.query_radius([[current_lat, current_lon]], r=search_radius)[0]

            # Only check overlaps with nearby points
            for idx in nearby_indices:
                row = df.iloc[idx]
                csv_lats = [row['V0_LAT'], row['V1_LAT'], row['V2_LAT'], row['V3_LAT']]
                csv_lons = [row['V0_LON'], row['V1_LON'], row['V2_LON'], row['V3_LON']]

                min_lat = min(csv_lats)
                max_lat = max(csv_lats)
                min_lon = min(csv_lons)
                max_lon = max(csv_lons)

                if (min_lat <= current_lat <= max_lat and min_lon <= current_lon <= max_lon):
                    class_name = row['class_file_name']
                    if class_name not in class_data:
                        class_data[class_name] = []

                    spectrum_data = ','.join(map(str, data[i, j, :]))
                    class_data[class_name].append([current_lat, current_lon, spectrum_data])
                    print(f"Overlap found with: {row['class_file_name']} at (Lat: {csv_lats}, Lon: {csv_lons})")
                    overlap_count += 1

                print(f"Time taken for pixel (i={i}, j={j}): {pixel_time:.6f} seconds")

                if overlap_count % 100 == 0 and overlap_count > 0:
                    print(f"\nProcessed {overlap_count} overlapping pixels. Saving intermediate results...")
                    for class_name, data_list in class_data.items():
                        output_file = os.path.join(output_dir, f"{class_name}.csv")
                        try:
                            with open(output_file, 'w') as f:
                                f.write("pixel_latitude,pixel_longitude,spectrum_data\n")
                                for item in data_list:
                                    f.write(f"{item[0]},{item[1]},{item[2]}\n")
                            print(f"✓ Saved intermediate results for {class_name}")
                        except Exception as e:
                            print(f"✗ Error saving intermediate results for {class_name}: {str(e)}")
            pixel_time = time.time() - start_time
            pixel_processing_times.append(pixel_time) 

    # Final save for any remaining data
    print(f"\n Processing complete. Total overlapping pixels: {overlap_count}")
    print("\n Saving final results...")

    for class_name, data_list in class_data.items():
        output_file = os.path.join(output_dir, f"{class_name}.csv")
        try:
            with open(output_file, 'w') as f:
                f.write("pixel_latitude,pixel_longitude,spectrum_data\n")
                for item in data_list:
                    f.write(f"{item[0]},{item[1]},{item[2]}\n")
            print(f"✓ Successfully saved final results for {class_name}")
        except Exception as e:
            print(f"✗ Error saving final results for {class_name}: {str(e)}")

    return list(class_data.keys())

# Main execution
xml_file = '/content/drive/MyDrive/Data/IIRS_sample/ch2_iir_nci_20240616T1338294007_d_img_d18/data/calibrated/20240616/ch2_iir_nci_20240616T1338294007_d_img_d18.xml'
qub_file = '/content/drive/MyDrive/Data/IIRS_sample/ch2_iir_nci_20240616T1338294007_d_img_d18/data/calibrated/20240616/ch2_iir_nci_20240616T1338294007_d_img_d18.qub'
csv_file = '/content/drive/MyDrive/Data/SD_Data_Unique_latlongtime/final_output.csv'
output_dir = '/content/drive/MyDrive/Data/IIRS_sample/Finding_overlaps_csv'

# Load the ball tree
bt = pickle.load(open('/content/drive/MyDrive/Data/SD_Data_Unique_latlongtime/ball_tree.pkl', 'rb'))

os.makedirs(output_dir, exist_ok=True)
upper_left_lat, upper_left_lon, upper_right_lat, upper_right_lon, lower_left_lat, lower_left_lon, lower_right_lat, lower_right_lon = get_coordinates(xml_file)

arr = overlap(data, upper_left_lat, upper_left_lon, upper_right_lat, upper_right_lon,
             lower_left_lat, lower_left_lon, lower_right_lat, lower_right_lon,
             csv_file, output_dir, bt)

print(f"Created CSV files for the following classes: {arr}")