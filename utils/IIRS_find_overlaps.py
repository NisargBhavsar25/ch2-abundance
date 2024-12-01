import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import rasterio
import matplotlib.pyplot as plt
import os
import time
import pickle
import glob
from pathlib import Path

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

                    spectrum_data = ','.join(map(str, data[:, j, i]))
                    class_data[class_name].append([current_lat, current_lon, spectrum_data])
                    print(f"Overlap found with: {row['class_file_name']} at (Lat: {csv_lats}, Lon: {csv_lons})")
                    overlap_count += 1

            pixel_time = time.time() - start_time
            pixel_processing_times.append(pixel_time)
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

def read_qub(qub_file):
    with rasterio.open(qub_file) as dataset:
        data = dataset.read()
    return data

def process_directory(input_dir, csv_file, output_base_dir, ball_tree_path):
    """
    Process all XML and QUB file pairs in the given directory.

    Parameters:
    input_dir (str): Path to directory containing XML and QUB files
    csv_file (str): Path to the CSV file containing reference data
    output_base_dir (str): Base directory where results will be saved
    ball_tree_path (str): Path to the pickle file containing the ball tree
    """
    # Load the ball tree once
    bt = pickle.load(open(ball_tree_path, 'rb'))

    # Find all XML files in the directory
    xml_files = glob.glob(os.path.join(input_dir, "**/*.xml"), recursive=True)

    print(f"Found {len(xml_files)} XML files to process")

    for xml_file in xml_files:
        try:
            qub_file = xml_file.replace('.xml', '.qub')

            if not os.path.exists(qub_file):
                print(f"Warning: No matching QUB file found for {xml_file}")
                continue

            print(f"\nProcessing file pair:")
            print(f"XML: {xml_file}")
            print(f"QUB: {qub_file}")

            file_name = Path(xml_file).stem
            output_dir = os.path.join(output_base_dir, file_name)
            os.makedirs(output_dir, exist_ok=True)

            coordinates = get_coordinates(xml_file)

            data = read_qub(qub_file)

            processed_classes = overlap(data, *coordinates, csv_file, output_dir, bt)

            print(f"\nSuccessfully processed {file_name}")
            print(f"Created CSV files for classes: {processed_classes}")

        except Exception as e:
            print(f"Error processing {xml_file}: {str(e)}")
            continue


# Example paths - modify these as needed
input_directory = "/path/to/input/directory"
csv_file = "/path/to/final_output.csv"
output_base_directory = "/path/to/output/directory"
ball_tree_path = "/path/to/ball_tree.pkl"

process_directory(input_directory, csv_file, output_base_directory, ball_tree_path)
