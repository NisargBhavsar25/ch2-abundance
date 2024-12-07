import os
import xml.etree.ElementTree as ET
import csv

# Directory containing XML files
directory = "/content/drive/MyDrive/Data/IIRS_sample/ch2_iir_nci_20240616T1338294007_d_img_d18/data/calibrated/20240616"

# Output CSV file
output_csv = "output2.csv"

# Define the namespace
namespaces = {
    'ns2': 'https://isda.issdc.gov.in/pds4/isda/v1'  # Replace with the actual namespace URI
}

# Open CSV file to write
with open(output_csv, mode='w', newline='') as csv_file:
    fieldnames = ['filename', 'upper_left_latitude', 'upper_left_longitude',
                  'upper_right_latitude', 'upper_right_longitude',
                  'lower_left_latitude', 'lower_left_longitude',
                  'lower_right_latitude', 'lower_right_longitude',
                  'refined_upper_left_latitude', 'refined_upper_left_longitude',
                  'refined_upper_right_latitude', 'refined_upper_right_longitude',
                  'refined_lower_left_latitude', 'refined_lower_left_longitude',
                  'refined_lower_right_latitude', 'refined_lower_right_longitude']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()

    # Loop over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".xml"):  # Only process .xml files
            file_path = os.path.join(directory, filename)

            # Parse the XML file
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()

                # Debug: Print out the XML structure for inspection
                print(f"\nInspecting file: {filename}")
                print(ET.tostring(root, encoding='unicode', method='xml'))  # Print entire XML content as string

                # Find the <System_Level_Coordinates> and <Refined_Corner_Coordinates> elements using the correct namespace
                system_level_coords = root.find('.//ns2:System_Level_Coordinates', namespaces)
                refined_corner_coords = root.find('.//ns2:Refined_Corner_Coordinates', namespaces)

                if system_level_coords is not None and refined_corner_coords is not None:
                    print("Found System_Level_Coordinates and Refined_Corner_Coordinates.")

                    data = {
                        'filename': filename,
                        'upper_left_latitude': system_level_coords.find('ns2:upper_left_latitude', namespaces).text if system_level_coords.find('ns2:upper_left_latitude', namespaces) is not None else 'N/A',
                        'upper_left_longitude': system_level_coords.find('ns2:upper_left_longitude', namespaces).text if system_level_coords.find('ns2:upper_left_longitude', namespaces) is not None else 'N/A',
                        'upper_right_latitude': system_level_coords.find('ns2:upper_right_latitude', namespaces).text if system_level_coords.find('ns2:upper_right_latitude', namespaces) is not None else 'N/A',
                        'upper_right_longitude': system_level_coords.find('ns2:upper_right_longitude', namespaces).text if system_level_coords.find('ns2:upper_right_longitude', namespaces) is not None else 'N/A',
                        'lower_left_latitude': system_level_coords.find('ns2:lower_left_latitude', namespaces).text if system_level_coords.find('ns2:lower_left_latitude', namespaces) is not None else 'N/A',
                        'lower_left_longitude': system_level_coords.find('ns2:lower_left_longitude', namespaces).text if system_level_coords.find('ns2:lower_left_longitude', namespaces) is not None else 'N/A',
                        'lower_right_latitude': system_level_coords.find('ns2:lower_right_latitude', namespaces).text if system_level_coords.find('ns2:lower_right_latitude', namespaces) is not None else 'N/A',
                        'lower_right_longitude': system_level_coords.find('ns2:lower_right_longitude', namespaces).text if system_level_coords.find('ns2:lower_right_longitude', namespaces) is not None else 'N/A',
                    }

                    # Refined coordinates
                    data.update({
                        'refined_upper_left_latitude': refined_corner_coords.find('ns2:upper_left_latitude', namespaces).text if refined_corner_coords.find('ns2:upper_left_latitude', namespaces) is not None else 'N/A',
                        'refined_upper_left_longitude': refined_corner_coords.find('ns2:upper_left_longitude', namespaces).text if refined_corner_coords.find('ns2:upper_left_longitude', namespaces) is not None else 'N/A',
                        'refined_upper_right_latitude': refined_corner_coords.find('ns2:upper_right_latitude', namespaces).text if refined_corner_coords.find('ns2:upper_right_latitude', namespaces) is not None else 'N/A',
                        'refined_upper_right_longitude': refined_corner_coords.find('ns2:upper_right_longitude', namespaces).text if refined_corner_coords.find('ns2:upper_right_longitude', namespaces) is not None else 'N/A',
                        'refined_lower_left_latitude': refined_corner_coords.find('ns2:lower_left_latitude', namespaces).text if refined_corner_coords.find('ns2:lower_left_latitude', namespaces) is not None else 'N/A',
                        'refined_lower_left_longitude': refined_corner_coords.find('ns2:lower_left_longitude', namespaces).text if refined_corner_coords.find('ns2:lower_left_longitude', namespaces) is not None else 'N/A',
                        'refined_lower_right_latitude': refined_corner_coords.find('ns2:lower_right_latitude', namespaces).text if refined_corner_coords.find('ns2:lower_right_latitude', namespaces) is not None else 'N/A',
                        'refined_lower_right_longitude': refined_corner_coords.find('ns2:lower_right_longitude', namespaces).text if refined_corner_coords.find('ns2:lower_right_longitude', namespaces) is not None else 'N/A',
                    })

                    # Write the data to the CSV file
                    writer.writerow(data)
                else:
                    print(f"Skipping file {filename} due to missing coordinates.")

            except ET.ParseError:
                print(f"Error parsing file {filename}. It may not be a valid XML.")
