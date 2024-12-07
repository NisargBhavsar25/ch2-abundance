import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os

def create_composition_layer(row):
    """
    Create composition layer dictionary from a row of the CSV file
    """
    return [{
        'elements': [
            {'atomic_number': '11', 'weight_fraction': str(row['na'])},    # Na
            {'atomic_number': '12', 'weight_fraction': str(row['mg'])},    # Mg
            {'atomic_number': '13', 'weight_fraction': str(row['al'])},    # Al
            {'atomic_number': '14', 'weight_fraction': str(row['si'])},    # Si
            {'atomic_number': '20', 'weight_fraction': str(row['ca'])},    # Ca
            {'atomic_number': '22', 'weight_fraction': str(row['ti'])},    # Ti
            {'atomic_number': '26', 'weight_fraction': str(row['fe'])},    # Fe
            {'atomic_number': '8', 'weight_fraction': str(row['o'])},      # O
        ],
        'density': '2.5',    # Adjust as appropriate
        'thickness': '1'     # Adjust as appropriate
    }]

# General settings and other configurations remain the same
general_settings = {
    'outputfile': 'test.xmso',
    'n_photons_interval': '1000',
    'n_photons_line': '100000',
    'n_interactions_trajectory': '4',
    'comments': ''
}

# Keep other settings as they were...
geometry_settings = {
    'd_sample_source': '100',
    'n_sample_orientation': {'x': '0', 'y': '0.707107', 'z': '0.707107'},
    'p_detector_window': {'x': '0', 'y': '-1', 'z': '100'},
    'n_detector_orientation': {'x': '0', 'y': '1', 'z': '0'},
    'area_detector': '0.3',
    'collimator_height': '0',
    'collimator_diameter': '0',
    'd_source_slit': '100',
    'slit_size': {'slit_size_x': '0.001', 'slit_size_y': '0.001'}
}

excitation_settings = [
    {
        'energy': '0.1',
        'horizontal_intensity': '1e+09',
        'vertical_intensity': '1e+09',
        'sigma_x': '0',
        'sigma_xp': '0',
        'sigma_y': '0',
        'sigma_yp': '0'
    },
    {
        'energy': '27',
        'horizontal_intensity': '1e+09',
        'vertical_intensity': '1e+09',
        'sigma_x': '0',
        'sigma_xp': '0',
        'sigma_y': '0',
        'sigma_yp': '0'
    }
]
    

absorbers_settings = {
    'detector_path': {
        'layer': {
            'elements': [
                {'atomic_number': '4', 'weight_fraction': '100'}  # Be window
            ],
            'density': '1.85',
            'thickness': '0.002'
        }
    }
}

detector_settings = {
    'detector_type': 'SiLi',
    'live_time': '1',
    'pulse_width': '1e-05',
    'nchannels': '2048',
    'gain': '0.02',
    'zero': '0',
    'fano': '0.12',
    'noise': '0.1',
    'crystal': {
        'layer': {
            'elements': [
                {'atomic_number': '14', 'weight_fraction': '100'}  # Si crystal
            ],
            'density': '2.33',
            'thickness': '0.5'
        }
    }
}

def build_xml_tree(composition_layers):
    """
    Build XML tree with the given composition layers
    """
    root = ET.Element('xmimsim')
    
    # Add general settings
    general = ET.SubElement(root, 'general', version='1.0')
    for key, value in general_settings.items():
        elem = ET.SubElement(general, key)
        elem.text = value

    # Add composition
    composition = ET.SubElement(root, 'composition')
    for layer_data in composition_layers:
        layer = ET.SubElement(composition, 'layer')
        for element_data in layer_data['elements']:
            element = ET.SubElement(layer, 'element')
            atomic_number = ET.SubElement(element, 'atomic_number')
            atomic_number.text = element_data['atomic_number']
            weight_fraction = ET.SubElement(element, 'weight_fraction')
            weight_fraction.text = element_data['weight_fraction']
        density = ET.SubElement(layer, 'density')
        density.text = layer_data['density']
        thickness = ET.SubElement(layer, 'thickness')
        thickness.text = layer_data['thickness']
    ref_layer = ET.SubElement(composition, 'reference_layer')
    ref_layer.text = '1'

    # Add other sections (geometry, excitation, absorbers, detector)
    # Geometry
    geometry = ET.SubElement(root, 'geometry')
    for key, value in geometry_settings.items():
        if isinstance(value, dict):
            sub_elem = ET.SubElement(geometry, key)
            for sub_key, sub_value in value.items():
                coord = ET.SubElement(sub_elem, sub_key)
                coord.text = sub_value
        else:
            elem = ET.SubElement(geometry, key)
            elem.text = value

    # Excitation
    excitation = ET.SubElement(root, 'excitation')
    for setting in excitation_settings:
        continuous = ET.SubElement(excitation, 'continuous')
        for key, value in setting.items():  # Iterate over key-value pairs in the dictionary
            elem = ET.SubElement(continuous, key)
            elem.text = value

    # Absorbers
    absorbers = ET.SubElement(root, 'absorbers')
    detector_path = ET.SubElement(absorbers, 'detector_path')
    layer = ET.SubElement(detector_path, 'layer')
    for element_data in absorbers_settings['detector_path']['layer']['elements']:
        element = ET.SubElement(layer, 'element')
        atomic_number = ET.SubElement(element, 'atomic_number')
        atomic_number.text = element_data['atomic_number']
        weight_fraction = ET.SubElement(element, 'weight_fraction')
        weight_fraction.text = element_data['weight_fraction']
    density = ET.SubElement(layer, 'density')
    density.text = absorbers_settings['detector_path']['layer']['density']
    thickness = ET.SubElement(layer, 'thickness')
    thickness.text = absorbers_settings['detector_path']['layer']['thickness']

    # Detector
    detector = ET.SubElement(root, 'detector')
    for key, value in detector_settings.items():
        if isinstance(value, dict):
            sub_elem = ET.SubElement(detector, key)
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    sub_layer = ET.SubElement(sub_elem, sub_key)
                    for elem_data in sub_value['elements']:
                        element = ET.SubElement(sub_layer, 'element')
                        atomic_number = ET.SubElement(element, 'atomic_number')
                        atomic_number.text = elem_data['atomic_number']
                        weight_fraction = ET.SubElement(element, 'weight_fraction')
                        weight_fraction.text = elem_data['weight_fraction']
                    density = ET.SubElement(sub_layer, 'density')
                    density.text = sub_value['density']
                    thickness = ET.SubElement(sub_layer, 'thickness')
                    thickness.text = sub_value['thickness']
                else:
                    coord = ET.SubElement(sub_elem, sub_key)
                    coord.text = sub_value
        else:
            elem = ET.SubElement(detector, key)
            elem.text = value

    return root

def write_xml_file(filename, root):
    """
    Write XML tree to file with DOCTYPE declaration
    """
    xml_str = ET.tostring(root, encoding='unicode')
    doctype = '<!DOCTYPE xmimsim SYSTEM "http://www.xmi.UGent.be/xml/xmimsim-1.0.dtd">'
    xml_str = '<?xml version="1.0"?>\n' + doctype + '\n' + xml_str
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent="  ")
    with open(filename, 'w') as f:
        f.write(pretty_xml_str)

def generate_xml_files():
    """
    Generate XML files for each row in the CSV
    """
    # Create output directory if it doesn't exist
    output_dir = 'xml_outputs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read CSV file
    df = pd.read_csv('selected.csv')
    
    # Generate XML file for each row
    for index, row in df.iterrows():
        # Create composition layers from row data
        composition_layers = create_composition_layer(row)
        
        # Build XML tree
        root = build_xml_tree(composition_layers)
        
        # Generate filename (you can modify this pattern)
        filename = os.path.join(output_dir, f'input_{index+1}.xmsi')
        
        # Write XML file
        write_xml_file(filename, root)
        print(f"Generated XML file: {filename}")

if __name__ == "__main__":
    generate_xml_files()