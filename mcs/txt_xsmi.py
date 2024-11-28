import xml.etree.ElementTree as ET
from xml.dom import minidom

# [All constants remain the same until excitation_settings]
# General settings
general_settings = {
    'outputfile': 'test.xmso',
    'n_photons_interval': '1000',
    'n_photons_line': '10000',
    'n_interactions_trajectory': '4',
    'comments': ''
}

# Composition layers (single layer)
composition_layers = [
    {
        'elements': [
            {'atomic_number': '11', 'weight_fraction': '5'},    # Na
            {'atomic_number': '12', 'weight_fraction': '5'},   # Mg
            {'atomic_number': '13', 'weight_fraction': '16'},   # Al
            {'atomic_number': '14', 'weight_fraction': '20'},   # Si
            {'atomic_number': '20', 'weight_fraction': '14'},   # Ca
            {'atomic_number': '22', 'weight_fraction': '5'},    # Ti
            {'atomic_number': '26', 'weight_fraction': '25'},   # Fe
            {'atomic_number': '8', 'weight_fraction': '10'},     # O
        ],
        'density': '2.5',
        'thickness': '1'
    }
]

reference_layer = '1'

# Geometry settings
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

def process_data_file(filepath):
    """
    Reads a text file with three columns and converts it into a list of dictionaries
    with specified format.
    
    Args:
        filepath (str): Path to the input text file
        
    Returns:
        list: List of dictionaries containing the processed data
    """
    result = []
    
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Split the line by whitespace and remove any empty strings
                columns = [col for col in line.strip().split() if col]
                
                if len(columns) >= 3:  # Ensure we have at least 3 columns
                    data_dict = {
                        'energy': columns[0],
                        'horizontal_intensity': columns[2],
                        'vertical_intensity': columns[2],
                        'sigma_x': '0',
                        'sigma_xp': '0',
                        'sigma_y': '0',
                        'sigma_yp': '0'
                    }
                    result.append(data_dict)
                
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        
    return result

# Absorbers settings
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

# Detector settings
detector_settings = {
    'detector_type': 'SiLi',
    'live_time': '1',
    'pulse_width': '1e-05',
    'nchannels': '1024',
    'gain': '0.04',
    'zero': '0',
    'fano': '0.12',
    'noise': '0.1',
    'crystal': {
        'layer': {
            'elements': [
                {'atomic_number': '14', 'weight_fraction': '100'},  # Si crystal
            ],
            'density': '2.33',
            'thickness': '0.5'
        }
    }
}

def build_xml_tree(excitation_settings):
    # Create the root element
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
    ref_layer.text = reference_layer

    # Add geometry
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

    # Add excitation
    excitation = ET.SubElement(root, 'excitation')
    for setting in excitation_settings:
        continuous = ET.SubElement(excitation, 'continuous')
        for key, value in setting.items():
            elem = ET.SubElement(continuous, key)
            elem.text = value

    # Add absorbers
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

    # Add detector
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

def write_xml_file(filename, excitation_settings):
    root = build_xml_tree(excitation_settings)
    # Convert the ElementTree to a string
    xml_str = ET.tostring(root, encoding='unicode')
    # Add DOCTYPE declaration
    doctype = '<!DOCTYPE xmimsim SYSTEM "http://www.xmi.UGent.be/xml/xmimsim-1.0.dtd">'
    # Use minidom for pretty printing
    xml_str = '<?xml version="1.0"?>\n' + doctype + '\n' + xml_str
    reparsed = minidom.parseString(xml_str)
    pretty_xml_str = reparsed.toprettyxml(indent="  ")
    # Write to file
    with open(filename, 'w') as f:
        f.write(pretty_xml_str)
    print(f"XML input file '{filename}' has been generated.")

# Main execution
if __name__ == "__main__":
    input_txt_file = "your_input.txt"  # Replace with your text file name
    excitation_settings = process_data_file(input_txt_file)
    write_xml_file('input.xmsi', excitation_settings)