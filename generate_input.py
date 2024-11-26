
import xml.etree.ElementTree as ET
from xml.dom import minidom

# Define the constants for your simulation

# General settings
general_settings = {
    'outputfile': 'test.xmso',
    'n_photons_interval': '10000',
    'n_photons_line': '1000000',
    'n_interactions_trajectory': '4',
    'comments': ''
}

# Composition layers (single layer)
composition_layers = [
    {
        'elements': [
            {'atomic_number': '11', 'weight_fraction': '5'},    # Na
            {'atomic_number': '12', 'weight_fraction': '10'},   # Mg
            {'atomic_number': '13', 'weight_fraction': '15'},   # Al
            {'atomic_number': '14', 'weight_fraction': '20'},   # Si
            {'atomic_number': '20', 'weight_fraction': '10'},   # Ca
            {'atomic_number': '22', 'weight_fraction': '5'},    # Ti
            {'atomic_number': '26', 'weight_fraction': '30'},   # Fe
            {'atomic_number': '8', 'weight_fraction': '5'},     # O
        ],
        'density': '2.5',    # Adjust as appropriate
        'thickness': '1'     # Adjust as appropriate
    }
]

reference_layer = '1'  # Since there's only one layer

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

# Excitation settings
excitation_settings = {
    'energy': '20',
    'horizontal_intensity': '1e+09',
    'vertical_intensity': '1e+09',
    'sigma_x': '0',
    'sigma_xp': '0',
    'sigma_y': '0',
    'sigma_yp': '0'
}

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

# Function to build the XML tree
def build_xml_tree():
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
    discrete = ET.SubElement(excitation, 'discrete')
    for key, value in excitation_settings.items():
        elem = ET.SubElement(discrete, key)
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

# Function to write the XML tree to a file with DOCTYPE declaration
def write_xml_file(filename):
    root = build_xml_tree()
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
    write_xml_file('input.xmsi')
