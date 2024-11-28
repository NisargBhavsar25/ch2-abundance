import pandas as pd
import xml.etree.ElementTree as ET
from xml.dom import minidom
import os
import subprocess
from pathlib import Path
import argparse

errors = []

class XMIMSIMGenerator:
    def __init__(self, csv_path, excitation_file=None):
        self.df = pd.read_csv(csv_path)
        self.excitation_settings = self._process_excitation_file(excitation_file)
        
    def _process_excitation_file(self, filepath):
        result = []
        with open(filepath, 'r') as file:
            for line in file:
                cols = [col for col in line.strip().split() if col]
                result.append({
                    'energy': cols[0],
                    'horizontal_intensity': cols[2],
                    'vertical_intensity': cols[2],
                    'sigma_x': '0',
                    'sigma_xp': '0',
                    'sigma_y': '0',
                    'sigma_yp': '0'
                })
        return result

    def _create_composition_layer(self, row):
        return [{
            'elements': [
                # {'atomic_number': '11', 'weight_fraction': str(row['na'])},
                {'atomic_number': '12', 'weight_fraction': str(row['mg'])},
                {'atomic_number': '13', 'weight_fraction': str(row['al'])},
                {'atomic_number': '14', 'weight_fraction': str(row['si'])},
                {'atomic_number': '20', 'weight_fraction': str(row['ca'])},
                {'atomic_number': '22', 'weight_fraction': str(row['ti'])},
                {'atomic_number': '26', 'weight_fraction': str(row['fe'])},
                {'atomic_number': '8', 'weight_fraction': str(row['o'])},
            ],
            'density': '2.5',
            'thickness': '1'
        }]

    def generate_xml_files(self, output_dir='xmsi_inputs'):
        os.makedirs(output_dir, exist_ok=True)
        
        for index, row in self.df.iterrows():
            root = self._build_xml_tree(self._create_composition_layer(row))
            
            filename = os.path.join(output_dir, f'input_{index+1}.xmsi')
            self._write_xml_file(filename, root)
        
        return output_dir

    def _build_xml_tree(self, composition_layers):
        root = ET.Element('xmimsim')
        
        # Add sections (general, composition, geometry, excitation, absorbers, detector)
        self._add_general_section(root)
        self._add_composition_section(root, composition_layers)
        self._add_geometry_section(root)
        self._add_excitation_section(root)
        self._add_absorbers_section(root)
        self._add_detector_section(root)
        
        return root

    def _write_xml_file(self, filename, root):
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
        print(f'{filename} written')

    # Add helper methods for each XML section
    def _add_general_section(self, root):
        general = ET.SubElement(root, 'general', version='1.0')
        settings = {
            'outputfile': 'test.xmso',
            'n_photons_interval': '100',
            'n_photons_line': '10000',
            'n_interactions_trajectory': '4',
            'comments': ''
        }
        for key, value in settings.items():
            elem = ET.SubElement(general, key)
            elem.text = value

    def _add_composition_section(self, root, composition_layers):
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

    def _add_geometry_section(self, root):
        geometry = ET.SubElement(root, 'geometry')
        settings = {
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
        for key, value in settings.items():
            if isinstance(value, dict):
                sub_elem = ET.SubElement(geometry, key)
                for sub_key, sub_value in value.items():
                    coord = ET.SubElement(sub_elem, sub_key)
                    coord.text = sub_value
            else:
                elem = ET.SubElement(geometry, key)
                elem.text = value

    def _add_excitation_section(self, root):
        excitation = ET.SubElement(root, 'excitation')
        for setting in self.excitation_settings:
            continuous = ET.SubElement(excitation, 'continuous')
            for key, value in setting.items():
                elem = ET.SubElement(continuous, key)
                elem.text = value

    def _add_absorbers_section(self, root):
        absorbers = ET.SubElement(root, 'absorbers')
        detector_path = ET.SubElement(absorbers, 'detector_path')
        layer = ET.SubElement(detector_path, 'layer')
        element = ET.SubElement(layer, 'element')
        atomic_number = ET.SubElement(element, 'atomic_number')
        atomic_number.text = '4'
        weight_fraction = ET.SubElement(element, 'weight_fraction')
        weight_fraction.text = '100'
        density = ET.SubElement(layer, 'density')
        density.text = '1.85'
        thickness = ET.SubElement(layer, 'thickness')
        thickness.text = '0.002'

    def _add_detector_section(self, root):
        detector = ET.SubElement(root, 'detector')
        settings = {
            'detector_type': 'SiLi',
            'live_time': '1',
            'pulse_width': '1e-05',
            'nchannels': '2048',
            'gain': '0.02',
            'zero': '0',
            'fano': '0.12',
            'noise': '0.1'
        }
        for key, value in settings.items():
            elem = ET.SubElement(detector, key)
            elem.text = value
        
        crystal = ET.SubElement(detector, 'crystal')
        layer = ET.SubElement(crystal, 'layer')
        element = ET.SubElement(layer, 'element')
        atomic_number = ET.SubElement(element, 'atomic_number')
        atomic_number.text = '14'
        weight_fraction = ET.SubElement(element, 'weight_fraction')
        weight_fraction.text = '100'
        density = ET.SubElement(layer, 'density')
        density.text = '2.33'
        thickness = ET.SubElement(layer, 'thickness')
        thickness.text = '0.5'

class XMIMSIMProcessor:
    def __init__(self):
        if not Path('xmimsimdata.h5').exists():
            subprocess.run(['xmimsim-db'], check=True)

    def process_directory(self, input_path, output_path='xmso_outputs'):
        root_path = Path(input_path)
        output_path = Path(output_path) if output_path else None
        
        if output_path and not output_path.exists():
            output_path.mkdir(parents=True, exist_ok=True)
        
        
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.lower().endswith('.xmsi'):
                    file_path = Path(root) / file
                    rel_path = file_path.relative_to(root_path)
                    new_output_path = output_path / rel_path.with_suffix('.xmso')

                    new_output_path.parent.mkdir(parents=True, exist_ok=True)
                    self._modify_output_path(file_path, str(new_output_path))
                    try:
                        subprocess.run(['xmimsim', str(file_path), '-v'], check=True)
                    except Exception as e:
                        errors.append(f"{e} in {file_path}")

    def _modify_output_path(self, file_path: Path, new_output_path: str) -> None:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        root_index = -1
        for i, line in enumerate(lines):
            if line.strip().startswith('<xmimsim'):
                root_index = i
                break
        
        header_lines = lines[:root_index]
        xml_content = ''.join(lines[root_index:])
        
        root = ET.fromstring(xml_content)
        general = root.find('general')
        output_path_elem = general.find('outputfile')
        output_path_elem.text = new_output_path
        
        modified_xml_content = ET.tostring(root, encoding='unicode')
        modified_lines = header_lines + [modified_xml_content]
        
        with open(file_path, 'w') as file:
            file.writelines(modified_lines)

def main():
    parser = argparse.ArgumentParser(description='Process CSV data through XMIMSIM')
    parser.add_argument('csv_file', help='Input CSV file')
    parser.add_argument('excitation_file', nargs='?', default='ch2_xsm_20221003_l1_vvapec.txt', help='Excitation data file')
    args = parser.parse_args()

    generator = XMIMSIMGenerator(args.csv_file, args.excitation_file)
    xml_dir = generator.generate_xml_files()
    
    processor = XMIMSIMProcessor()
    processor.process_directory(xml_dir)
    print(errors)

if __name__ == "__main__":
    main()