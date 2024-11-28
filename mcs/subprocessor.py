import os
import subprocess
from pathlib import Path
import xml.etree.ElementTree as ET

class XMIMSIMProcessor:
    def __init__(self):
        xmimsimdata_path = Path('xmimsimdata.h5')
        if not xmimsimdata_path.exists():  
            subprocess.run(['xmimsim-db'], check=True)

    def process_directory(self, input_path: str, output_path: str = None):
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
                    self._process_file(file_path)

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

    def _process_file(self, file_path: Path) -> None:
        subprocess.run(['xmimsim', str(file_path)], check=True)

def main():
    processor = XMIMSIMProcessor()
    processor.process_directory(
        'xmsi_inputs',
        'xmso_outputs'
    )

if __name__ == "__main__":
    main()