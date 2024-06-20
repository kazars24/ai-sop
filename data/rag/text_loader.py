import json
import xml.etree.ElementTree as ET
import os
import logging as log

class JSONData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath):
        log.info(f'[Preprocess][JSON] Reading {filepath}')
        with open(filepath, 'r') as f:
            self.data.append(json.load(f))

    def get_text(self):
        return '\n'.join([str(d) for d in self.data])

class XMLData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath):
        log.info(f'[Preprocess][XML] Reading {filepath}')
        tree = ET.parse(filepath)
        root = tree.getroot()
        self.data.append(ET.tostring(root, encoding='unicode'))

    def get_text(self):
        return '\n'.join(self.data)

class TextData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath):
        log.info(f'[Preprocess][TXT Common] Reading {filepath}')
        with open(filepath, 'r') as f:
            self.data.append(f.read())

    def get_text(self):
        return '\n'.join(self.data)

def data_loader(raw_data_folder):
    json_data = JSONData()
    xml_data = XMLData()
    text_data = TextData()

    for filename in os.listdir(raw_data_folder):
        if filename.endswith('.json'):
            json_data.add_file(os.path.join(raw_data_folder, filename))
        elif filename.endswith('.xml'):
            xml_data.add_file(os.path.join(raw_data_folder, filename))
        else:
            text_data.add_file(os.path.join(raw_data_folder, filename))
    return json_data, xml_data, text_data
