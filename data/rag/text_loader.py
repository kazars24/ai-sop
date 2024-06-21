import json
import csv
import xml.etree.ElementTree as ET
import os
import logging as log
from pathlib import Path

class JSONData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath, read_top_lines=False):
        filepath = Path(filepath)
        log.info(f'[Preprocess][JSON] Reading {filepath}')
        with open(filepath, 'r') as f:
            content = json.load(f)
            if read_top_lines:
                content_str = json.dumps(content, indent=2)
                lines = content_str.split('\n')
                top_lines = '\n'.join(lines[:10])
                self.data.append(f'<FILE {filepath.name} content BEGIN>\n{top_lines}\n...\n<FILE {filepath.name} content END>\n')
            else:
                self.data.append(f'<FILE {filepath.name} content BEGIN>\n{json.dumps(content, indent=2)}\n<FILE {filepath.name} content END>\n')

    def add_files_recursively(self, path, read_top_lines=False):
        path = Path(path)
        if path.is_file():
            if path.suffix == '.json':
                self.add_file(path, read_top_lines)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.json'):
                        self.add_file(os.path.join(root, file), read_top_lines)

    def get_text(self):
        return '\n'.join(self.data)

class XMLData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath, read_top_lines=False):
        filepath = Path(filepath)
        log.info(f'[Preprocess][XML] Reading {filepath}')
        tree = ET.parse(filepath)
        root = tree.getroot()
        content = ET.tostring(root, encoding='unicode')
        if read_top_lines:
            lines = content.split('\n')
            top_lines = '\n'.join(lines[:10])
            self.data.append(f'<FILE {filepath.name} content BEGIN>\n{top_lines}\n...\n<FILE {filepath.name} content END>\n')
        else:
            self.data.append(f'<FILE {filepath.name} content BEGIN>\n{content}\n<FILE {filepath.name} content END>\n')

    def add_files_recursively(self, path, read_top_lines=False):
        path = Path(path)
        if path.is_file():
            if path.suffix == '.xml':
                self.add_file(path, read_top_lines)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.xml'):
                        self.add_file(os.path.join(root, file), read_top_lines)

    def get_text(self):
        return '\n'.join(self.data)

class TextData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath, read_top_lines=False):
        filepath = Path(filepath)
        log.info(f'[Preprocess][TXT Common] Reading {filepath}')
        with open(filepath, 'r') as f:
            content = f.read()
            if read_top_lines:
                lines = content.split('\n')
                top_lines = '\n'.join(lines[:10])
                self.data.append(f'<FILE {filepath.name} content BEGIN>\n{top_lines}\n...\n<FILE {filepath.name} content END>\n')
            else:
                self.data.append(f'<FILE {filepath.name} content BEGIN>\n{content}\n<FILE {filepath.name} content END>\n')

    def add_files_recursively(self, path, read_top_lines=False):
        path = Path(path)
        if path.is_file():
            if path.suffix == '.txt':
                self.add_file(path, read_top_lines)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.txt'):
                        self.add_file(os.path.join(root, file), read_top_lines)

    def get_text(self):
        return '\n'.join(self.data)

class CSVData:
    def __init__(self):
        self.data = []

    def add_file(self, filepath, read_top_lines=False):
        filepath = Path(filepath)
        log.info(f'[Preprocess][CSV] Reading {filepath}')
        with open(filepath, 'r') as f:
            reader = csv.reader(f)
            if read_top_lines:
                rows = list(reader)
                top_rows = rows[:10]
                content = '\n'.join([','.join(row) for row in top_rows])
                self.data.append(f'<FILE {filepath.name} content BEGIN>\n{content}\n...\n<FILE {filepath.name} content END>\n')
            else:
                content = '\n'.join([','.join(row) for row in reader])
                self.data.append(f'<FILE {filepath.name} content BEGIN>\n{content}\n<FILE {filepath.name} content END>\n')

    def add_files_recursively(self, path, read_top_lines=False):
        path = Path(path)
        if path.is_file():
            if path.suffix == '.csv':
                self.add_file(path, read_top_lines)
        else:
            for root, _, files in os.walk(path):
                for file in files:
                    if file.endswith('.csv'):
                        self.add_file(os.path.join(root, file), read_top_lines)

    def get_text(self):
        return '\n'.join(self.data)

def data_loader(data_path, read_top_lines=False):
    json_data = JSONData()
    xml_data = XMLData()
    text_data = TextData()
    csv_data = CSVData()

    json_data.add_files_recursively(data_path, read_top_lines)
    xml_data.add_files_recursively(data_path, read_top_lines)
    text_data.add_files_recursively(data_path, read_top_lines)
    csv_data.add_files_recursively(data_path, read_top_lines)

    return json_data, xml_data, text_data, csv_data
