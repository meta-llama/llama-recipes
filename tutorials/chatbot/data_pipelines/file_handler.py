import os
import magic
from PyPDF2 import PdfReader
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_text_file(file_path):
    try:
        with open(file_path, 'r') as f:
            return f.read().strip() + ' '
    except Exception as e:
        logging.error(f"Error reading text file {file_path}: {e}")
    return ''

def read_pdf_file(file_path):
    try:
        with open(file_path, 'rb') as f:
            pdf_reader = PdfReader(f)
            num_pages = len(pdf_reader.pages)
            file_text = [pdf_reader.pages[page_num].extract_text().strip() + ' ' for page_num in range(num_pages)]
            return ''.join(file_text)
    except Exception as e:
        logging.error(f"Error reading PDF file {file_path}: {e}")
    return ''

def process_file(file_path):
    file_type = magic.from_file(file_path, mime=True)
    if file_type in ['text/plain', 'text/markdown']:
        return read_text_file(file_path)
    elif file_type == 'application/pdf':
        return read_pdf_file(file_path)
    else:
        logging.warning(f"Unsupported file type {file_type} for file {file_path}")
        return ''

def get_file_string(context):
    file_strings = []

    for root, _, files in os.walk(context['data_dir']):
        for file in files:
            file_path = os.path.join(root, file)
            file_text = process_file(file_path)
            if file_text:
                file_strings.append(file_text)

    return ' '.join(file_strings)

