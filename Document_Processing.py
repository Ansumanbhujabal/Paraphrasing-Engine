import fitz  
import json
import os
from docx import Document




def extract_text_from_pdf(file_path):
    """Extract text from a PDF using PyMuPDF."""
    doc = fitz.open(file_path)
    text = ""
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)  # Load page by page
        text += page.get_text("text")  # Extract text
    return text

def extract_text_from_txt(file_path):
    """Extract text from a TXT file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def extract_text_from_docx(file_path):
    """Extract text from a DOCX file using python-docx."""
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_json(file_path):
    """Extract text from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return json.dumps(data, indent=4)  # Format the JSON content as text

def extract_text(file_path):
    """Main function to extract text from various document types."""
    _, file_extension = os.path.splitext(file_path)

    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        return extract_text_from_txt(file_path)
    elif file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.json':
        return extract_text_from_json(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")


# text=extract_text("/teamspace/studios/this_studio/Paraphrasing-Engine/pdf/Devops_AWS_Certificate.pdf")
# text=extract_text("/teamspace/studios/this_studio/Paraphrasing-Engine/pdf/2101109132intern.pdf")
# text=extract_text("/teamspace/studios/this_studio/Paraphrasing-Engine/pdf/Ansuman_ComputerScience_Graduate_Resume.pdf")
# text=extract_text("/teamspace/studios/this_studio/Paraphrasing-Engine/pdf/student policy .pdf")

# print(text)
