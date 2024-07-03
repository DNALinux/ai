import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader

def save_text_to_file(text, output_dir, filename):
    """
    Saves text to a file in the specified directory.

    Args:
    - text (str): Text content to be saved.
    - output_dir (str): Directory path where the file will be saved.
    - filename (str): Name of the file to be saved.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_file = output_dir / filename
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

def extract_and_save_pdf_text(pdf_path, output_dir):
    """
    Extracts text from a PDF file and saves each page's text to a separate file.

    Args:
    - pdf_path (str): Path to the PDF file.
    - output_dir (str): Directory path where the text files will be saved.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load_and_split()

    for page_num, page_text in enumerate(pages, start=1):
        filename = f"page_{page_num}.txt"
        save_text_to_file(page_text.page_content[:], output_dir, filename)  # Ensure page_text.text is a string
        print(f"Saved page {page_num} to {output_dir}/{filename}")

