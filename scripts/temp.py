import os
from pathlib import Path
import requests
import math
import hashlib
from bs4 import BeautifulSoup
import certifi
from langchain_community.document_loaders import PyPDFLoader

class TextExtractor:
    def __init__(self, output_dir: str, input_dir: str, chroma_db_dir: str):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.chroma_db_path = Path(chroma_db_dir)

        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.chroma_db_path.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_pdf(self) -> list:
        pdf_file_paths = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.pdf'):
                    pdf_file_paths.append(Path(root) / file)
        return pdf_file_paths
    
    def get_html(self) -> list:
        html_file_paths = []
        for root, _, files in os.walk(self.input_dir):
            for file in files:
                if file.endswith('.html'):
                    html_file_paths.append(Path(root) / file)
        return html_file_paths
    
    def get_urls(self) -> list:
        urls = []
        with open(self.input_dir / 'urls.txt', 'r') as f:
            urls = [line.strip() for line in f]
        return urls
    
    def save_text_to_file(self, text: str, filename: str) -> None:
        """
        Saves text to a file in the specified directory.

        Args:
        - text (str): Text content to be saved.
        - filename (str): Name of the file to be saved.
        """
        output_file = self.output_dir / filename
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(text)

    def save_a_pdf_text(self, pdf_path: str) -> None:
        """
        Extracts text from a PDF file and saves each page's text to a separate file,
        ensuring a limited overlap between pages and avoiding pages with only one line of text.

        Args:
        - pdf_path (str): Path to the PDF file.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Extract the source name and create a directory for it
        source_name = Path(pdf_path).name.replace('.pdf', '')  # Remove the '.pdf' extension
        source_dir = self.output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        previous_page_text = ""  # Initialize previous page text

        for page_num, page_text in enumerate(pages, start=1):
            current_page_text = page_text.page_content

            # Check if the current page has more than one line of text
            if len(current_page_text.splitlines()) > 1:
                # Add a limited overlap from the previous page if it's not the first page
                if page_num > 1:
                    overlap = "\n" + previous_page_text[-100:]  # Adjust the number of characters to overlap
                    combined_text = overlap + current_page_text
                else:
                    combined_text = current_page_text
                
                filename = f"{source_name}_page_{page_num}.txt"
                output_file = source_dir / filename  # Save the file in the source directory
                self.save_text_to_file(combined_text, output_file)  # Pass the combined text for saving

                print(f"Saved page {page_num} to {output_file}")
            
            previous_page_text = current_page_text  # Update previous page text for the next iteration

    
    def save_pdfs_texts(self, pdf_paths: list) -> None:
        """
        Extracts text from a list of PDF files.

        Args:
        - pdf_paths (list): List of paths to PDF files.
        """
        for pdf_path in pdf_paths:
            self.save_a_pdf_text(pdf_path)

    def delete_oneline_files(self) -> None:
        """
        Deletes files that have only one line of text in the output directory and its subdirectories.
        """
        for root, _, files in os.walk(self.output_dir):
            for file in files:
                file_path = Path(root) / file  # Construct the full file path
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) == 1:
                        os.remove(file_path)
                        print(f"Deleted {file_path}")

    def extract_pdf_texts(sef, pdf_path: str) -> None:
        """
        Extracts text from a PDF file and add them  all to a list,
        ensuring a limited overlap between pages and avoiding pages with only one line of text.

        Args:
        - pdf_path (str): Path to the PDF file.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        result = []
        previous_page_text = ""  # Initialize previous page text

        for page_num, page_text in enumerate(pages, start=1):
            current_page_text = page_text.page_content

            # Check if the current page has more than one line of text
            if len(current_page_text.splitlines()) > 1:
                # Add a limited overlap from the previous page if it's not the first page
                if page_num > 1:
                    overlap = "\n" + previous_page_text[-100:]  # Adjust the number of characters to overlap
                    combined_text = overlap + current_page_text
                else:
                    combined_text = current_page_text

                result.append(combined_text)

            previous_page_text = current_page_text  # Update previous page text for the next iteration

        return result
    
    def save_a_html_text(self, html_path: str, chars_per_file: int = 1500, overlap: int = 100) -> None:
        """
        Extracts text from a HTML file and saves it to multiple files with specified character limits and overlap.

        Args:
        - html_path (str): Path to the HTML file.
        - chars_per_file (int): Number of characters per output file.
        - overlap (int): Number of overlapping characters between consecutive files.
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\r', '')
            text = ' '.join(text.split())  # Remove extra spaces

        # Extract the source name and create a directory for it
        source_name = Path(html_path).stem  # Remove the file extension
        source_dir = self.output_dir / source_name
        source_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist

        start = 0
        file_num = 1
        while start < len(text):
            end = start + chars_per_file
            chunk = text[start:end]

            # Add overlap if it's not the first file
            if start > 0:
                chunk = text[start - overlap:end]

            filename = f"{source_name}_part_{file_num}.txt"
            output_file = source_dir / filename
            self.save_text_to_file(chunk, output_file)
            print(f"Saved part {file_num} to {output_file}")

            start += chars_per_file  # Move to the next chunk
            file_num += 1

    def save_html_texts(self, html_paths: list) -> None:
        """
        Extracts text from a list of HTML files and saves them to multiple files.

        Args:
        - html_paths (list): List of paths to HTML files.
        """
        for html_path in html_paths:
            self.save_a_html_text(html_path)

    def extract_html_text(self, html_path: str, chars_per_file: int = 1500, overlap: int = 100) -> list:
        """
        Extracts text from a HTML file and returns it as a string.

        Args:
        - html_path (str): Path to the HTML file.

        Returns:
        - list: list of extracted text content.
        """
        result = []

        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\r', '')
            text = ' '.join(text.split())

        start = 0
        file_num = 1
        while start < len(text):
            end = start + chars_per_file
            chunk = text[start:end]

            # Add overlap if it's not the first file
            if start > 0:
                chunk = text[start - overlap:end]

            result.append(chunk)

            start += chars_per_file

        return result
        