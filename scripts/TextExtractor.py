import os
from pathlib import Path
import requests
import math
import numpy as np
import hashlib
from bs4 import BeautifulSoup
import certifi
from langchain_community.document_loaders import PyPDFLoader
import logging

class TextExtractor:
    def __init__(self,input_dir: str, output_dir: str, urls_file: str = 'urls.txt'):
        self.output_dir = Path(output_dir)
        self.input_dir = Path(input_dir)
        self.urls_file = Path(urls_file)
        if not self.urls_file.exists():
            with open(self.urls_file, 'w', encoding='utf-8') as file:
                pass  # Just create an empty file
        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
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
        try:
            with open(self.urls_file, 'r') as f:  # Use self.urls_file directly
                urls = [line.strip() for line in f]
        except FileNotFoundError:
            logging.error(f"File not found: {self.urls_file}. Make sure the file exists.")
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

    def extract_pdf_texts(self, pdf_path: str) -> list:
        """
        Extracts text from a PDF file and add them all to a list,
        ensuring a limited overlap between pages and avoiding pages with only one line of text.

        Args:
        - pdf_path (str): Path to the PDF file.
        """
        loader = PyPDFLoader(pdf_path)
        pages = loader.load_and_split()
        
        # Extract page contents
        page_texts = [page.page_content for page in pages]
        
        # Filter out pages with only one line of text
        valid_texts = [text for text in page_texts if len(text.splitlines()) > 1]

        # Initialize the result list
        result = []
        
        # Convert the list to a NumPy array for vectorized operations
        valid_texts = np.array(valid_texts)
        
        if len(valid_texts) > 0:
            # Handle the first page
            result.append(valid_texts[0])
        
            # Create overlaps for subsequent pages
            overlaps = np.char.str_slice(valid_texts[:-1], -100, None)  # Get the last 100 characters of each page
            combined_texts = np.char.add(overlaps, valid_texts[1:])
            
            # Append combined texts to the result list
            result.extend(combined_texts.tolist())
        
        return result
    
    def save_a_html_text(self, html_path: str, chars_per_file: int = 500, overlap: int = 100) -> None:
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

    def extract_html_text(self, html_path: str, chars_per_file: int = 500, overlap: int = 100) -> list:
        """
        Extracts text from a HTML file and returns it as a string.

        Args:
        - html_path (str): Path to the HTML file.
        - chars_per_file (int): Number of characters per chunk.
        - overlap (int): Number of overlapping characters between chunks.

        Returns:
        - list: List of extracted text content.
        """
        with open(html_path, 'r', encoding='utf-8') as f:
            html_content = f.read()
            soup = BeautifulSoup(html_content, 'html.parser')
            text = soup.get_text()
            text = text.replace('\n', ' ').replace('\r', '')
            text = ' '.join(text.split())

        # Calculate the start indices for the chunks
        start_indices = np.arange(0, len(text), chars_per_file)
        
        # Ensure overlap for chunks after the first
        chunks = [text[max(0, start - overlap):start + chars_per_file] for start in start_indices]
    
        return chunks
        
    def is_relevant_link(self, link, base_url):
        """Determine if a link is relevant based on simple heuristics."""
        if link.startswith('/'):
            return True
        if base_url in link:
            return True
        
        keywords = ['chapter', 'article', 'content', 'tutorial', 'guide', 'section', 'blast']
        return any(keyword in link for keyword in keywords)

    def extract_url_text(self, url, visited):
        """Extract headings and paragraphs from a single URL."""
        visited.add(url)
        response = requests.get(url, verify=certifi.where())
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract headings and paragraphs
        elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p'])
        grouped_text = []
        buffer = ""

        for element in elements:
            text = element.get_text().strip()
            if text:  # Skip empty lines
                if element.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    if buffer:
                        grouped_text.append(buffer)
                        buffer = ""
                    buffer += text + "\n"
                else:
                    buffer += text + "\n"
                    grouped_text.append(buffer)
                    buffer = ""

        if buffer:
            grouped_text.append(buffer)

        return '\n'.join(grouped_text), [link['href'] for link in soup.find_all('a', href=True) if self.is_relevant_link(link['href'], url)]

    def save_chunks_to_files(self, chunks, directory, base_name):
        """Save chunks to files in the specified directory."""
        os.makedirs(directory, exist_ok=True)
        for i, chunk in enumerate(chunks):
            file_name = f"{base_name}_part_{i+1}.txt"
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(chunk)
            print(f"Saved part {i+1} to {file_path}")

    def split_text_into_chunks(self, text, chunk_size, overlap_size):
        """Split the text into chunks with overlap."""
        num_chunks = math.ceil(len(text) / (chunk_size - overlap_size))
        start_indices = np.arange(num_chunks) * (chunk_size - overlap_size)
        end_indices = start_indices + chunk_size

        chunks = [text[start:end].strip() for start, end in zip(start_indices, end_indices)]
        
        return chunks

    def get_main_domain(self, url):
        """Extract the main domain from a URL."""
        if url.startswith('http://'):
            url = url[7:]
        elif url.startswith('https://'):
            url = url[8:]

        domain = url.split('/')[0]
        return domain
    
    def crawl_and_save(self, url, chunk_size=500, overlap_size=50, max_depth=1, depth=0, visited=None):
        """
        Crawl the website and save content recursively.
        """
        if visited is None:
            visited = set()

        if depth > max_depth or url in visited:
            return

        # Extract content from the current URL
        text, links = self.extract_url_text(url, visited)

        # Create a directory for the main domain
        main_domain = self.get_main_domain(url)
        directory = os.path.join(self.output_dir, main_domain)

        # Split text into chunks with overlap
        chunks = self.split_text_into_chunks(text, chunk_size, overlap_size)

        # Save chunks to files
        safe_base_name = hashlib.md5(text.encode()).hexdigest()
        self.save_chunks_to_files(chunks, directory, safe_base_name)

        # Recursively crawl links
        for link in links:
            if link.startswith('/'):
                link = f"{url}{link}"
            elif not link.startswith('http'):
                link = f"{url}/{link}"

            if link not in visited:
                self.crawl_and_save(link, chunk_size, overlap_size, max_depth, depth + 1, visited)

    def crawl_and_extract(self, url, chunk_size=500, overlap_size=50, max_depth=1, depth=0, visited=None):
        """
        Crawl the website and extract content recursively.

        Args:
        - url (str): The starting URL for crawling.
        - chunk_size (int): The number of characters per chunk.
        - overlap_size (int): The number of overlapping characters between chunks.
        - max_depth (int): The maximum depth of recursion.
        - depth (int): The current depth of recursion.
        - visited (set): The set of visited URLs.

        Returns:
        - list: A list of text chunks extracted from the website.
        """
        if visited is None:
            visited = set()

        if depth > max_depth or url in visited:
            return []

        # Extract content from the current URL
        text, links = self.extract_url_text(url, visited)

        # Split text into chunks with overlap
        chunks = self.split_text_into_chunks(text, chunk_size, overlap_size)

        # Recursively crawl links and add their chunks
        for link in links:
            if link.startswith('/'):
                link = f"{url}{link}"
            elif not link.startswith('http'):
                link = f"{url}/{link}"

            if link not in visited:
                chunks += self.crawl_and_extract(link, chunk_size, overlap_size, max_depth, depth + 1, visited)

        chunks = set(chunks)  # Remove duplicates
        
        return list(chunks)