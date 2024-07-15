import os
from pathlib import Path
import requests
import math
import hashlib
from bs4 import BeautifulSoup
import certifi
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


def is_relevant_link(link, base_url):
    """Determine if a link is relevant based on simple heuristics."""
    if link.startswith('/'):
        return True
    if base_url in link:
        return True
    
    keywords = ['chapter', 'article', 'content', 'tutorial', 'guide']
    return any(keyword in link for keyword in keywords)

def extract_content(url, visited):
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

    return '\n'.join(grouped_text), [link['href'] for link in soup.find_all('a', href=True) if is_relevant_link(link['href'], url)]

def save_content_to_files(text, directory, chunk_size, overlap_size, base_url, depth):
    """Save the extracted text into files of a specified chunk size with overlap."""
    os.makedirs(directory, exist_ok=True)
    num_files = math.ceil(len(text) / (chunk_size - overlap_size))

    # Generate a base file name using a hash of the URL
    safe_url = hashlib.md5(base_url.encode()).hexdigest()
    
    for i in range(num_files):
        start_index = i * (chunk_size - overlap_size)
        end_index = start_index + chunk_size

        # Ensure we don't go out of bounds
        chunk_text = text[start_index:end_index].strip()
        
        # Generate a unique file name
        file_name = f"{safe_url}_d{depth}_p{i+1}.txt"
        file_path = os.path.join(directory, file_name)

        # Save chunk of text into a text file at the specified location
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(chunk_text)

        print(f"Part {i+1} of text from {base_url} has been saved to {file_path}")

def crawl_and_extract(url, directory, chunk_size=200, overlap_size=50, max_depth=1, depth=0, visited=None):
    """Crawl the website and extract content recursively."""
    if visited is None:
        visited = set()

    if depth > max_depth or url in visited:
        return

    # Extract content from the current URL
    text, links = extract_content(url, visited)

    # Save extracted content to files
    save_content_to_files(text, directory, chunk_size, overlap_size, url, depth)

    # Recursively crawl links
    for link in links:
        if link.startswith('/'):
            link = f"{url}{link}"
        elif not link.startswith('http'):
            link = f"{url}/{link}"

        if link not in visited:
            crawl_and_extract(link, directory, chunk_size, overlap_size, max_depth, depth + 1, visited)
