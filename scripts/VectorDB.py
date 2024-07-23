import os
import time
import ollama
import chromadb
from pathlib import Path
from chromadb.config import Settings
import hashlib
import sys
import logging
sys.path.append("/home/tagore/repos/ai/scripts")
import TextExtractor as te

class VectorDB:
    def __init__(self, input_dir: str, output_dir: str , chroma_db_dir: str, chroma_db_name: str, model="mxbai-embed-large"):
        self.raw_data = te.TextExtractor(input_dir, output_dir)
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name = chroma_db_name
        self.model = model
        self.vector_db = self._load_or_create_vector_db()  # Load or create the Chroma vector database
        logging.basicConfig(level=logging.INFO)
        
    def _load_or_create_vector_db(self):
        """Load an existing Chroma vector database or create a new one."""
        # Ensure the directory exists
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
    
        collection = client.get_or_create_collection(self.collection_name)

        return collection
    
    def _get_embeddings(self, texts):
        """Get embeddings for a list of texts from Ollama."""
        try:
            response = [ollama.embeddings(prompt=t, model=self.model) for t in texts]
            return [r['embedding'] for r in response]
        except Exception as e:
            logging.error(f"Error obtaining embeddings: {e}")
            return []
    
    def _process_and_add(self, texts, sources):
        """Process texts and add them to the vector database."""
        embeddings = self._get_embeddings(texts)
        ids = [hashlib.md5(t.encode()).hexdigest() for t in texts]
        metadatas = [{"source": src} for src in sources]
        try:
            self.vector_db.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logging.info(f"Added {len(texts)} documents to the vector database.")
        except Exception as e:
            logging.error(f"Error adding data to vector database: {e}")

    def load_url(self):
        """Load and process data from URLs."""
        urls = self.raw_data.get_urls()
        for url in urls:
            text = self.raw_data.crawl_and_extract(url)
            self._process_and_add(text, [str(url)] * len(text))
    
    def load_pdf(self):
        """Load and process data from PDFs."""
        pdfs = self.raw_data.get_pdf()
        for pdf in pdfs:
            text = self.raw_data.extract_pdf_texts(pdf)
            source_name = Path(pdf).name.replace('.pdf', '')  # Remove the '.pdf' extension
            self._process_and_add(text, [str(source_name)] * len(text))

    def load_html(self):
        """Load and process data from HTML files."""
        htmls = self.raw_data.get_html()
        for html in htmls:
            source_name = Path(html).name.replace('.html', '')  # Remove the '.pdf' extension
            text = self.raw_data.extract_html_text(html)
            self._process_and_add(text, [str(source_name)] * len(text))

    def peek(self):
        """Retrieve first 10 documents from the vector database."""
        try:
            results = self.vector_db.peek()
            logging.info(f"Retrieved {len(results)} documents from the vector database.")
            return results
        except Exception as e:
            logging.error(f"Error querying documents: {e}")
            return []

    def delete_source(self, source):
        """Delete all documents from a specific source."""
        try:
            self.vector_db.delete(
                where={"source": source}
                )
            logging.info(f"Deleted documents from source: {source}")
        except Exception as e:
            logging.error(f"Error deleting documents: {e}")
