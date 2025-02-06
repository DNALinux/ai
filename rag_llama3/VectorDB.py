import os
import time
import ollama
import chromadb
from pathlib import Path
from chromadb.config import Settings
import hashlib
import logging
from rag_llama3 import TextExtractor as te
import tempfile

class VectorDB:
    def __init__(self, chroma_db_dir: str, chroma_db_name: str, model="mxbai-embed-large"):
        self.chroma_db_dir = Path(chroma_db_dir)
        self.collection_name = chroma_db_name
        self.model = model
        # Load or create the Chroma vector database
        self.vector_db = self._load_or_create_vector_db()
        logging.basicConfig(level=logging.INFO)


    def _load_or_create_vector_db(self):
        """Load an existing Chroma vector database or create a new one."""
        # Ensure the directory exists
        logging.info("Loading or creating Chroma vector database...")
        self.chroma_db_dir.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(self.chroma_db_dir))
    
        collection = client.get_or_create_collection(self.collection_name)
        logging.info(f"Database '{self.collection_name}' loaded or created at '{self.chroma_db_dir}'.")

        return collection


    def load_data(self, input_dir: str = None, output_dir: str = None, urls_path: str = None):
        """
        Load and process data based on provided directories or URLs.

        - If `urls_path` is provided, loads URLs.
        - If `input_dir` is provided, loads PDFs and HTMLs.
        - `output_dir` is optional; if not provided, a temporary directory is used.
        """
        # Set default output_dir if not provided
        if output_dir is None:
            # Use a temporary directory as the default output directory
            output_dir = tempfile.mkdtemp()
            logging.info(f"No `output_dir` specified. Using temporary directory: {output_dir}")

        if urls_path:
            # Load data from URLs only
            self.load_url(urls_path, output_dir)
            
        elif input_dir:
            # Load data from PDFs and HTML files only
            self.raw_data = te.TextExtractor(input_dir=input_dir, output_dir=output_dir)
            self.load_pdf()
            self.load_html()
            
        else:
            logging.error("Invalid input: Provide `urls_path` or `input_dir`.")

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

    def load_url(self, urls_path, output_dir):
        """Load and process data from URLs."""
        logging.info("Loading data from URLs...")
        self.raw_data = te.TextExtractor(input_dir='', output_dir=output_dir, urls_file=urls_path)
        urls = self.raw_data.get_urls()
        sources = self.show_sources()
        for url in urls:
            logging.info(f"Processing URL: {url}")
            text = self.raw_data.crawl_and_extract(url)
            if url in sources:
                self.delete_source(url)
            self._process_and_add(text, [str(url)] * len(text))
        logging.info("Finished loading data from URLs.")
    
    def load_pdf(self):
        """Load and process data from PDFs."""
        logging.info("Loading data from PDFs...")
        pdfs = self.raw_data.get_pdf()
        sources = self.show_sources()
        for pdf in pdfs:    
            logging.info(f"Processing PDF: {pdf}")
            text = self.raw_data.extract_pdf_texts(pdf)
            source_name = Path(pdf).name.replace('.pdf', '')  # Remove the '.pdf' extension
            if source_name in sources:
                self.delete_source(source_name)
            self._process_and_add(text, [str(source_name)] * len(text))
        logging.info("Finished loading data from PDFs.")

    def load_html(self):
        """Load and process data from HTML files."""
        logging.info("Loading data from HTML files...")
        htmls = self.raw_data.get_html()
        sources = self.show_sources()
        for html in htmls:
            logging.info(f"Processing HTML: {html}")
            source_name = Path(html).name.replace('.html', '')  # Remove the '.pdf' extension
            text = self.raw_data.extract_html_text(html)
            if source_name in sources:
                self.delete_source(source_name)
            self._process_and_add(text, [str(source_name)] * len(text))
        logging.info("Finished loading data from HTML files.")

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

    def show_sources(self):
        """Show all sources in the vector database."""
        try:
            sources = self.vector_db.get(
                include=["metadatas"]
            )
            logging.info(f"Retrieved sources from the vector database.")
            return set([s["source"] for s in sources["metadatas"]])
        except Exception as e:
            logging.error(f"Error getting sources: {e}")

    def query_sources(self, source):
        """Query the vector database with a specific source."""
        try:
            response = self.vector_db.query(
                where={"source": source}
            )
            logging.info(f"Query results: {response}")
            return response["documents"]
        except Exception as e:
            logging.error(f"Error querying the vector database: {e}")
            return []
        
    def query(self, query_text, k=5):
        """Query the vector database with a text prompt."""
        try:
            response = self.vector_db.query(
                query_embeddings=ollama.embeddings(prompt=query_text, model=self.model)['embedding'],
                n_results=k
            )
            logging.info(f"Query results: {response}")
            return response["documents"][0]
        except Exception as e:
            logging.error(f"Error querying the vector database: {e}")
            return []


    def clear_database(self):
        """Clear the vector database."""
        try:
            sources = self.show_sources()
            for source in sources:
                self.delete_source(source)
            logging.info("Cleared the vector database.")
        except Exception as e:
            logging.error(f"Error clearing the vector database: {e}")
