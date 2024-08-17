import argparse
import logging
import ollama  # Ensure the Ollama library is imported
from rag_llama3.TextExtractor import TextExtractor as te
from rag_llama3.VectorDB import VectorDB as vdb
import yaml
import sys

class RAG:
    def __init__(self, input_dir: str, output_dir: str, urls_file:str, chroma_db_dir: str, chroma_db_name: str, model="mxbai-embed-large"):
        self.vector_db = self._setup_vector_db(input_dir, output_dir, urls_file, chroma_db_dir, chroma_db_name, model)
        logging.basicConfig(level=logging.INFO)
    
    def _setup_vector_db(self, input_dir, output_dir, urls_file, chroma_db_dir, chroma_db_name, model):
        """Check if the database exists and set up if not."""
        try:
            # Attempt to load existing vector database
            vector_db = vdb(input_dir, output_dir, urls_file, chroma_db_dir, chroma_db_name, model)
            # Check if the database is empty or needs updating
            if not self._is_database_populated(vector_db):
                print("Database is not populated. Loading data...")
                logging.info("Database is not populated. Loading data...")
                vector_db.load_data()
            return vector_db
        except Exception as e:
            logging.error(f"Error setting up VectorDB: {e}")
            raise

    def _is_database_populated(self, vector_db):
        """Check if the vector database has data."""
        return len(vector_db.peek()) > 0
    
    def generate_prompt(self, question, context):
        template = """You need to answer questions about specific software.
        Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
        keep the answer concise.
        user
        Question: {question} 
        Context: {context} 
        Do not say according to the text. just give the answer, no comment."""
        return template.format(question=question, context=context)

    def generate_answer(self, query_text, k=4):
        """Generate an answer using the vector database and Ollama model."""
        try:
            output = ollama.generate(
                    model="llama3",
                    prompt= self.generate_prompt(query_text, self.vector_db.query(query_text, k)),
                )
            return output['response']
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "Error generating answer."
    
    def stream_answer(self, query_text, k=4):
        """Generate an answer using the vector database and Ollama model with streaming output."""
        try:
            prompt = self.generate_prompt(query_text, self.vector_db.query(query_text, k))
            
            # Generate text with streaming enabled
            output_stream = ollama.generate(model="llama3", prompt=prompt, stream=True)
            
            # Print each chunk of the response as it arrives
            for chunk in output_stream:
                if isinstance(chunk['response'], str):  # Ensure chunk is a string
                    sys.stdout.write(chunk['response'])
                    sys.stdout.flush()
                else:
                    logging.warning("Received non-string chunk: %s", chunk)
            
        except Exception as e:
            logging.error(f"Error generating answer: {e}")

    def load_config(file_path='config_template.yaml'):
        with open(file_path, 'r') as stream:
            return yaml.safe_load(stream)