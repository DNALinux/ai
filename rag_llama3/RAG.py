import argparse
import logging
import ollama
from rag_llama3 import VectorDB as vdb
import yaml
import sys


class RAG:
    def __init__(self, chroma_db_dir: str, chroma_db_name: str, v_model="mxbai-embed-large"):
        self.vector_db = self._setup_vector_db(chroma_db_dir, chroma_db_name, v_model)
        logging.basicConfig(level=logging.INFO)

    def _setup_vector_db(self, chroma_db_dir, chroma_db_name, v_model):
        """Check if the database exists and set up if not."""
        try:
            # Attempt to load existing vector database
            vector_db = vdb.VectorDB(chroma_db_dir, chroma_db_name, v_model)
            # Check if the database is empty or needs updating
            if not self._is_database_populated(vector_db):
                print("Database is not populated. Be sure to load data before querying.")
            return vector_db
        except Exception as e:
            logging.error(f"Error setting up VectorDB: {e}")
            raise

    def _is_database_populated(self, vector_db):
        """Check if the vector database has data."""
        return len(vector_db.peek()) > 0

    def generate_prompt(self, question, context):
        template = """Use the following pieces of retrieved context to answer the question. 
        If you don't know the answer, just say that you don't know. 
        In the answer copy much of the context you find relevant.
        user
        Question: {question} 
        Context: {context} 
        Do not say according to the text. just give the answer, no much comment."""
        return template.format(question=question, context=context)

    def generate_answer(self, query_text, k=4, model="llama3.1"):
        """Generate an answer using the vector database and a chosen model."""
        try:
            output = ollama.generate(
                    model=model,
                    prompt= self.generate_prompt(query_text, self.vector_db.query(query_text, k)),
                )
            return output['response']
        except Exception as e:
            logging.error(f"Error generating answer: {e}")
            return "Error generating answer."
    
    def _stream_answer(self, query_text, k=4):
        """Generate an answer using the vector database and chosen model with streaming output."""
        try:
            prompt = self.generate_prompt(query_text, self.vector_db.query(query_text, k))
            
            # Generate text with streaming enabled
            output_stream = ollama.generate(model="llama3", prompt=prompt, stream=True)
            
            # Print each chunk of the response as it arrives
            for chunk in output_stream:
                if isinstance(chunk['response'], str):
                    sys.stdout.write(chunk['response'])
                    sys.stdout.flush()
                else:
                    logging.warning("Received non-string chunk: %s", chunk)
            
        except Exception as e:
            logging.error(f"Error generating answer: {e}")

    def load_config(file_path='config_template.yaml'):
        with open(file_path, 'r') as stream:
            return yaml.safe_load(stream)
