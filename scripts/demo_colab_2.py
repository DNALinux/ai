"""
For colab
"""
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__)) 
target_dir = os.path.join(current_dir, '..') 

sys.path.append(target_dir)


from rag_llama3 import RAG as rag
from rag_llama3 import VectorDB as vdb

# Load configuration, insert path to your configuration file.
#config = rag.load_config('/content/ai/rag_llama3/config_colab.yml')

# Extract configuration values
#cfg = config.get('vector_db')
#chroma_db_dir = cfg.get('chroma_db_dir')
#chroma_db_name = cfg.get('chroma_db_name')
#v_model = cfg.get('model')
v_model = "nomic-embed-text"
print(f"Vector model: {v_model}")
#cfg = config.get('LLM')
#model = cfg.get('model')
# TAKE IT FROM ENV
model = "llama3.2:3b"
#print(f"LLM: {model}")

chroma_db_dir = "/content/testdata1/"
chroma_db_name = "blastdb"

colabRAG = rag(chroma_db_dir, chroma_db_name, v_model=v_model)
q="What is the MedGen data model? What are the components of a MedGen record?"
print(f"Q: {q}")
print(colabRAG.generate_answer(q, model=model))
