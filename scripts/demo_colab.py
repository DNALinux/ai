"""
For colab
"""

from rag_llama3 import RAG as rag
from rag_llama3 import VectorDB as vdb

# Load configuration, insert path to your configuration file.
config = rag.load_config('/content/ai/rag_llama3/config_colab.yml')

# Extract configuration values
cfg = config.get('vector_db')
chroma_db_dir = cfg.get('chroma_db_dir')
chroma_db_name = cfg.get('chroma_db_name')
v_model = cfg.get('model')
#v_model = "nomic-embed-text"
print(f"Vector model: {v_model}")
cfg = config.get('LLM')
model = cfg.get('model')
print(f"LLM: {model}")

colabRAG = rag(chroma_db_dir, chroma_db_name, v_model=v_model)
print("Q: Name all blast programs")
print(colabRAG.generate_answer("Name all blast programs", model=model))
print("Q: What is the word size parameter in BLAST?")
print(colabRAG.generate_answer("What is the word size parameter in BLAST?", model=model))
print("Q: How to get the results of BLASTP in XML format?")
print(colabRAG.generate_answer("How to get the results of BLASTP in XML format?", model=model))
print("Q: How to perform a BLAST on a specific taxonomic group?")
print(colabRAG.generate_answer("How to perform a BLAST on a specific taxonomic group?", model=model))
print("Q: What parameters do I use to perform BLAST with epitopes smaller than 10 amino acids?")
print(colabRAG.generate_answer("What parameters do I use to perform BLAST with epitopes smaller than 10 amino acids?", model=model))
print("Q: What is the difference between BLASTP and BLASTX?")
print(colabRAG.generate_answer("Which kind of databases can be searched with BLASTX?", model=model))


"""
model = vector_db_config.get('ollama_model').get('model_name')
model = "llama3.1"
model = "llama3.2:3b"
model = "phi4"
#model = "gemma3:12b"
#model = "deepseek-r1:8b"
#model = "deepseek-r1:14b"
#model = "gemma3:1b"
#test_vector_db = vdb(input_dir, output_dir, urls_path, chroma_db_dir, chroma_db_name)
#test_vector_db.load_url()
#test_vector_db.peek()
"""
