from rag_llama3 import RAG as rag
from rag_llama3 import VectorDB as vdb
# Load configuration, please insert the path to your configuration file.
config = rag.load_config('config.yaml')

# Extract configuration values
vector_db_config = config.get('vector_db')
input_dir = vector_db_config.get('input_dir')
output_dir = vector_db_config.get('output_dir')
urls_path = vector_db_config.get('urls_path')
chroma_db_dir = vector_db_config.get('chroma_db_dir')
chroma_db_name = vector_db_config.get('chroma_db_name')
v_model = vector_db_config.get('model')
v_model = "nomic-embed-text"
print(f"Vector model: {v_model}")
model = vector_db_config.get('ollama_model').get('model_name')
model = "llama3.1"
model = "llama3.2:3b"
model = "phi4"
model = "deepseek-r1:8b"
model = "deepseek-r1:14b"
print(f"LLM: {model}")
#test_vector_db = vdb(input_dir, output_dir, urls_path, chroma_db_dir, chroma_db_name)
#test_vector_db.load_url()
#test_vector_db.peek()

testRAG = rag(chroma_db_dir, chroma_db_name)
print("Q: Name all blast programs")
print(testRAG.generate_answer("Name all blast programs", model=model))
print("Q: What is the word size parameter in BLAST?")
print(testRAG.generate_answer("What is the word size parameter in BLAST?", model=model))
print("Q: How to get the results of BLASTP in XML format?")
print(testRAG.generate_answer("How to get the results of BLASTP in XML format?", model=model))
print("Q: How to perform a BLAST on a specific taxonomic group?")
print(testRAG.generate_answer("How to perform a BLAST on a specific taxonomic group?", model=model))
print("Q: What parameters do I use to perform BLAST with epitopes smaller than 10 amino acids?")
print(testRAG.generate_answer("What parameters do I use to perform BLAST with epitopes smaller than 10 amino acids?", model=model))
print("Q: What is the difference between BLASTP and BLASTX?")
print(testRAG.generate_answer("Which kind of databases can be searched with BLASTX?", model=model))

