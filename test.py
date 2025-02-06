from rag_llama3 import RAG as rag
from rag_llama3 import VectorDB as vdb
# Load configuration, please insert the path to your configuration file.
config = rag.load_config('/home/ran/projects/dnalinux/aiSet20/ai/rag_llama3/config.yaml')

# Extract configuration values
vector_db_config = config.get('vector_db')
input_dir = vector_db_config.get('input_dir')
output_dir = vector_db_config.get('output_dir')
urls_path = vector_db_config.get('urls_path')
chroma_db_dir = vector_db_config.get('chroma_db_dir')
chroma_db_name = vector_db_config.get('chroma_db_name')
model = vector_db_config.get('model')

#test_vector_db = vdb(input_dir, output_dir, urls_path, chroma_db_dir, chroma_db_name)
#test_vector_db.load_url()
#test_vector_db.peek()

testRAG = rag(chroma_db_dir, chroma_db_name)
print(testRAG.generate_answer("Name all blast programs"))

