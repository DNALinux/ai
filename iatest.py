from rag_llama3 import RAG as rag
from rag_llama3 import TextExtractor as te
from rag_llama3 import VectorDB as vdb
# Load configuration, please insert the path to your configuration file.
config = rag.load_config('/home/azureuser/dl/2/ai/config.yaml')

# Extract configuration values
vector_db_config = config.get('vector_db')
input_dir = vector_db_config.get('input_dir')
output_dir = vector_db_config.get('output_dir')
urls_path = vector_db_config.get('urls_path')
chroma_db_dir = vector_db_config.get('chroma_db_dir')
chroma_db_name = vector_db_config.get('chroma_db_name')
model = vector_db_config.get('model')


vector_db = vdb(chroma_db_dir, chroma_db_name)
#vector_db.load_url(urls_path, output_dir)
print(vector_db.peek())

testRAG = rag(chroma_db_dir, chroma_db_name)
print(testRAG.generate_answer("What is BLAST?"))
