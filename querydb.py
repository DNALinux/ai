"""
"""

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
model = vector_db_config.get('model')


# (self, chroma_db_dir: str, chroma_db_name: str, model="mxbai-embed-large"):

test_vector_db = vdb(chroma_db_dir, chroma_db_name)
print(test_vector_db.show_sources())
#print(dir(test_vector_db))
print("show count docs")
print(test_vector_db.count_docs())
#def load_data(self, input_dir: str = None, output_dir: str = None, urls_path: str = None):
test_vector_db.load_data(input_dir)
print("show sources")
print(test_vector_db.show_sources())
print("show count docs")
print(test_vector_db.count_docs())
# test_vector_db.load_url()
#test_vector_db.peek()

#testRAG = rag(chroma_db_dir, chroma_db_name)
#print(testRAG.generate_answer("Name all blast programs"))

