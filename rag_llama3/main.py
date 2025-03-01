import sys
from rag_llama3.RAG import RAG as rag
import yaml


def main():
    # Load configuration, please insert the path to your configuration file.
    #config = rag.load_config('/home/tagore/repos/ai/rag_llama3/config.yaml')
    config = rag.load_config('../config.yaml')

    # Extract configuration values
    vector_db_config = config.get('vector_db')
    input_dir = vector_db_config.get('input_dir')
    output_dir = vector_db_config.get('output_dir')
    urls_path = vector_db_config.get('urls_path')
    chroma_db_dir = vector_db_config.get('chroma_db_dir')
    chroma_db_name = vector_db_config.get('chroma_db_name')
    model = vector_db_config.get('model')

    ragtest = rag(input_dir, output_dir, urls_path, chroma_db_dir, chroma_db_name, model)

    if len(sys.argv) < 2:
        print("Usage: python main.py 'your query'")
        sys.exit(1)

    query_text = sys.argv[1]
    ragtest.stream_answer(query_text)
    print("\n")

if __name__ == "__main__":
    main()