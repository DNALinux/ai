import sys
import RAG as rag
import yaml


def main():
    # Load configuration
    config = rag.load_config()

    # Extract configuration values
    vector_db_config = config.get('vector_db')
    input_dir = vector_db_config.get('input_dir')
    output_dir = vector_db_config.get('output_dir')
    chroma_db_dir = vector_db_config.get('chroma_db_dir')
    chroma_db_name = vector_db_config.get('chroma_db_name')
    model = vector_db_config.get('model')

    ragtest = rag.RAG(input_dir, output_dir, chroma_db_dir, chroma_db_name)

    if len(sys.argv) < 2:
        print("Usage: python main.py 'your query'")
        sys.exit(1)

    query_text = sys.argv[1]
    ragtest.stream_answer(query_text)
    print("\n")

if __name__ == "__main__":
    main()