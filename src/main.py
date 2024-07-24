import sys
import RAG as rag

def main():
    input_dir = '/home/tagore/repos/ai/data/example_data/'
    output_dir = '/home/tagore/repos/ai/data/exampe_debug_folder'
    chroma_db_dir = '/home/tagore/repos/ai/data/example_db/'
    chroma_db_name = 'test'

    ragtest = rag.RAG(input_dir, output_dir, chroma_db_dir, chroma_db_name)

    if len(sys.argv) < 2:
        print("Usage: python main.py 'your query'")
        sys.exit(1)

    query_text = sys.argv[1]
    answer = ragtest.generate_answer(query_text)
    print("Answer:", answer)

if __name__ == "__main__":
    main()