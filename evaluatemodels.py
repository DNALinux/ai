from rag_llama3 import RAG as rag

import uuid
import time


config = rag.load_config('config.yaml')

# Extract configuration values
vector_db_config = config.get('vector_db')
input_dir = vector_db_config.get('input_dir')
output_dir = vector_db_config.get('output_dir')
urls_path = vector_db_config.get('urls_path')
chroma_db_dir = vector_db_config.get('chroma_db_dir')
chroma_db_name = vector_db_config.get('chroma_db_name')

model_list = ["llama3.1", "llama3.2:3b", "phi4", "deepseek-r1:8b", "deepseek-r1:14b"]
v_model_list = ["mxbai-embed-large", "nomic-embed-text", "snowflake-arctic-embed"]
#
# ollama pull mistral:7b-instruct && ollama pull mistral:7b-instruct-q6_K && ollama pull gemma && ollama pull gemma:7b-instruct-q4_K_M && ollama pull gemma2 && ollama pull gemma2:9b-instruct-q4_0 && ollama pull qwen2.5

model_list = ["mistral:7b-instruct", "mistral:7b-instruct-q6_K", "gemma", "gemma:7b-instruct-q4_K_M", "gemma2", "gemma2:9b-instruct-q4_0", "qwen2.5"]


for v_model in v_model_list:
    for model in model_list:
        unique_id = uuid.uuid4()
        uid = unique_id.__repr__()[:-2].split("-")[-1]
        fname = f"rag_{uid}.txt"
        testRAG = rag(chroma_db_dir, chroma_db_name)
        start_time = time.time()
        with open(fname, "w") as f:
            f.write("\n\nQ: Name all blast programs\n")
            f.write(testRAG.generate_answer("Name all blast programs", model=model))
            f.write("\n\nQ: What is the word size parameter in BLAST?\n")
            f.write(testRAG.generate_answer("What is the word size parameter in BLAST?", model=model))
            f.write("\n\nQ: How to get the results of BLASTP in XML format?\n")
            f.write(testRAG.generate_answer("How to get the results of BLASTP in XML format?", model=model))
            f.write("\n\nQ: How to perform a BLAST on a specific taxonomic group?\n")
            f.write(testRAG.generate_answer("How to perform a BLAST on a specific taxonomic group?", model=model))
            f.write("\n\nQ: What parameters do I use to perform BLAST with epitopes smaller than 10 amino acids?\n")
            f.write(testRAG.generate_answer("What parameters do I use to perform BLAST with epitopes smaller than 10 amino acids?", model=model))
            f.write("\n\nQ: What is the difference between BLASTP and BLASTX?\n")
            f.write(testRAG.generate_answer("Which kind of databases can be searched with BLASTX?", model=model))
            f.write("\n\n-----------\n")
        end_time = time.time()
        timetaken = int(end_time - start_time)
        print(f"{uid}: {v_model} {model} {timetaken}s")

