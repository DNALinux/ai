"""
For colab

RAG_query.py --e_model mxbai-embed-large --LLM_model phi4 --db_dir /content/testdata1/ --db_name blastdb --query "INSERT QUERY HERE"

embedded models:

nomic-embed-text
mxbai-embed-large

LLM Models:

llama3.2:3b
llama3.3:70b
phi4

"""
import os
import sys
import argparse

current_dir = os.path.dirname(os.path.abspath(__file__)) 
target_dir = os.path.join(current_dir, '..') 

sys.path.append(target_dir)

# get from command line vmodel, model, vector dir and vector name

from rag_llama3 import RAG as rag
from rag_llama3 import VectorDB as vdb

parser = argparse.ArgumentParser(description='Create a database.')
parser.add_argument('--e_model', type=str, required=False, 
                    default='mxbai-embed-large', help='Vector Model (default: mxbai-embed-large)')
parser.add_argument('--LLM_model', type=str, required=False, 
                    default='phi4', help='LLM Model (default: phi4)')
parser.add_argument('--db_dir', type=str, required=True, 
    help='Path to the DB')
parser.add_argument('--db_name', type=str, required=True,  
    help='Name of the database')
parser.add_argument('--query', type=str, required=True,  
    help='Name of the database')

args = parser.parse_args()

e_model = args.e_model
q = args.query
print(f"Embedding model: {e_model}")
model = args.LLM_model
print(f"LLM: {model}")
chroma_db_dir = args.db_dir
chroma_db_name = args.db_name

colabRAG = rag(chroma_db_dir, chroma_db_name, v_model=e_model)
print(f"Q: {q}\n")
print(colabRAG.generate_answer(q, model=model))
print("***********\n")

