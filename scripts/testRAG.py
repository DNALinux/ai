"""
For colab


testRAG.py --v_model mxbai-embed-large --LLM_model phi4 --db_dir /content/testdata1/ --db_name blastdb

vmodels:

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
parser.add_argument('--v_model', type=str, required=False, 
                    default='mxbai-embed-large', help='Vector Model (default: mxbai-embed-large)')
parser.add_argument('--LLM_model', type=str, required=False, 
                    default='phi4', help='LLM Model (default: phi4)')
parser.add_argument('--db_dir', type=str, required=True, 
    help='Path to the DB')
parser.add_argument('--db_name', type=str, required=True,  
    help='Name of the database')


args = parser.parse_args()

v_model = args.v_model
print(f"Vector model: {v_model}")
model = args.LLM_model
print(f"LLM: {model}")
chroma_db_dir = args.db_dir
chroma_db_name = args.db_name

#chroma_db_dir = "/content/testdata2/"
#chroma_db_name = "blastdb"


colabRAG = rag(chroma_db_dir, chroma_db_name, v_model=v_model)
qs = ["Which are the 2 most important criteria to design the new BLAST code?",
    "What is the MedGen data model? What are the components of a MedGen record?", 
     "Which are the dimensions of the fixed amount of computation per cell of a path graph in the standard dynamic programming algorithms for pairwise sequence alignment",
     "What is the difference between the alignment of a simple sequence with a pattern embodied by a position-specific score matrix to the alignment of two simple sequences"]

for q in qs:
    print(f"Q: {q}")
    print(colabRAG.generate_answer(q, model=model))
    print("***********\n")
