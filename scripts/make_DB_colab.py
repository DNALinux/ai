"""
Make the embbeding and DB in Colab
WIP
"""

import os

import generate_embeddings as ge
from extract_text import extract_and_save_pdf_text, crawl_and_extract
from langchain_community.document_loaders import PyPDFLoader
from preprocess_text import preprocess_files

os.environ['ALLOW_RESET'] = 'TRUE'
COMMON_PATH = '/content/'
pdf_path = 'ai/data/raw/pdfs/Bookshelf_NBK279690.pdf'
processed_path = COMMON_PATH + 'ai/data/processed/texts/'
docs_dir = COMMON_PATH +  'ai/data/processed/processed_texts/Bookshelf_NBK279690.txt'
collection_name = "blast_db"
#collection_name2 = "blast_db_unprocessed"
db_file = COMMON_PATH + 'ai/data/processed/embeddings/' + collection_name
#db_file2 = COMMON_PATH + 'ai/data/processed/embeddings/' + collection_name2


extract_and_save_pdf_text(pdf_path, processed_path)
preprocess_files(processed_path, docs_dir)

collection = ge.create_embeddings_and_store_in_chroma(docs_dir, collection_name, database_path=db_file)
#collection2 = ge.create_embeddings_and_store_in_chroma(processed_path, collection_name2, database_path=db_file2)


url = "https://github.com/enormandeau/ncbi_blast_tutorial"
directory = COMMON_PATH +  'ai/data/processed/texts/ncbi_blast_tutorial'
crawl_and_extract(url, directory, chunk_size=1500, max_depth=1)

url = 'https://open.oregonstate.education/computationalbiology/chapter/command-line-blast/'
directory = COMMON_PATH + 'ai/data/processed/texts/oregonstate'
crawl_and_extract(url, directory, chunk_size=1500, max_depth=1)

url = 'https://en.wikipedia.org/wiki/BLAST_(biotechnology)'
directory = COMMON_PATH + 'ai/data/processed/texts/wikipedia'
crawl_and_extract(url, directory, chunk_size=1500, max_depth=0)


documents_directory = [COMMON_PATH + 'ai/data/processed/texts/ncbi_blast_tutorial',
                       COMMON_PATH + 'ai/data/processed/texts/oregonstate',
                       COMMON_PATH + 'ai/data/processed/texts/wikipedia']
for doc_dir in documents_directory:
    ge.translate_and_add_embeddings(doc_dir, collection_name, database_path=db_file)
