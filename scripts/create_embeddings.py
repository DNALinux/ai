import os

import generate_embeddings as ge
from extract_text import extract_and_save_pdf_text
from langchain_community.document_loaders import PyPDFLoader
from preprocess_text import preprocess_files


os.environ['ALLOW_RESET'] = 'TRUE'
COMMON_PATH = '/home/ran/projects/dnalinux/'
pdf_path = COMMON_PATH + 'ai/data/raw/pdfs/Bookshelf_NBK279690.pdf'
processed_path = COMMON_PATH + 'ai/data/processed/texts/Bookshelf_NBK279690.pdf'
docs_dir = COMMON_PATH +  'ai/data/processed/processed_texts/Bookshelf_NBK279690.txt'
collection_name = "blast_db"
collection_name2 = "blast_db_unprocessed"
db_file = COMMON_PATH + 'ai/data/processed/embeddings/' + collection_name
db_file2 = COMMON_PATH + 'ai/data/processed/embeddings/' + collection_name2


extract_and_save_pdf_text(pdf_path, processed_path)
preprocess_files(processed_path, docs_dir)

collection = ge.create_embeddings_and_store_in_chroma(docs_dir, collection_name, database_path=db_file)
collection2 = ge.create_embeddings_and_store_in_chroma(processed_path, collection_name2, database_path=db_file2)