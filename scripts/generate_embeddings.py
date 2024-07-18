import os
import time
import ollama
import chromadb
import torch
from transformers import AutoTokenizer, AutoModel

def create_embeddings_and_store_in_chroma(documents_dir, collection_name, model="mxbai-embed-large", database_path='./chroma_db'):
    """
    Embeds documents from the specified directory using the Ollama embedding model
    and stores them in a local Chroma database file.

    Args:
    - documents_dir (str): Directory path containing text documents (.txt files).
    - collection_name (str): Name of the Chroma database collection to create or use.
    - model (str): Ollama embedding model to use (default: "mxbai-embed-large").
    - database_path (str): Path to the local Chroma database file (default: './chroma_db.sqlite').

    Returns:
    - chromadb.Collection: Chroma database collection object.
    """
    # Initialize Chroma database client with local database path
    client = chromadb.PersistentClient(path=database_path)
    if collection_name in client.list_collections():
        collection = client.get_collection(name=collection_name)
    else:
        collection = client.create_collection(name=collection_name)

    # Get total number of documents
    total_documents = len([filename for filename in os.listdir(documents_dir) if filename.endswith(".txt")])
    processed_documents = 0

    # Iterate through text files in the directory
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_dir, filename)
            
            # Read document text from the file
            with open(file_path, 'r', encoding='utf-8') as file:
                document_text = file.read().strip()

            # Embedding the document using Ollama
            start_time = time.time()
            response = ollama.embeddings(model=model, prompt=document_text)
            embedding = response["embedding"]
            end_time = time.time()

            # Store document in Chroma collection
            collection.add(
                ids=[filename],  # Use filename as ID (you can customize this as needed)
                embeddings=[embedding],
                documents=[document_text]
            )

            processed_documents += 1
            elapsed_time = end_time - start_time
            estimated_time_left = (total_documents - processed_documents) * elapsed_time

            print(f"Processed document {processed_documents}/{total_documents}. "
                  f"Elapsed Time: {elapsed_time:.2f} seconds. "
                  f"Estimated Time Left: {estimated_time_left:.2f} seconds.")

    print(f"Documents embedded and stored in local Chroma database '{database_path}'.")

    return collection

def translate_and_add_embeddings(documents_dir, collection_name, model="mxbai-embed-large", database_path='./chroma_db'):
    """
    Translates documents from the specified directory into embeddings using the
    Ollama embedding model and adds them to an existing Chroma database collection.

    Args:
    - documents_dir (str): Directory path containing text documents (.txt files).
    - collection_name (str): Name of the Chroma database collection to add to.
    - model (str): Ollama embedding model to use (default: "mxbai-embed-large").
    - database_path (str): Path to the local Chroma database file (default: './chroma_db').

    Returns:
    - chromadb.Collection: Chroma database collection object, or None if an error occurred.
    """
    # Initialize Chroma database client with local database path
    client = chromadb.PersistentClient(path=database_path)

    try:
        # Attempt to get the existing collection
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        print(f"Error accessing collection '{collection_name}': {e}")
        return None

    # Get total number of documents
    total_documents = len([filename for filename in os.listdir(documents_dir) if filename.endswith(".txt")])
    processed_documents = 0

    # Get existing document IDs in the collection
    existing_ids = set(collection.get()['ids'])

    # Iterate through text files in the directory
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_dir, filename)

            try:
                # Read document text from the file
                with open(file_path, 'r', encoding='utf-8') as file:
                    document_text = file.read().strip()
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
                continue  # Skip this file and move to the next

            # Skip if the document ID (filename) already exists in the collection
            if filename in existing_ids:
                print(f"Skipping duplicate document {filename}.")
                continue

            try:
                # Embedding the document using Ollama
                start_time = time.time()
                response = ollama.embeddings(model=model, prompt=document_text)
                embedding = response["embedding"]
                end_time = time.time()

                # Store document in Chroma collection
                collection.add(
                    ids=[filename],  # Use filename as ID
                    embeddings=[embedding],
                    documents=[document_text]
                )

                processed_documents += 1
                elapsed_time = end_time - start_time
                estimated_time_left = (total_documents - processed_documents) * elapsed_time

                print(f"Processed document {processed_documents}/{total_documents}. "
                      f"Elapsed Time: {elapsed_time:.2f} seconds. "
                      f"Estimated Time Left: {estimated_time_left:.2f} seconds.")

            except Exception as e:
                print(f"Error embedding or storing document {filename}: {e}")

    print(f"Documents translated into embeddings and added to Chroma database '{database_path}'.")

    return collection

def delete_embeddings(directory, database_path='./chroma_db', collection_name='default'):
    '''
    Deletes all vectors in the chroma database that generated from the files in the directory.

    Args:
    - directory (str): Path to the directory to search for files.
    - database_path (str): Path to the local Chroma database file (default: './chroma_db').
    - collection_name (str): Name of the Chroma database collection to delete from (default: 'default').

    Returns:
    - None
    '''
    file_names = get_file_names(directory)
    client = chromadb.PersistentClient(path=database_path)
    collection = client.get_collection(name=collection_name)
    collection.delete(
        ids = file_names
        )


def get_file_names(directory):
    """
    Returns a list of all file names in the specified directory.

    Args:
    - directory (str): Path to the directory to search for files.

    Returns:
    - list: A list of file names (strings) in the directory.
    """
    try:
        # List all files in the directory
        file_names = [filename for filename in os.listdir(directory) if os.path.isfile(os.path.join(directory, filename))]
        return file_names
    except Exception as e:
        print(f"Error accessing directory '{directory}': {e}")
        return []

# Function to get embeddings
def get_embeddings(texts, model, tokenizer, max_length=384, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings_batch = outputs.last_hidden_state.mean(dim=1)
            embeddings.append(embeddings_batch)
    return torch.cat(embeddings)

def create_embeddings_Biobert_in_chroma(documents_dir, collection_name, model_name="dmis-lab/biobert-base-cased-v1.2", database_path='./chroma_db'):
    """
    Embeds documents from the specified directory using the BioBERT embedding model
    and stores them in a local Chroma database file.

    Args:
    - documents_dir (str): Directory path containing text documents (.txt files).
    - collection_name (str): Name of the Chroma database collection to create or use.
    - model_name (str): BioBERT embedding model to use (default: "dmis-lab/biobert-base-cased-v1.1").
    - database_path (str): Path to the local Chroma database file (default: './chroma_db').

    Returns:
    - None
    """
    # Initialize BioBERT model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Initialize Chroma database client with local database path
    client = chromadb.PersistentClient(path=database_path)
    collection = client.create_collection(name=collection_name)

    # Get total number of documents
    total_documents = len([filename for filename in os.listdir(documents_dir) if filename.endswith(".txt")])
    processed_documents = 0

    # Iterate through text files in the directory
    for filename in os.listdir(documents_dir):
        if filename.endswith(".txt"):
            file_path = os.path.join(documents_dir, filename)
            
            # Read document text from the file
            with open(file_path, 'r', encoding='utf-8') as file:
                document_text = file.read().strip()

            # Embedding the document using BioBERT
            start_time = time.time()
            embedding = get_embeddings([document_text], model, tokenizer).numpy().tolist()[0]
            end_time = time.time()

            # Store document in Chroma collection
            collection.add(
                ids=[filename],  # Use filename as ID (you can customize this as needed)
                embeddings=[embedding],
                documents=[document_text]
            )

            processed_documents += 1
            elapsed_time = end_time - start_time
            estimated_time_left = (total_documents - processed_documents) * elapsed_time

            print(f"Processed document {processed_documents}/{total_documents}. "
                  f"Elapsed Time: {elapsed_time:.2f} seconds. "
                  f"Estimated Time Left: {estimated_time_left:.2f} seconds.")

    print(f"Documents embedded and stored in local Chroma database '{database_path}'.")