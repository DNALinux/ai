import chromadb
import torch
import h5py
from transformers import AutoTokenizer, AutoModel



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