"""
"""

import ollama
import chromadb


COMMON_PATH = '/home/ran/projects/dnalinux/'
pdf_path = COMMON_PATH + 'ai/data/raw/pdfs/Bookshelf_NBK279690.pdf'
processed_path = COMMON_PATH + 'ai/data/processed/texts/Bookshelf_NBK279690.pdf'
docs_dir = COMMON_PATH +  'ai/data/processed/processed_texts/Bookshelf_NBK279690.txt'
collection_name = "blast_db"
collection_name2 = "blast_db_unprocessed"
db_file = COMMON_PATH + 'ai/data/processed/embeddings/' + collection_name
db_file2 = COMMON_PATH + 'ai/data/processed/embeddings/' + collection_name2

client = chromadb.PersistentClient(path=db_file)
collection = client.get_collection(name=collection_name)
client2 = chromadb.PersistentClient(path=db_file2)
collection2 = client2.get_collection(name=collection_name2)


blast_dna_questions = (
    "What is word size parameter in BLAST?",
    "How to get the results of BLASTP in XML format?",
    "How to perform a BLAST on a specific taxonomic group?",
    "What blast program do I use to perform BLAST with epitopes smaller than 10 amino acids?",
    "What blast program and task I should use to perform BLAST with epitopes smaller than 10 amino acids?",
    "Which kind of databases can be searched with BLASTX?",
    "What is blast_formater program for?",
    "What is a database free search?",
    "What is Windowmasker and when should I use it?",
    "What is the parameter '-subject' for in a BLAST search?",
    "How can I multi-thread my search?"
    )

prompts = blast_dna_questions


# for each question.

for i, q in enumerate(prompts):
  print(f"Question {i+1}: {q}")
  # generate an embedding for the prompt and retrieve the most relevant doc
  response = ollama.embeddings(
    prompt=q,
    model="mxbai-embed-large"
  )
  results = collection2.query(
    query_embeddings=[response["embedding"]],
    n_results=3
  )
  #data = results['documents'][0][0]
  data = results['documents'][0]
  #print(data)

  output = ollama.generate(
    model="llama3",
    prompt=f"Using this data: {data}. Respond to this prompt: {q}. "\
      "Only the answer, don't say according to the text. " \
      "If an aswer can be inferred from the provided data, use your data"
  )

  print(output['response'])
  print()

#model_name="dmis-lab/biobert-base-cased-v1.2"
# Initialize BioBERT model and tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name)
#model = AutoModel.from_pretrained(model_name)
#em = ge.get_embeddings(prompt, model, tokenizer)
#results = collection3.query(
#  query_embeddings=em.tolist(),
#  n_results=3
#)
#results
