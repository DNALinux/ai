# RAG-LLaMA3 AI Project

This repository contains a project that implements a Retrieval-Augmented Generation (RAG) system using the LLaMA3 model. The project focuses on creating embeddings for instructions of a professional bioinformatic software to help users conduct biology research.

## Directory Structure

```
ai/
├── data/
│   ├── raw/
│   │   ├── pdfs/
│   │   ├── htmls/
│   │   └── ...
│   ├── processed/
│   │   ├── texts/
│   │   ├── embeddings/
│   │   └── ...
│   └── example_data/
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_embedding_generation.ipynb
│   └── 04_rag_pipeline.ipynb
├── scripts/
│   ├── extract_text.py
│   ├── preprocess_text.py
│   ├── generate_embeddings.py
│   ├── create_rag_pipeline.py
│   └── ...
├── src/
│ ├── init.py
│ ├── main.py
│ ├── VectorDB.py
│ ├── TextExtractor.py
│ └── RAG.py
├── tests/
│   ├── test_extract.py
│   ├── test_preprocess.py
│   ├── test_generate_embeddings.py
│   ├── test_rag_pipeline.py
│   └── ...
├── requirements.txt
├── setup.py
├── README.md
├── LICENSE
├── .gitignore
└── config.yaml
```

## Getting Started

### Prerequisites

- Python 3.10.12
- PyTorch
- CUDA (if using GPU)
- LLaMA3

### Installation

1. Clone the repository:
   ```
   git clone git@github.com:DNALinux/ai.git
   cd ai
   ```

2. Create the Environment from 'environment.yml':
   ```
   conda env create -f environment.yml
   ```

### Configuration

Edit the `config.yaml` file to set your project-specific parameters and paths.

## Project Workflow

1. **Data Extraction**: Extract text from raw data sources.
2. **Data Preprocessing**: Preprocess the extracted text. We provide functions for preprocessing text but did not use them when creating vector embeddings.
3. **Embedding Generation**: Generate embeddings for the preprocessed text using LLaMA3 and store them in a Chroma database.
4. **RAG Pipeline**: Set up and run the RAG pipeline.

## How to Use

1. **Configuration**:
   - Open `config.yaml` to set up the directory paths for storing your data and Chroma database. You do not need to create them manually; just specify where they should be, and they will be automatically created:
     - `input_dir`: Directory for PDF, HTML files, and URLs.
     - `output_dir`: Directory where TextExtractor will store all `.txt` files (for debugging purposes).
     - `chroma_db_dir`: Directory where your Chroma database will be stored.
     - `chroma_db_name`: Collection name for your Chroma database.
   - Note: The embedding model defaults to `'mxbai-embed-large'`. Feel free to choose your preferred Ollama embedding model.

2. **Directory Setup**:
   - Open a Jupyter notebook and run the following code to ensure that your directory is created:
     ```python
     test_vector_db = vdb.VectorDB(input, output, chroma_db, collection_name)
     ```
   - This will create an object that you can use to manipulate your Chroma vector database. It will automatically create all the directories and an empty Chroma database. If everything is already created, it will not overwrite existing files.

3. **Add Files**:
   - Place all PDF and HTML files in the input directory. There will be a `urls.txt` file inside the input directory where you can list all URLs, each on a new line.

4. **Load Data**:
   - Use the `test_vector_db` object to load files into the vector database:
     ```python
     test_vector_db.load_data()
     ```
   - Alternatively, load different types of files individually:
     ```python
     test_vector_db.load_url()
     test_vector_db.load_pdf()
     test_vector_db.load_html()
     ```
   - Loading might take some time. After loading, check if the vector database has been populated successfully:
     ```python
     test_vector_db.peek()
     test_vector_db.show_sources()
     ```
   - Query data from a specific source:
     ```python
     test_vector_db.query_sources(source_name)
     ```
   - To delete data from a source:
     ```python
     test_vector_db.delete_source(source_name)
     ```
   - Or to clear the entire database (be cautious as this is destructive):
     ```python
     test_vector_db.clear_database()
     ```

5. **Generate Answers**:
   - To get your RAG version answer from the command line, navigate to the `src` path and type:
     ```bash
     python3 main.py "your question goes here"
     ```
   - Or in a Jupyter notebook, use:
     ```python
     import TextExtractor as te
     import VectorDB as vdb
     import RAG as rag

     testRAG = rag.RAG(input, output, chroma_db, collection_name)
     print(testRAG.generate_answer("Your question goes here"))
     ```

## Testing

Run the tests to ensure everything is working correctly:
```
pytest tests/
```

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.

