# RAG-LLaMA3 AI Project

We utilize Retrieval Augmented Generation on the LLaMA3 model to create an AI agent that can answer questions about bioinformatics software DNALinux. It helps users navigate through a large range of bioinformatics tools. Additionally, you will be able to create a simple RAG AI agent with your own resources.

You can check out our code at our [repo](https://github.com/DNALinux/ai/tree/main).

## Quick Start with Pre-installed AI Agent

1. **Install the package**:
    ```bash
    pip install rag-llama3
    ```
    If you are unsure whether you have installed this package, you can use the following command:
    ```bash
    pip show rag_llama3
    ```

2. **Ask a question to our pre-installed DNALinux AI agent**:
    ```bash
    rag-llama3 "your question goes here"
    ```

## How to Construct Your Own RAG AI Agent

Make sure that you have already installed the package. There is a [Instruction.ipynb](https://github.com/DNALinux/ai/blob/main/Instruction.ipynb) you can use to test the code.

1. **Configuration**:
   - Set up the directory paths for storing your data and Chroma database. You do not need to create them manually; just specify where they should be, and they will be automatically created:
     - `input_dir`: Directory for PDF, HTML files, and URLs.
     - `urls_path`: Path to a file named `urls.txt` where you put all the URLs.
     - `output_dir`: Directory where `TextExtractor` will store all `.txt` files (for debugging purposes).
     - `chroma_db_dir`: Directory where your Chroma database will be stored.
     - `chroma_db_name`: Collection name for your Chroma database.
   - Note: The embedding model defaults to `'mxbai-embed-large'`. Feel free to choose your preferred Ollama embedding model.
   ```python
   from rag_llama3 import RAG as rag
   from rag_llama3 import TextExtractor as te
   from rag_llama3 import VectorDB as vdb
   ```

2. **Directory Setup**:
   - Open a Jupyter notebook and run the following code to ensure that your directory is created:
     ```python
     test_vector_db = vdb(input_dir, output_dir, urls_path, chroma_db_dir, chroma_db_name)
     ```
   - This will create an object that you can use to manipulate your Chroma vector database. It will automatically create all the directories and an empty Chroma database. If everything is already created, it will not overwrite existing files.

3. **Add Files**:
   - Place all PDF and HTML files in the input directory. List all URLs in the `urls.txt` file, each on a new line.

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
   - In a Jupyter notebook, use:
     ```python
     testRAG = rag(input, output, chroma_db, collection_name)
     print(testRAG.generate_answer("Your question goes here"))
     ```