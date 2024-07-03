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
│   ├── data/
│   │   ├── __init__.py
│   │   ├── extract.py
│   │   ├── preprocess.py
│   │   └── ...
│   ├── embeddings/
│   │   ├── __init__.py
│   │   ├── generate.py
│   │   └── ...
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── pipeline.py
│   │   └── ...
│   └── utils/
│       ├── __init__.py
│       ├── logging.py
│       ├── config.py
│       └── ...
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

- Python 3.8 or higher
- PyTorch
- CUDA (if using GPU)
- LLaMA3

### Installation

1. Clone the repository:
   ```
   git clone git@github.com:DNALinux/ai.git
   cd ai
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

### Configuration

Edit the `config.yaml` file to set your project-specific parameters and paths.

## Project Workflow

1. **Data Extraction**: Extract text from raw data sources.
2. **Data Preprocessing**: Preprocess the extracted text.
3. **Embedding Generation**: Generate embeddings for the preprocessed text using LLaMA3.
4. **RAG Pipeline**: Set up and run the RAG pipeline.

## Testing

Run the tests to ensure everything is working correctly:
```
pytest tests/
```

## License

This project is licensed under the GNU GENERAL PUBLIC LICENSE. See the [LICENSE](LICENSE) file for details.

