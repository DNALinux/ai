"""
Testing the lib using uv

/app/ai# uv run python libtest.py
"""



import argparse
import sys
from pathlib import Path
sys.path.append("/app/ai")
from rag_llama3 import VectorDB as vdb
from rag_llama3 import RAG as rag


def parse_cli_args() -> argparse.Namespace:
    """
    Build an ArgumentParser that provides the five variables.  
    All arguments are optional.

    Returns
    -------
    argparse.Namespace
        An object with attributes: e_llm, db_dir, db_name, data_src, llm.
    """

    q = "What is the difference between the alignment of a simple sequence "\
        "with a pattern embodied by a position-specific score matrix to the"\
        " alignment of two simple sequences?"

    parser = argparse.ArgumentParser(
        description=(
            "For testing Takollama."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--e-llm",
        dest="e_llm",
        default="nomic-embed-text",
        help="Embedding LLM to use (e.g. 'nomic-embed-text').",
    )
    parser.add_argument(
        "--db-dir",
        dest="db_dir",
        default="dbdir",
        help="Directory that will hold the vector store.",
    )
    parser.add_argument(
        "--db-name",
        dest="db_name",
        default="blast",
        help="Base name for the vector store.",
    )
    parser.add_argument(
        "--data-src",
        dest="data_src",
        default="/app/ai/data/blastpdfs",
        help="Path to the folder that contains the source PDFs (or other data).",
    )
    parser.add_argument(
        "--llm",
        dest="llm",
        default="phi3:3.8b",
        help="Chat LLM to use for answering queries (e.g. 'phi3:3.8b').",
    )
    parser.add_argument(
        "--q",
        dest="q",
        default=q,
        help="Chat LLM to use for answering queries (e.g. 'phi3:3.8b').",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_cli_args()
    e_llm = args.e_llm
    db_dir = args.db_dir
    db_name = args.db_name
    data_src = args.data_src
    llm = args.llm
    q = args.q
    db_dir_path = Path(db_dir).expanduser().resolve()
    data_src_path = Path(data_src).expanduser().resolve()
    if not data_src_path.is_dir():
        sys.stderr.write(f"⚠️ data source directory does not exist: {data_src_path}\n")
        sys.exit(1)
    db_dir_path.mkdir(parents=True, exist_ok=True)
    t_vector_db = vdb(db_dir, db_name, e_llm)
    print(t_vector_db.count_docs())
    t_vector_db.load_data(data_src, "tmp")
    print(t_vector_db.count_docs())
    colabRAG = rag(db_dir, db_name, v_model=e_llm)
    ans = colabRAG.generate_answer(q, model=llm)
    print(ans)
