# new_src/rag_build.py
import os, glob
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path("data")
INDEX_DIR = DATA_DIR / "index"  # Chroma persist directory

def load_ipynb_docs():
    paths = glob.glob(str(DATA_DIR / "**" / "*.ipynb"), recursive=True)
    docs = []
    for p in paths:
        loader = NotebookLoader(p, include_outputs=False, max_output_length=0)
        docs.extend(loader.load())
    for d in docs:
        d.metadata["source"] = d.metadata.get("source", d.metadata.get("path", "notebook"))
    return docs

def main():
    assert DATA_DIR.exists(), "data/ folder not found"
    docs = load_ipynb_docs()
    if not docs:
        raise RuntimeError("No .ipynb files found under data/")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    # Create/update a persistent Chroma index
    Chroma.from_documents(
        chunks,
        embedding=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_name="notebooks",
    )
    print(f"✅ Built Chroma index with {len(chunks)} chunks → {INDEX_DIR}")

if __name__ == "__main__":
    main()
