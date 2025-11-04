# new_src/upload_helpers.py
import os
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

def extract_text_from_py(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def extract_text_from_ipynb(path: str) -> str:
    import nbformat
    nb = nbformat.read(path, as_version=4)
    texts: List[str] = []
    for cell in nb.cells:
        if cell.get("cell_type") in {"code", "markdown"}:
            src = cell.get("source") or ""
            texts.append(str(src))
    return "\n\n".join(texts)

def extract_text(path: str) -> str:
    path_lower = path.lower()
    if path_lower.endswith(".py"):
        return extract_text_from_py(path)
    if path_lower.endswith(".ipynb"):
        return extract_text_from_ipynb(path)
    raise ValueError("Unsupported file type (only .py or .ipynb).")

def build_temp_retriever(path: str, k: int = 4):
    """Build a temporary in-memory Chroma retriever from a single uploaded file."""
    text = extract_text(path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    docs = splitter.create_documents([text], metadatas=[{"source": path}])

    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(docs, embedding=emb)  # in-memory (no persist_directory)
    return db.as_retriever(search_kwargs={"k": k})
