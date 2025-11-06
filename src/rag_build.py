# new_src/rag_build.py
import os, glob, json
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
load_dotenv()

from langchain_community.document_loaders import NotebookLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# -------------------------------
# Paths
# -------------------------------
DATA_DIR = Path("data")
UPLOADS_DIR = Path("uploads")
INDEX_DIR = DATA_DIR / "index"            # Chroma persist directory
MANIFEST_PATH = INDEX_DIR / "manifest.json"

# -------------------------------
# Settings
# -------------------------------
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
BATCH_SIZE = 256  # safe + fast

# -------------------------------
# Helpers
# -------------------------------
def _notebook_paths() -> Dict[str, float]:
    """Return {path_str: mtime} for .ipynb under data/ and uploads/."""
    paths = []
    for root in [DATA_DIR, UPLOADS_DIR]:
        if root.exists():
            paths += glob.glob(str(root / "**" / "*.ipynb"), recursive=True)
    return {p: os.path.getmtime(p) for p in sorted(paths)}

def _load_manifest() -> Dict[str, float]:
    if MANIFEST_PATH.exists():
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def _save_manifest(manifest: Dict[str, float]) -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

def _load_ipynb_docs(file_paths: List[str]):
    docs = []
    for p in file_paths:
        loader = NotebookLoader(p, include_outputs=False, max_output_length=0)
        docs.extend(loader.load())
    # normalize metadata
    for d in docs:
        d.metadata["source"] = d.metadata.get("source", d.metadata.get("path", "notebook"))
    return docs

def _split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

def _ensure_chroma(embeddings) -> Chroma:
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(INDEX_DIR),
        collection_name="notebooks",
    )

# -------------------------------
# Main
# -------------------------------
def main():
    # 0) sanity
    if not DATA_DIR.exists() and not UPLOADS_DIR.exists():
        raise AssertionError("Neither data/ nor uploads/ folder found")

    # 1) discover files and manifest
    current = _notebook_paths()              # {path: mtime}
    manifest = _load_manifest()              # {path: mtime}

    to_add_or_update = [p for p, mt in current.items() if manifest.get(p) != mt]
    to_delete = [p for p in manifest.keys() if p not in current]

    print("🗂️  Found notebooks:", len(current))
    print("✅ Unchanged:", len(current) - len(to_add_or_update))
    print("✳️  To (re)index:", len(to_add_or_update))
    print("🗑️  To delete:", len(to_delete))

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    chroma = _ensure_chroma(embeddings)

    # 2) delete removed/changed files from index
    if to_delete:
        for p in to_delete:
            chroma.delete(where={"source": p})
        print(f"🗑️  Deleted old entries for {len(to_delete)} removed files.")

    if to_add_or_update:
        # also clear existing chunks for files that changed
        for p in to_add_or_update:
            chroma.delete(where={"source": p})

        # 3) load + split
        docs = _load_ipynb_docs(to_add_or_update)
        chunks = _split_docs(docs)

        # 4) batch add
        total = len(chunks)
        print(f"\n🔹 Embedding {total} chunks in batches of {BATCH_SIZE}...\n")
        for i in range(0, total, BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            chroma.add_documents(batch)
            print(f"  → Embedded {i + len(batch)}/{total}")

        # 5) update manifest mtimes for reindexed files
        for p in to_add_or_update:
            manifest[p] = current[p]

    # 6) drop deleted files from manifest and save
    for p in to_delete:
        manifest.pop(p, None)
    _save_manifest(manifest)

    # 7) materialize/sync to disk
    chroma.get()
    print("\n✅ Incremental index build complete.")
    print(f"   Indexed folders: data/  uploads/")
    print(f"   Persisted at   : {INDEX_DIR}")

if __name__ == "__main__":
    main()
