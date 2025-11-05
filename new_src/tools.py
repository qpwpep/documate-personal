import os
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# ─────────────────────────────────────────────
# 🔹 LangChain / External dependencies
# ─────────────────────────────────────────────
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# ─────────────────────────────────────────────
# 1. Environment setup
# ─────────────────────────────────────────────
load_dotenv(find_dotenv(), override=True)

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")

assert TAVILY_KEY, "Missing TAVILY_API_KEY in environment (.env not loaded or key not set)."
assert OPENAI_KEY, "Missing OPENAI_API_KEY in environment (.env not loaded or key not set)."

# ─────────────────────────────────────────────
# 2. TavilySearch configuration
# ─────────────────────────────────────────────
DEFAULT_DOCS = {
    "python": "https://docs.python.org/3/",
    "git": "https://git-scm.com/docs",
    "LangChain": "https://python.langchain.com/docs",
    "Matplotlib": "https://matplotlib.org/stable/api/index.html",
    "NumPy": "https://numpy.org/doc/stable/",
    "pandas": "https://pandas.pydata.org/docs/",
    "PyTorch": "https://docs.pytorch.org/docs/stable/index.html",
    "Hugging Face": "https://huggingface.co/docs",
    "FastAPI": "https://fastapi.tiangolo.com/reference/",
    "BeautifulSoup": "https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
    "streamlit": "https://docs.streamlit.io/",
    "gradio": "https://www.gradio.app/docs",
    "scikit-learn": "https://scikit-learn.org/stable/api/index.html",
    "Pydantic": "https://docs.pydantic.dev/latest/api/base_model/",
}

tavilysearch = TavilySearch(
    max_results=3,
    include_domains=list(DEFAULT_DOCS.values()),
)

# ─────────────────────────────────────────────
# 3. Save-to-text tool
# ─────────────────────────────────────────────
def save_text_to_file(content: str, filename_prefix: str = "response") -> str:
    """Save text content to a timestamped .txt file and return the file path message."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{ts}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved output to {filename}"

class SaveArgs(BaseModel):
    content: str = Field(description="The exact final response text to write into the .txt file.")
    filename_prefix: Optional[str] = Field(
        default="response",
        description="Optional short prefix for the filename (no extension)."
    )

save_text_tool = StructuredTool.from_function(
    name="save_text",
    description=(
        "Save the given final response text into a timestamped .txt file in the current directory. "
        "Call this at most ONCE per user request. If you already saved, do not call again."
    ),
    func=save_text_to_file,
    args_schema=SaveArgs,
)

# ─────────────────────────────────────────────
# 4. RAG (Chroma) search tool
# ─────────────────────────────────────────────
INDEX_PATH = "data/index"

def _load_chroma():
    emb = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        embedding_function=emb,
        persist_directory=INDEX_PATH,
        collection_name="notebooks",
    )

def rag_search(query: str, k: int = 4) -> str:
    """Search local .ipynb notebooks and return relevant snippets with sources."""
    if not os.path.isdir(INDEX_PATH):
        return "RAG index not found. Please build it first (python -m new_src.rag_build)."

    db = _load_chroma()
    docs = db.similarity_search(query, k=k)
    if not docs:
        return "No relevant passages found in local notebooks."

    lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "notebook")
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + " …"
        lines.append(f"{i}. {snippet}\n   [◆ 로컬 예제] {src}")
    return "\n".join(lines)

class RagArgs(BaseModel):
    query: str = Field(description="The user's information need to search over local notebooks.")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to return.")

rag_search_tool = StructuredTool.from_function(
    name="rag_search",
    description=(
        "Search local .ipynb notebooks (vector index) and return relevant snippets with sources. "
        "Use this when the question is covered by our local documents."
    ),
    func=rag_search,
    args_schema=RagArgs,
)

# ─────────────────────────────────────────────
# 5. Export
# ─────────────────────────────────────────────
tools = [tavilysearch, rag_search_tool, save_text_tool]
