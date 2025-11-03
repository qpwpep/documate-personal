import os
from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime

from dotenv import load_dotenv, find_dotenv
from langchain_tavily import TavilySearch
from langchain_core.tools import StructuredTool

# Load .env deterministically
load_dotenv(find_dotenv(), override=True)

# Read key and fail fast if missing
TAVILY_KEY = os.getenv("TAVILY_API_KEY")
assert TAVILY_KEY, "Missing TAVILY_API_KEY in environment (.env not loaded or key not set)."

# TavilySearch > include_domains, Streamlit > 지원 문서 text에 사용
DEFAULT_DOCS = {
    "python":"https://docs.python.org/3/",
    "git":"https://git-scm.com/docs",
    "LangChain":"https://python.langchain.com/docs",
    "Matplotlib":"https://matplotlib.org/stable/api/index.html",
    "NumPy":"https://numpy.org/doc/stable/",
    "pandas":"https://pandas.pydata.org/docs/",
    "PyTorch":"https://docs.pytorch.org/docs/stable/index.html",
    "Hugging Face":"https://huggingface.co/docs",
    "FastAPI":"https://fastapi.tiangolo.com/reference/",
    "BeautifulSoup":"https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
    "streamlit":"https://docs.streamlit.io/",
    "gradio":"https://www.gradio.app/docs",
    "scikit-learn":"https://scikit-learn.org/stable/api/index.html",
    "Pydantic":"https://docs.pydantic.dev/latest/api/base_model/"
}

# Configure your external tools here
tavilysearch = TavilySearch(max_results=3, include_domains=list(DEFAULT_DOCS.values()))

# --- Save-to-text implementation ---
def save_text_to_file(content: str, filename_prefix: str = "response") -> str:
    """Save text content to a timestamped .txt file and return the file path message."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{ts}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved output to {filename}"

# Pydantic schema so the LLM can pass structured args
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

# Export the tools list BOTH tools available to the agent
tools = [tavilysearch, save_text_tool]
