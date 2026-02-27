import os
from datetime import datetime
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError

from src.util.util import get_save_text_output_dir

from .slack_utils import create_slack_client, resolve_destination

load_dotenv(find_dotenv(), override=True)

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")
SLACK_DEFAULT_DM_EMAIL = os.getenv("SLACK_DEFAULT_DM_EMAIL")
SLACK_DEFAULT_USER_ID = os.getenv("SLACK_DEFAULT_USER_ID")

assert TAVILY_KEY, "Missing TAVILY_API_KEY in environment (.env not loaded or key not set)."
assert OPENAI_KEY, "Missing OPENAI_API_KEY in environment (.env not loaded or key not set)."

slack_client = create_slack_client(SLACK_BOT_TOKEN)

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


def save_text_to_file(content: str, filename_prefix: str = "response") -> dict:
    """Save text content to a timestamped .txt file and return a status payload."""
    try:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = get_save_text_output_dir()
        os.makedirs(output_path, exist_ok=True)

        filename = f"{filename_prefix}_{ts}.txt"
        filepath = os.path.join(output_path, filename)
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)

        return {
            "message": f"Saved output to {filename}",
            "file_path": filepath,
        }
    except Exception as exc:
        raise RuntimeError(f"Failed to save file: {exc}") from exc


class SaveArgs(BaseModel):
    content: str = Field(description="The exact final response text to write into the .txt file.")
    filename_prefix: Optional[str] = Field(
        default="response",
        description="Optional short prefix for the filename (no extension).",
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

INDEX_PATH = "data/index"


def _load_chroma() -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        embedding_function=embeddings,
        persist_directory=INDEX_PATH,
        collection_name="notebooks",
    )


def rag_search(query: str, k: int = 4) -> str:
    """Search local notebook index and return relevant snippets with sources."""
    if not os.path.isdir(INDEX_PATH):
        return "RAG index not found. Please build it first (python -m src.rag_build)."

    db = _load_chroma()
    docs = db.similarity_search(query, k=k)
    if not docs:
        return "No relevant passages found in local notebooks."

    lines = []
    for index, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "notebook")
        snippet = (doc.page_content or "").strip().replace("\n", " ")
        if len(snippet) > 500:
            snippet = snippet[:500] + " ..."
        lines.append(f"{index}. {snippet}\n   [로컬 예제] {source}")

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


class SlackArgs(BaseModel):
    text: str = Field(description="Slack으로 보낼 최종 메시지(plain text).")
    user_id: Optional[str] = Field(default=None, description="Slack Uxxxxx (DM 대상).")
    email: Optional[str] = Field(default=None, description="Slack 이메일 (DM 대상).")
    channel_id: Optional[str] = Field(default=None, description="채널 ID (Cxxxx/Gxxxx/Dxxxx). 제공되면 우선 사용.")
    target: str = Field(default="auto", description="auto|dm|channel|group")


def slack_notify(
    text: str,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    channel_id: Optional[str] = None,
    target: str = "auto",
) -> dict:
    """Send a message to Slack (DM/channel/group) and return a structured status."""
    _ = target

    if not slack_client:
        return {"status": "skipped", "reason": "SLACK_BOT_TOKEN not set"}

    resolved_id, target_type = resolve_destination(
        slack_client=slack_client,
        channel_id=channel_id,
        user_id=user_id,
        email=email,
        default_user_id=SLACK_DEFAULT_USER_ID,
        default_email=SLACK_DEFAULT_DM_EMAIL,
    )

    if not resolved_id:
        return {"status": "skipped", "reason": "No valid Slack destination resolved"}

    try:
        slack_client.chat_postMessage(channel=resolved_id, text=text)
        return {"status": "ok", "channel_id": resolved_id, "target_type": target_type}
    except SlackApiError as exc:
        return {"status": "error", "error": str(exc)}


slack_notify_tool = StructuredTool.from_function(
    name="slack_notify",
    description=(
        "Send a message to Slack. Use when the user asks to DM or post the answer to Slack. "
        "Provide either channel_id (C/G/D...) or a user_id/email for DM. "
        "If neither is present, the tool tries environment defaults."
    ),
    func=slack_notify,
    args_schema=SlackArgs,
)

tools = [tavilysearch, rag_search_tool, save_text_tool, slack_notify_tool]
