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

from slack_sdk.web import WebClient
from slack_sdk.errors import SlackApiError

from src.util.util import get_save_text_output_dir 

# ─────────────────────────────────────────────
# 1. Environment setup
# ─────────────────────────────────────────────
load_dotenv(find_dotenv(), override=True)

TAVILY_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
SLACK_BOT_TOKEN = os.getenv("SLACK_BOT_TOKEN")

assert TAVILY_KEY, "Missing TAVILY_API_KEY in environment (.env not loaded or key not set)."
assert OPENAI_KEY, "Missing OPENAI_API_KEY in environment (.env not loaded or key not set)."

slack_client = WebClient(token=SLACK_BOT_TOKEN) if SLACK_BOT_TOKEN else None

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
def save_text_to_file(content: str, filename_prefix: str = "response") -> dict:
    try:
        """Save text content to a timestamped .txt file and return the file path message."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        output_path = get_save_text_output_dir()
        os.makedirs(output_path, exist_ok=True)
        filename = f"{filename_prefix}_{ts}.txt"
        filepath = os.path.join(output_path, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        
        return {
            "message": f"Saved output to {filename}",
            "file_path": filepath  # 파일의 절대 경로
        }
    
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {e}")

# LLM (Agent)이 툴을 호출할 때 넘겨줘야 하는 입력 인자를 정의
# save_text_to_file > args_schema
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
        return "RAG index not found. Please build it first (python -m src.rag_build)."

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
# 5. Slack Notify tool
# ─────────────────────────────────────────────

def _resolve_user_id(user_id: Optional[str], email: Optional[str]) -> Optional[str]:
    """우선순위: user_id → email → 환경변수 기본값"""
    if user_id and user_id.startswith("U"):
        return user_id
    if email and slack_client:
        try:
            r = slack_client.users_lookupByEmail(email=email)
            return r["user"]["id"]
        except SlackApiError:
            pass
    if SLACK_DEFAULT_USER_ID:
        return SLACK_DEFAULT_USER_ID
    if SLACK_DEFAULT_DM_EMAIL and slack_client:
        try:
            r = slack_client.users_lookupByEmail(email=SLACK_DEFAULT_DM_EMAIL)
            return r["user"]["id"]
        except SlackApiError:
            pass
    return None

def _open_dm_channel(uid: str) -> Optional[str]:
    if not slack_client:
        return None
    try:
        r = slack_client.conversations_open(users=uid)
        return r["channel"]["id"]  # Dxxxx...
    except SlackApiError:
        return None

class SlackArgs(BaseModel):
    text: str = Field(description="Slack으로 보낼 최종 메시지(plain text).")
    user_id: Optional[str] = Field(default=None, description="Slack Uxxxxx (DM 보낼 대상).")
    email: Optional[str] = Field(default=None, description="Slack 이메일 (DM 보낼 대상).")
    channel_id: Optional[str] = Field(default=None, description="채널 ID (Cxxxx/Gxxxx/Dxxxx). 제공되면 우선 사용.")
    target: str = Field(default="auto", description="auto|dm|channel|group")

def slack_notify(text: str,
                 user_id: Optional[str] = None,
                 email: Optional[str] = None,
                 channel_id: Optional[str] = None,
                 target: str = "auto") -> dict:
    """
    Slack 메시지 전송 (DM/채널/그룹).
    모델이 호출하는 Tool. 성공 시 channel_id/target_type/status를 반환.
    """
    if not slack_client:
        return {"status": "skipped", "reason": "SLACK_BOT_TOKEN not set"}

    resolved_id = None
    target_type = "Unknown"

    # 0) 명시 채널 우선
    if channel_id:
        resolved_id = channel_id
        if resolved_id.startswith("D"):
            target_type = "DM"
        elif resolved_id.startswith("C"):
            target_type = "Public Channel"
        elif resolved_id.startswith("G"):
            target_type = "Private Channel"
        else:
            target_type = "Unknown Channel"

    # 1) DM 대상 지정
    if not resolved_id and (user_id or email or SLACK_DEFAULT_USER_ID or SLACK_DEFAULT_DM_EMAIL):
        uid = _resolve_user_id(user_id, email)
        if uid and uid.startswith("U"):
            dm_id = _open_dm_channel(uid)
            if dm_id:
                resolved_id = dm_id
                target_type = "DM"

    if not resolved_id:
        return {"status": "skipped", "reason": "No valid Slack destination resolved"}

    # 2) target 유효성 경고는 툴 내에서는 로깅 없이 그대로 전송
    try:
        slack_client.chat_postMessage(channel=resolved_id, text=text)
        return {"status": "ok", "channel_id": resolved_id, "target_type": target_type}
    except SlackApiError as e:
        return {"status": "error", "error": str(e)}

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

# ─────────────────────────────────────────────
# 6. Export
# ─────────────────────────────────────────────
tools = [tavilysearch, rag_search_tool, save_text_tool, slack_notify_tool]