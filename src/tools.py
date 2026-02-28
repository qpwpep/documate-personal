from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError

from src.util.util import get_save_text_output_dir

from .domain_docs import DEFAULT_DOCS
from .settings import AppSettings
from .slack_utils import create_slack_client, resolve_destination


INDEX_PATH = Path("data/index")


@dataclass(frozen=True)
class ToolRegistry:
    tavilysearch: Any
    rag_search_tool: Any
    save_text_tool: Any
    slack_notify_tool: Any
    all_tools: list[Any]


class SaveArgs(BaseModel):
    content: str = Field(description="The exact final response text to write into the .txt file.")
    filename_prefix: Optional[str] = Field(
        default="response",
        description="Optional short prefix for the filename (no extension).",
    )


class RagArgs(BaseModel):
    query: str = Field(description="The user's information need to search over local notebooks.")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to return.")


class SlackArgs(BaseModel):
    text: str = Field(description="Final message to send to Slack (plain text).")
    user_id: Optional[str] = Field(default=None, description="Slack Uxxxxx user id for DM.")
    email: Optional[str] = Field(default=None, description="Slack email for DM.")
    channel_id: Optional[str] = Field(default=None, description="Slack channel id (C/G/D...).")
    target: str = Field(default="auto", description="auto|dm|channel|group")


def _normalize_include_domains(raw_values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in raw_values:
        candidate = value.strip()
        if not candidate:
            continue
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
        domain = (parsed.netloc or parsed.path).strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if domain and domain not in normalized:
            normalized.append(domain)
    return normalized


def _load_chroma(openai_api_key: str) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(INDEX_PATH),
        collection_name="notebooks",
    )


def build_tool_registry(settings: AppSettings) -> ToolRegistry:
    tavilysearch = TavilySearch(
        max_results=3,
        include_domains=_normalize_include_domains(list(DEFAULT_DOCS.values())),
        tavily_api_key=settings.tavily_api_key,
    )

    def save_text_to_file(content: str, filename_prefix: str = "response") -> dict:
        """Save text content to a timestamped .txt file and return a status payload."""
        try:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = Path(get_save_text_output_dir())
            output_path.mkdir(parents=True, exist_ok=True)

            filename = f"{filename_prefix}_{ts}.txt"
            filepath = output_path / filename
            filepath.write_text(content, encoding="utf-8")

            return {
                "message": f"Saved output to {filename}",
                "file_path": str(filepath),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to save file: {exc}") from exc

    save_text_tool = StructuredTool.from_function(
        name="save_text",
        description=(
            "Save the given final response text into a timestamped .txt file in the current directory. "
            "Call this at most ONCE per user request. If you already saved, do not call again."
        ),
        func=save_text_to_file,
        args_schema=SaveArgs,
    )

    def rag_search(query: str, k: int = 4) -> str:
        """Search local notebook index and return relevant snippets with sources."""
        if not INDEX_PATH.is_dir():
            return "RAG index not found. Please build it first (python -m src.rag_build)."
        if not settings.openai_api_key:
            return "RAG is unavailable because OPENAI_API_KEY is missing."

        db = _load_chroma(settings.openai_api_key)
        docs = db.similarity_search(query, k=k)
        if not docs:
            return "No relevant passages found in local notebooks."

        lines = []
        for index, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "notebook")
            snippet = (doc.page_content or "").strip().replace("\n", " ")
            if len(snippet) > 500:
                snippet = snippet[:500] + " ..."
            lines.append(f"{index}. {snippet}\n   [Local Example] {source}")

        return "\n".join(lines)

    rag_search_tool = StructuredTool.from_function(
        name="rag_search",
        description=(
            "Search local .ipynb notebooks (vector index) and return relevant snippets with sources. "
            "Use this when the question is covered by our local documents."
        ),
        func=rag_search,
        args_schema=RagArgs,
    )

    slack_client = create_slack_client(settings.slack_bot_token)

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
            default_user_id=settings.slack_default_user_id,
            default_email=settings.slack_default_dm_email,
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

    all_tools = [tavilysearch, rag_search_tool, save_text_tool, slack_notify_tool]
    return ToolRegistry(
        tavilysearch=tavilysearch,
        rag_search_tool=rag_search_tool,
        save_text_tool=save_text_tool,
        slack_notify_tool=slack_notify_tool,
        all_tools=all_tools,
    )
