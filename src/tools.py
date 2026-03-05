from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Literal, Optional
from urllib.parse import urlparse

from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilySearch
from langgraph.prebuilt import InjectedState
from pydantic import BaseModel, Field
from slack_sdk.errors import SlackApiError

from src.util.util import get_save_text_output_dir

from .domain_docs import DEFAULT_DOCS
from .evidence import EvidenceItem, dedupe_evidence, evidence_to_dicts, normalize_source_id, truncate_snippet
from .settings import AppSettings
from .slack_utils import create_slack_client, resolve_destination


INDEX_PATH = Path("data/index")


@dataclass(frozen=True)
class ToolRegistry:
    tavily_search_tool: Any
    rag_search_tool: Any
    upload_search_tool: Any
    save_text_tool: Any
    slack_notify_tool: Any
    all_tools: list[Any]


class TavilyArgs(BaseModel):
    query: str = Field(description="Search query for official documentation.")
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = Field(
        default="basic",
        description="Search depth for Tavily.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Optional domain whitelist for this query.",
    )


class SaveArgs(BaseModel):
    content: str = Field(description="The exact final response text to write into the .txt file.")
    filename_prefix: Optional[str] = Field(
        default="response",
        description="Optional short prefix for the filename (no extension).",
    )


class RagArgs(BaseModel):
    query: str = Field(description="The user's information need to search over local notebooks.")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to return.")


class UploadArgs(BaseModel):
    query: str = Field(description="The user's information need to search over uploaded files.")
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


def _to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _build_evidence_item(
    *,
    kind: Literal["official", "local"],
    tool: str,
    url_or_path: str,
    title: Any = None,
    snippet: Any = None,
    score: Any = None,
) -> EvidenceItem | None:
    source = str(url_or_path or "").strip()
    if not source:
        return None

    source_id = normalize_source_id(source)
    if not source_id:
        return None

    title_text = str(title).strip() if title else None
    return EvidenceItem(
        kind=kind,
        tool=tool,
        source_id=source_id,
        url_or_path=source,
        title=title_text or None,
        snippet=truncate_snippet(str(snippet) if snippet else None),
        score=_to_float_or_none(score),
    )


def build_tool_registry(settings: AppSettings) -> ToolRegistry:
    default_domains = _normalize_include_domains(list(DEFAULT_DOCS.values()))
    tavily_client = TavilySearch(
        max_results=3,
        include_domains=default_domains,
        tavily_api_key=settings.tavily_api_key,
    )

    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search official web docs and return typed evidence items."""
        domains = _normalize_include_domains(include_domains or default_domains)
        payload: dict[str, Any] = {
            "query": query,
            "search_depth": search_depth,
            "include_domains": domains,
        }
        try:
            raw_results = tavily_client.invoke(payload)
        except Exception:
            return []

        if not isinstance(raw_results, dict):
            return []
        results = raw_results.get("results")
        if not isinstance(results, list):
            return []

        evidence_items: list[EvidenceItem] = []
        for result in results:
            if not isinstance(result, dict):
                continue
            evidence_item = _build_evidence_item(
                kind="official",
                tool="tavily_search",
                url_or_path=str(result.get("url") or "").strip(),
                title=result.get("title"),
                snippet=result.get("content"),
                score=result.get("score"),
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)

        return evidence_to_dicts(dedupe_evidence(evidence_items))

    tavily_search_tool = StructuredTool.from_function(
        name="tavily_search",
        description=(
            "Search official documentation on the web and return structured evidence items. "
            "Use this for current or official references."
        ),
        func=tavily_search,
        args_schema=TavilyArgs,
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

    def rag_search(query: str, k: int = 4) -> list[dict[str, Any]]:
        """Search local notebook index and return typed evidence items."""
        if not INDEX_PATH.is_dir():
            return []
        if not settings.openai_api_key:
            return []

        db = _load_chroma(settings.openai_api_key)
        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            raw_docs_with_scores = db.similarity_search_with_relevance_scores(query, k=k)
            for doc, score in raw_docs_with_scores:
                docs_with_scores.append((doc, _to_float_or_none(score)))
        except Exception:
            docs = db.similarity_search(query, k=k)
            docs_with_scores = [(doc, None) for doc in docs]

        evidence_items: list[EvidenceItem] = []
        for doc, score in docs_with_scores:
            if not hasattr(doc, "metadata"):
                continue
            source = doc.metadata.get("source", "notebook")
            evidence_item = _build_evidence_item(
                kind="local",
                tool="rag_search",
                url_or_path=str(source),
                snippet=(doc.page_content or "").replace("\n", " "),
                score=score,
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)

        return evidence_to_dicts(dedupe_evidence(evidence_items))

    rag_search_tool = StructuredTool.from_function(
        name="rag_search",
        description=(
            "Search local .ipynb notebooks (vector index) and return structured evidence items. "
            "Use this when the question is covered by our local documents."
        ),
        func=rag_search,
        args_schema=RagArgs,
    )

    def upload_search(
        query: str,
        k: int = 4,
        retriever: Annotated[Any, InjectedState("retriever")] = None,
    ) -> list[dict[str, Any]]:
        """Search uploaded file retriever and return typed evidence items."""
        if retriever is None:
            return []

        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            vectorstore = getattr(retriever, "vectorstore", None)
            if vectorstore is not None and hasattr(vectorstore, "similarity_search_with_relevance_scores"):
                raw_docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
                for doc, score in raw_docs_with_scores:
                    docs_with_scores.append((doc, _to_float_or_none(score)))
            else:
                docs = retriever.invoke(query)
                docs_with_scores = [(doc, None) for doc in docs]
        except Exception:
            return []

        evidence_items: list[EvidenceItem] = []
        for doc, score in docs_with_scores:
            if not hasattr(doc, "metadata"):
                continue
            source = doc.metadata.get("source", "uploaded")
            evidence_item = _build_evidence_item(
                kind="local",
                tool="upload_search",
                url_or_path=str(source),
                snippet=(doc.page_content or "").replace("\n", " "),
                score=score,
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)
        return evidence_to_dicts(dedupe_evidence(evidence_items))

    upload_search_tool = StructuredTool.from_function(
        name="upload_search",
        description=(
            "Search only the currently uploaded file context and return structured evidence items. "
            "Use this when user asks about uploaded file content."
        ),
        func=upload_search,
        args_schema=UploadArgs,
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

    all_tools = [tavily_search_tool, rag_search_tool, upload_search_tool, save_text_tool, slack_notify_tool]
    return ToolRegistry(
        tavily_search_tool=tavily_search_tool,
        rag_search_tool=rag_search_tool,
        upload_search_tool=upload_search_tool,
        save_text_tool=save_text_tool,
        slack_notify_tool=slack_notify_tool,
        all_tools=all_tools,
    )
