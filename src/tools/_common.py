from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..evidence import (
    EvidenceItem,
    build_local_source_id,
    dedupe_evidence,
    evidence_to_dicts,
    normalize_source_id,
    truncate_snippet,
)


def build_retrieval_payload(
    *,
    tool: str,
    route: Literal["docs", "upload", "local"],
    query: str,
    evidence: list[dict[str, Any]] | None = None,
    status: Literal["success", "no_result", "error", "unavailable"] = "success",
    message: str = "",
) -> dict[str, Any]:
    return {
        "evidence": list(evidence or []),
        "diagnostics": {
            "tool": tool,
            "route": route,
            "status": status,
            "message": message,
            "query": query,
        },
    }


class SaveArgs(BaseModel):
    content: str = Field(description="The exact final response text to write into the .txt file.")
    filename_prefix: str | None = Field(
        default="response",
        description="Optional short prefix for the filename (no extension).",
    )


class SlackArgs(BaseModel):
    text: str = Field(description="Final message to send to Slack (plain text).")
    user_id: str | None = Field(default=None, description="Slack Uxxxxx user id for DM.")
    email: str | None = Field(default=None, description="Slack email for DM.")
    channel_id: str | None = Field(default=None, description="Slack channel id (C/G/D...).")
    target: str = Field(default="auto", description="auto|dm|channel|group")


class RagArgs(BaseModel):
    query: str = Field(description="The user's information need to search over local notebooks.")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to return.")


class UploadArgs(BaseModel):
    query: str = Field(description="The user's information need to search over uploaded files.")
    k: int = Field(default=4, ge=1, le=10, description="Number of chunks to return.")


def to_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_evidence_item(
    *,
    kind: Literal["official", "local"],
    tool: str,
    url_or_path: str,
    title: Any = None,
    snippet: Any = None,
    score: Any = None,
    metadata: dict[str, Any] | None = None,
) -> EvidenceItem | None:
    source = str(url_or_path or "").strip()
    if not source:
        return None

    metadata = dict(metadata or {})
    document_id = normalize_source_id(source)
    if not document_id:
        return None

    chunk_id = metadata.get("chunk_id")
    cell_id = metadata.get("cell_id")
    start_offset = metadata.get("start_offset", metadata.get("start_index"))
    end_offset = metadata.get("end_offset")
    if end_offset is None and start_offset is not None:
        end_offset = int(start_offset or 0) + len(str(snippet or ""))

    source_id = document_id
    if kind == "local":
        source_id = build_local_source_id(
            url_or_path=source,
            chunk_id=chunk_id,
            start_offset=start_offset,
            end_offset=end_offset,
            cell_id=cell_id if str(source).lower().endswith(".ipynb") else None,
        )
        if not source_id:
            return None

    title_text = str(title).strip() if title else None
    return EvidenceItem(
        kind=kind,
        tool=tool,
        source_id=source_id,
        document_id=document_id,
        url_or_path=source,
        title=title_text or None,
        snippet=truncate_snippet(str(snippet) if snippet else None),
        score=to_float_or_none(score),
        chunk_id=int(chunk_id) if chunk_id is not None else None,
        cell_id=int(cell_id) if cell_id is not None else None,
        start_offset=int(start_offset) if start_offset is not None else None,
        end_offset=int(end_offset) if end_offset is not None else None,
    )


def dedupe_evidence_dicts(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    return evidence_to_dicts(dedupe_evidence(items))
