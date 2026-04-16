from __future__ import annotations

import math
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
    normalized_score: float | None = None,
    raw_score: float | None = None,
    result_count: int | None = None,
    metric: str | None = None,
    score_direction: Literal["higher_is_better", "lower_is_better"] | None = None,
    warnings: list[str] | None = None,
) -> dict[str, Any]:
    evidence_items = list(evidence or [])
    resolved_metric = metric or ("provider_score" if route == "docs" else "l2")
    resolved_score_direction = (
        score_direction or ("higher_is_better" if route == "docs" else "lower_is_better")
    )
    resolved_normalized_score = normalized_score
    if resolved_normalized_score is None:
        scores = [
            float(item.get("score"))
            for item in evidence_items
            if isinstance(item, dict) and item.get("score") is not None
        ]
        if scores:
            resolved_normalized_score = max(0.0, min(1.0, max(scores)))
    return {
        "evidence": evidence_items,
        "diagnostics": {
            "tool": tool,
            "route": route,
            "status": status,
            "message": message,
            "query": query,
            "evidence_count": len(evidence_items),
            "metric": resolved_metric,
            "score_direction": resolved_score_direction,
            "normalized_score": resolved_normalized_score,
            "raw_score": raw_score,
            "result_count": len(evidence_items) if result_count is None else max(0, int(result_count)),
            "warnings": [str(item).strip() for item in (warnings or []) if str(item).strip()],
        },
    }


class SaveArgs(BaseModel):
    content: str = Field(
        description=(
            "The exact final message body to write into the .txt file. "
            "When the user asked to save in this turn, this should be the self-contained final answer body generated for that request."
        )
    )
    filename_prefix: str | None = Field(
        default="response",
        description="Optional short prefix for the filename (no extension).",
    )


class SlackArgs(BaseModel):
    text: str = Field(
        description=(
            "Final plain-text message body to send to Slack. "
            "When the user asked to share in this turn, this should be the exact self-contained final answer body generated for that request."
        )
    )
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


def normalize_relevance_score(
    value: Any,
    *,
    warnings: list[str] | None = None,
) -> tuple[float | None, float | None]:
    raw_score = to_float_or_none(value)
    if raw_score is None or not math.isfinite(raw_score):
        if value is not None and warnings is not None:
            warnings.append("invalid_relevance_score")
        return None, None

    normalized_score = raw_score
    if raw_score < 0.0 or raw_score > 1.0:
        normalized_score = max(0.0, min(1.0, raw_score))
        if warnings is not None:
            warnings.append("relevance_score_clamped")
    return normalized_score, raw_score


def normalize_notebook_cell_id(
    value: Any,
    *,
    is_notebook: bool,
    warnings: list[str] | None = None,
) -> int | None:
    if not is_notebook:
        if value is not None and warnings is not None:
            warnings.append("non_notebook_cell_id_dropped")
        return None
    if value is None:
        return None
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        digits = "".join(ch for ch in str(value) if ch.isdigit())
        if digits:
            if warnings is not None:
                warnings.append("notebook_cell_id_normalized")
            normalized = int(digits)
        else:
            if warnings is not None:
                warnings.append("notebook_cell_id_normalized")
            return 0
    if normalized < 0:
        if warnings is not None:
            warnings.append("notebook_cell_id_normalized")
        return 0
    return normalized


def build_evidence_item(
    *,
    kind: Literal["official", "local"],
    tool: str,
    url_or_path: str,
    title: Any = None,
    snippet: Any = None,
    score: Any = None,
    metadata: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
) -> EvidenceItem | None:
    source = str(url_or_path or "").strip()
    if not source:
        return None

    metadata = dict(metadata or {})
    document_id = normalize_source_id(source)
    if not document_id:
        return None

    chunk_id = metadata.get("chunk_id")
    is_notebook = str(source).lower().endswith(".ipynb")
    cell_id = normalize_notebook_cell_id(
        metadata.get("cell_id"),
        is_notebook=is_notebook,
        warnings=warnings,
    )
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
            cell_id=cell_id,
        )
        if not source_id:
            return None

    title_text = str(title).strip() if title else None
    normalized_score, _raw_score = normalize_relevance_score(score, warnings=warnings)
    snippet_text = str(snippet).strip() if snippet else None
    if kind != "local":
        snippet_text = truncate_snippet(snippet_text)
    return EvidenceItem(
        kind=kind,
        tool=tool,
        source_id=source_id,
        document_id=document_id,
        url_or_path=source,
        title=title_text or None,
        snippet=snippet_text,
        score=normalized_score,
        chunk_id=int(chunk_id) if chunk_id is not None else None,
        cell_id=cell_id,
        start_offset=int(start_offset) if start_offset is not None else None,
        end_offset=int(end_offset) if end_offset is not None else None,
    )


def dedupe_evidence_dicts(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    return evidence_to_dicts(dedupe_evidence(items))
