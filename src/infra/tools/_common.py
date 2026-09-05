from __future__ import annotations

import json
import math
from typing import Any, Literal

from src.core.evidence import DocMetadata, EvidenceItem, build_local_source_id, dedupe_evidence, evidence_to_dicts, normalize_source_id, truncate_snippet


def build_retrieval_payload(
    *,
    tool: str,
    route: Literal["docs", "upload"],
    query: str,
    evidence: list[dict[str, Any]] | None = None,
    status: Literal["success", "no_result", "error", "unavailable"] = "success",
    message: str = "",
    normalized_score: float | None = None,
    raw_score: float | None = None,
    provider_ms: int = 0,
    url_validation_ms: int = 0,
    post_filter_ms: int = 0,
    include_raw_content_requested: bool = False,
    result_count: int | None = None,
    provider_result_count: int | None = None,
    filtered_invalid_url_count: int = 0,
    filtered_path_prefix_count: int = 0,
    filtered_cross_domain_count: int = 0,
    filtered_http_error_count: int = 0,
    filtered_redirect_policy_count: int = 0,
    filtered_url_request_failed_count: int = 0,
    filtered_identifier_mismatch_count: int = 0,
    validated_url_count: int = 0,
    final_evidence_count: int | None = None,
    metric: str | None = None,
    score_direction: Literal["higher_is_better", "lower_is_better"] | None = None,
    warnings: list[str] | None = None,
    error_code: str | None = None,
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
            "error_code": str(error_code or "").strip().upper() or None,
            "query": query,
            "evidence_count": len(evidence_items),
            "metric": resolved_metric,
            "score_direction": resolved_score_direction,
            "normalized_score": resolved_normalized_score,
            "raw_score": raw_score,
            "provider_ms": max(0, int(provider_ms)),
            "url_validation_ms": max(0, int(url_validation_ms)),
            "post_filter_ms": max(0, int(post_filter_ms)),
            "include_raw_content_requested": bool(include_raw_content_requested),
            "result_count": len(evidence_items) if result_count is None else max(0, int(result_count)),
            "provider_result_count": 0 if provider_result_count is None else max(0, int(provider_result_count)),
            "filtered_invalid_url_count": max(0, int(filtered_invalid_url_count)),
            "filtered_path_prefix_count": max(0, int(filtered_path_prefix_count)),
            "filtered_cross_domain_count": max(0, int(filtered_cross_domain_count)),
            "filtered_http_error_count": max(0, int(filtered_http_error_count)),
            "filtered_redirect_policy_count": max(0, int(filtered_redirect_policy_count)),
            "filtered_url_request_failed_count": max(0, int(filtered_url_request_failed_count)),
            "filtered_identifier_mismatch_count": max(0, int(filtered_identifier_mismatch_count)),
            "validated_url_count": max(0, int(validated_url_count)),
            "final_evidence_count": len(evidence_items) if final_evidence_count is None else max(0, int(final_evidence_count)),
            "warnings": [str(item).strip() for item in (warnings or []) if str(item).strip()],
        },
    }


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


def normalize_code_metadata(value: Any) -> dict[str, Any] | None:
    payload = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None

    normalized: dict[str, Any] = {}
    cell_id = payload.get("cell_id")
    if cell_id is not None:
        try:
            normalized["cell_id"] = max(0, int(cell_id))
        except (TypeError, ValueError):
            pass

    calls = []
    for call in payload.get("calls") or []:
        if not isinstance(call, dict):
            continue
        call_name = str(call.get("call_name") or "").strip()
        if not call_name:
            continue
        call_payload: dict[str, Any] = {"call_name": call_name}
        kwargs = call.get("kwargs")
        if isinstance(kwargs, dict):
            normalized_kwargs = {
                str(key): str(item)
                for key, item in kwargs.items()
                if str(key).strip() and str(item).strip()
            }
            if normalized_kwargs:
                call_payload["kwargs"] = normalized_kwargs
        line = call.get("line")
        try:
            if line is not None:
                call_payload["line"] = max(1, int(line))
        except (TypeError, ValueError):
            pass
        calls.append(call_payload)
    if calls:
        normalized["calls"] = calls

    option_literals = []
    seen_options: set[str] = set()
    for option in payload.get("option_literals") or []:
        option_text = " ".join(str(option or "").split())
        compact = "".join(option_text.lower().split())
        if not option_text or compact in seen_options:
            continue
        option_literals.append(option_text)
        seen_options.add(compact)
    if option_literals:
        normalized["option_literals"] = option_literals

    return normalized or None


def normalize_doc_metadata(value: Any) -> DocMetadata | None:
    payload = value
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        return None
    try:
        metadata = DocMetadata.model_validate(payload)
    except Exception:
        return None
    if not any(
        (
            metadata.doc_family,
            metadata.symbol,
            metadata.signature,
            metadata.parameters,
            metadata.returns,
            metadata.options,
            metadata.examples,
            metadata.notes,
            metadata.source_sections,
        )
    ):
        return None
    return metadata


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
    code_metadata = normalize_code_metadata(metadata.get("code_metadata"))
    doc_metadata = normalize_doc_metadata(metadata.get("doc_metadata"))
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
        code_metadata=code_metadata,
        doc_metadata=doc_metadata,
    )


def dedupe_evidence_dicts(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    return evidence_to_dicts(dedupe_evidence(items))
