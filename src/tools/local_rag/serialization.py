from __future__ import annotations

from typing import Any

from ...chroma_store import normalize_l2_distance
from .._common import build_evidence_item, dedupe_evidence_dicts, to_float_or_none
from .ranking import extract_identifiers, extract_keywords


def score_ranked_rows(
    docs_with_scores: list[tuple[Any, float | None]],
) -> list[tuple[Any, float | None, float | None]]:
    scored_rows: list[tuple[Any, float | None, float | None]] = []
    for doc, score in docs_with_scores:
        raw_score = to_float_or_none(score)
        scored_rows.append((doc, normalize_l2_distance(raw_score), raw_score))
    return scored_rows


def build_query_focused_snippet(text: str, *, query: str, max_length: int = 500) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_length:
        return normalized

    query_identifiers = extract_identifiers(query)
    lowered_text = normalized.lower()
    best_index: int | None = None
    for token in query_identifiers:
        index = lowered_text.find(token.lower())
        if index < 0:
            continue
        best_index = index if best_index is None else min(best_index, index)

    if best_index is None:
        keywords = sorted(extract_keywords(query), key=len, reverse=True)
        for keyword in keywords:
            index = lowered_text.find(keyword.lower())
            if index < 0:
                continue
            best_index = index if best_index is None else min(best_index, index)

    if best_index is None:
        return normalized[:max_length]

    line_start = normalized.rfind("\n", 0, best_index)
    if line_start < 0:
        line_start = max(0, best_index - (max_length // 3))
    else:
        line_start += 1
    snippet = normalized[line_start : line_start + max_length]
    return snippet.strip()


def build_local_evidence_bundle(
    docs_with_scores: list[tuple[Any, float | None]],
    *,
    query: str,
    tool_name: str,
    default_source: str,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[str]]:
    evidence_items = []
    retrieval_warnings: list[str] = []
    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    for doc, score, raw_score in score_ranked_rows(docs_with_scores):
        if not hasattr(doc, "metadata"):
            continue
        source = doc.metadata.get("source", default_source)
        evidence_item = build_evidence_item(
            kind="local",
            tool=tool_name,
            url_or_path=str(source),
            snippet=build_query_focused_snippet(
                doc.page_content or "",
                query=query,
            ).replace("\n", " "),
            score=score,
            metadata=getattr(doc, "metadata", None),
            warnings=retrieval_warnings,
        )
        if evidence_item is not None:
            evidence_items.append(evidence_item)
            if raw_score is not None:
                raw_scores.append(raw_score)
            if score is not None:
                normalized_scores.append(score)
    return dedupe_evidence_dicts(evidence_items), normalized_scores, raw_scores, retrieval_warnings
