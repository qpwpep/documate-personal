from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from src.core.answer_schema import clean_grounded_text


def tokenize_topic_terms(text: str) -> set[str]:
    stopwords = {"official", "docs", "documentation", "reference"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.:/-]+", str(text or ""))
        if len(token) >= 2 and token.lower() not in stopwords
    }


def entity_hit_score(query: str, evidence_item: dict[str, Any]) -> float:
    query_terms = tokenize_topic_terms(query)
    haystack = " ".join(
        [
            str(evidence_item.get("title") or ""),
            str(evidence_item.get("url_or_path") or ""),
            str(evidence_item.get("snippet") or ""),
        ]
    ).lower()
    return float(sum(1 for token in query_terms if token in haystack))


def path_cluster(value: str) -> str:
    parsed = urlparse(str(value or ""))
    parts = [part for part in str(parsed.path or "").split("/") if part]
    return "/".join(parts[:4]).lower()


def filter_docs_evidence_by_topic_purity(
    query: str,
    evidence_items: list[dict[str, Any]],
    retrieval_warnings: list[str],
) -> list[dict[str, Any]]:
    if len(evidence_items) <= 1:
        return evidence_items

    ranked = sorted(
        evidence_items,
        key=lambda item: (entity_hit_score(query, item), float(item.get("score") or 0.0)),
        reverse=True,
    )
    if entity_hit_score(query, ranked[0]) <= 0.0:
        return ranked[:2]
    anchor = ranked[0]
    anchor_cluster = path_cluster(str(anchor.get("url_or_path") or ""))
    kept = [anchor]
    for item in ranked[1:]:
        same_cluster = path_cluster(str(item.get("url_or_path") or "")) == anchor_cluster
        strong_entity_match = entity_hit_score(query, item) >= 2.0
        if same_cluster or strong_entity_match:
            kept.append(item)
    if len(kept) < len(evidence_items):
        retrieval_warnings.append("topic_purity_pruned")
    return kept[:2]


def evidence_item_has_grounded_text(item: dict[str, Any]) -> bool:
    if not isinstance(item, dict):
        return False
    cleaned_snippet = clean_grounded_text(str(item.get("snippet") or ""))
    cleaned_title = clean_grounded_text(str(item.get("title") or ""))
    if cleaned_snippet or cleaned_title:
        return True
    combined_raw = " ".join(
        part.strip().lower()
        for part in (str(item.get("title") or ""), str(item.get("snippet") or ""))
        if part and part.strip()
    )
    chrome_markers = (
        "table of contents",
        "on this page",
        "previous:",
        "next:",
        "skip to content",
        "edit this page",
        "view source",
        "home >",
    )
    return not any(marker in combined_raw for marker in chrome_markers)


def has_meaningful_docs_evidence(evidence_items: list[dict[str, Any]]) -> bool:
    return any(evidence_item_has_grounded_text(item) for item in evidence_items)


def docs_evidence_preference(item: Any) -> tuple[int, int, float]:
    grounded_snippet = clean_grounded_text(str(getattr(item, "snippet", "") or ""))
    grounded_title = clean_grounded_text(str(getattr(item, "title", "") or ""))
    score = float(getattr(item, "score", 0.0) or 0.0)
    return (
        1 if grounded_snippet or grounded_title else 0,
        len(grounded_snippet),
        score,
    )


def merge_docs_evidence_items(items: list[Any]) -> list[Any]:
    merged_by_source: dict[str, Any] = {}
    ordered_source_ids: list[str] = []
    for item in items:
        source_id = str(getattr(item, "source_id", "") or "").strip()
        if not source_id:
            continue
        current = merged_by_source.get(source_id)
        if current is None:
            merged_by_source[source_id] = item
            ordered_source_ids.append(source_id)
            continue
        if docs_evidence_preference(item) > docs_evidence_preference(current):
            merged_by_source[source_id] = item
    return [merged_by_source[source_id] for source_id in ordered_source_ids]
