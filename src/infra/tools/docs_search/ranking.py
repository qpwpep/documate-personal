from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

from src.core.answer_schema import clean_grounded_text
from src.core.evidence import EvidenceItem
from src.core.rules import get_rules_config
from src.infra.tools.docs_search.normalization import (
    normalize_identifier_reference_text,
    normalize_identifier_token,
)


def tokenize_topic_terms(text: str) -> set[str]:
    stopwords = {"official", "docs", "documentation", "reference"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.:/-]+", str(text or ""))
        if len(token) >= 2 and token.lower() not in stopwords
    }


_ASCII_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9._-]*|v\d+)(?![A-Za-z0-9_])"
)


def _identifier_stopwords(*, library_name: str = "") -> set[str]:
    stopwords = {item.lower() for item in get_rules_config().planner.docs_identifier_stopwords}
    stopwords.update(
        {
            "api",
            "parameters",
            "parameter",
            "reference",
            "validation",
            "validator",
        }
    )
    for part in re.findall(r"[A-Za-z0-9_.-]+", str(library_name or "").lower()):
        stopwords.add(part)
    return stopwords


def extract_exact_identifier_terms(query: str, *, library_name: str = "") -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    stopwords = _identifier_stopwords(library_name=library_name)
    normalized_query = normalize_identifier_reference_text(query)
    for token in _ASCII_IDENTIFIER_PATTERN.findall(normalized_query):
        normalized = normalize_identifier_token(token)
        lowered = normalized.lower()
        if not normalized or lowered in stopwords:
            continue
        if re.fullmatch(r"v\d+", lowered):
            continue
        if not (
            "." in normalized
            or "_" in normalized
            or "-" in normalized
            or normalized != normalized.lower()
        ):
            continue
        if lowered not in seen:
            identifiers.append(normalized)
            seen.add(lowered)
    return identifiers


def has_exact_identifier_coverage(
    query: str,
    evidence_items: list[dict[str, Any]],
    *,
    library_name: str = "",
) -> bool:
    required_identifiers = extract_exact_identifier_terms(query, library_name=library_name)
    if not required_identifiers:
        return True
    combined_text = " ".join(
        part
        for item in evidence_items
        for part in (
            str(item.get("title") or ""),
            str(item.get("url_or_path") or ""),
            str(item.get("snippet") or ""),
        )
        if part
    )
    normalized_combined_text = normalize_identifier_reference_text(combined_text)
    return all(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            normalized_combined_text,
            flags=re.I,
        )
        is not None
        for identifier in required_identifiers
    )


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


def query_requests_api_detail(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(
        marker in lowered
        for marker in (
            "api",
            "reference",
            "signature",
            "option",
            "options",
            "parameter",
            "parameters",
            "argument",
            "arguments",
            "옵션",
            "파라미터",
            "매개변수",
            "인자",
        )
    )


def api_reference_preference_score(query: str, evidence_item: dict[str, Any]) -> float:
    if not query_requests_api_detail(query):
        return 0.0

    url = str(evidence_item.get("url_or_path") or "").lower()
    title = str(evidence_item.get("title") or "").lower()
    metadata = evidence_item.get("doc_metadata")
    score = 0.0

    if "/api/_as_gen/" in url:
        score += 6.0
    if "/reference/generated/" in url or "/reference/api/" in url:
        score += 4.0
    if "/plot_types/" in url or "/gallery/" in url:
        score -= 3.0

    if isinstance(metadata, dict):
        if metadata.get("parameters") or metadata.get("options"):
            score += 5.0
        if metadata.get("signature"):
            score += 2.0
        symbol = str(metadata.get("symbol") or "").lower()
        if "." in symbol:
            score += 1.0
            if symbol and symbol in f"{title} {url}":
                score += 1.0

    if "matplotlib.pyplot.pie" in f"{title} {url}":
        score += 4.0
    return score


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
        key=lambda item: (
            api_reference_preference_score(query, item),
            entity_hit_score(query, item),
            float(item.get("score") or 0.0),
        ),
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


def merge_docs_evidence_items(items: list[EvidenceItem]) -> list[EvidenceItem]:
    merged_by_source: dict[str, EvidenceItem] = {}
    ordered_source_ids: list[str] = []
    for item in items:
        source_id = str(item.source_id or "").strip()
        if not source_id:
            continue
        current = merged_by_source.get(source_id)
        if current is None:
            merged_by_source[source_id] = item
            ordered_source_ids.append(source_id)
            continue
        merged_updates = {
            "title": _merge_unique_text(
                current.title,
                item.title,
            )
            or None,
            "snippet": _merge_unique_text(
                current.snippet,
                item.snippet,
            )
            or None,
            "score": max(float(current.score or 0.0), float(item.score or 0.0)),
        }
        merged_by_source[source_id] = current.model_copy(update=merged_updates)
    return [merged_by_source[source_id] for source_id in ordered_source_ids]


def _merge_unique_text(*values: Any) -> str:
    lines: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(str(value or "").split()).strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        lines.append(normalized)
        seen.add(key)
    return "\n".join(lines)
