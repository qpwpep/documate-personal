from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.infra.tools._common import normalize_relevance_score
from src.infra.tools.docs_search.extraction import extract_doc_content
from src.infra.tools.docs_search.policy import (
    doc_url_filter_reason,
    normalized_domain_set,
    result_matches_domains,
)


@dataclass(slots=True)
class DocsSearchFilterCounters:
    provider_result_count: int = 0
    filtered_invalid_url_count: int = 0
    filtered_path_prefix_count: int = 0
    filtered_cross_domain_count: int = 0
    filtered_identifier_mismatch_count: int = 0

    def record_url_filter(self, reason: str | None) -> None:
        if reason == "invalid_url":
            self.filtered_invalid_url_count += 1
        elif reason == "path_prefix":
            self.filtered_path_prefix_count += 1


def collect_docs_search_hits(
    results: list[dict[str, Any]],
    *,
    allowed_domains: list[str] | None,
    retrieval_warnings: list[str],
    query: str = "",
    filter_counters: DocsSearchFilterCounters | None = None,
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    normalized_domains = normalized_domain_set(allowed_domains)
    if filter_counters is not None:
        filter_counters.provider_result_count += len(results)

    for result in results:
        if not isinstance(result, dict):
            if filter_counters is not None:
                filter_counters.filtered_invalid_url_count += 1
            continue
        original_url = str(result.get("url") or "").strip()
        reason = doc_url_filter_reason(original_url)
        if reason is not None:
            if filter_counters is not None:
                filter_counters.record_url_filter(reason)
            continue
        if not result_matches_domains(original_url, normalized_domains):
            if filter_counters is not None:
                filter_counters.filtered_cross_domain_count += 1
            if "cross_library_domain_filtered" not in retrieval_warnings:
                retrieval_warnings.append("cross_library_domain_filtered")
            continue
        title = str(result.get("title") or "").strip() or original_url
        normalized_score, raw_score = normalize_relevance_score(
            result.get("score"),
            warnings=retrieval_warnings,
        )
        metadata: dict[str, Any] = {}
        raw_content = result.get("raw_content")
        provider_field = "raw_content" if isinstance(raw_content, str) and raw_content.strip() else "content"
        content = result.get(provider_field)
        if not isinstance(content, str) or not content.strip():
            continue
        if provider_field == "raw_content":
            doc_metadata, structured_snippet = extract_doc_content(
                url=original_url,
                title=title,
                content=content,
            )
            if doc_metadata:
                metadata["doc_metadata"] = doc_metadata
            if structured_snippet:
                metadata["search_summary"] = structured_snippet
        snapshot = build_snapshot(
            source_uri=original_url,
            title=title,
            media_type="text/markdown" if provider_field == "raw_content" else "text/plain",
            source_type="official",
            content=content,
            parser="tavily",
            parser_version="1",
            parser_config={"provider_field": provider_field},
            capture_scope="provider_excerpt",
        )
        element = DocumentElement(
            element_id=f"{snapshot.snapshot_id}:body",
            kind="paragraph",
            text=content,
            order=0,
            anchors=[SourceAnchor(kind="web", start=0, end=len(content), precision="exact")],
            metadata=metadata,
        )
        start, end = _select_excerpt_range(content, query=query)
        hits.append(SearchHit(
            evidence=build_evidence(snapshot=snapshot, element=element, start=start, end=end),
            score=RetrievalScore(metric="provider_score", raw=raw_score, direction="higher", normalized=normalized_score),
            rank=len(hits) + 1,
        ))
    return hits


def _select_excerpt_range(text: str, *, query: str, max_chars: int = 3200) -> tuple[int, int]:
    """Select one contiguous range; the citation remains a slice of provider text."""
    if len(text) <= max_chars:
        return 0, len(text)
    identifiers = re.findall(r"[A-Za-z_][A-Za-z0-9_.-]{2,}", query)
    ignored = {"official", "documentation", "reference", "docs", "parameters", "options"}
    positions = [
        position
        for token in identifiers
        if token.lower() not in ignored
        for position in [text.lower().find(token.lower())]
        if position >= 0
    ]
    target = min(positions, default=0)
    start = max(0, min(target - max_chars // 4, len(text) - max_chars))
    return start, min(len(text), start + max_chars)
