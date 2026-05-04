from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from src.infra.tools._common import build_evidence_item, normalize_relevance_score
from src.infra.tools.docs_search.extraction import extract_doc_content
from src.infra.tools.docs_search.policy import canonicalize_doc_title, canonicalize_doc_url, is_valid_doc_result, normalize_domain, normalized_domain_set, result_matches_domains


def url_domain(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    return normalize_domain(parsed.netloc)


def filter_evidence_to_domains(
    evidence: list[dict[str, Any]],
    *,
    allowed_domains: list[str],
) -> list[dict[str, Any]]:
    normalized_domains = normalized_domain_set(allowed_domains)
    if not normalized_domains:
        return evidence
    return [
        item
        for item in evidence
        if url_domain(str(item.get("url_or_path") or "")) in normalized_domains
    ]


def collect_docs_search_evidence(
    results: list[dict[str, Any]],
    *,
    allowed_domains: list[str] | None,
    retrieval_warnings: list[str],
    query: str = "",
) -> tuple[list[Any], list[float]]:
    evidence_items: list[Any] = []
    raw_scores: list[float] = []
    normalized_domains = normalized_domain_set(allowed_domains)
    for result in results:
        if not isinstance(result, dict):
            continue
        original_url = str(result.get("url") or "").strip()
        url = canonicalize_doc_url(original_url)
        title = canonicalize_doc_title(
            title=result.get("title"),
            original_url=original_url,
            canonical_url=url,
        )
        if not is_valid_doc_result(
            url=url,
            title=title,
            snippet=result.get("content"),
        ):
            continue
        if not result_matches_domains(url, normalized_domains):
            if "cross_library_domain_filtered" not in retrieval_warnings:
                retrieval_warnings.append("cross_library_domain_filtered")
            continue
        normalized_score, raw_score = normalize_relevance_score(
            result.get("score"),
            warnings=retrieval_warnings,
        )
        metadata: dict[str, Any] = {}
        snippet = result.get("content")
        raw_content = result.get("raw_content")
        if raw_content:
            doc_metadata, structured_snippet = extract_doc_content(
                url=url,
                title=title,
                content=raw_content,
                query=query,
            )
            if doc_metadata:
                metadata["doc_metadata"] = doc_metadata
                if structured_snippet:
                    snippet = structured_snippet
            elif structured_snippet:
                snippet = structured_snippet
        evidence_item = build_evidence_item(
            kind="official",
            tool="tavily_search",
            url_or_path=url,
            title=title,
            snippet=snippet,
            score=normalized_score,
            metadata=metadata,
            warnings=retrieval_warnings,
        )
        if evidence_item is not None:
            evidence_items.append(evidence_item)
            if raw_score is not None:
                raw_scores.append(raw_score)
    return evidence_items, raw_scores
