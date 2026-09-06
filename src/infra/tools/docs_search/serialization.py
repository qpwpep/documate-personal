from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import re
import time
from typing import Any
from urllib.parse import urlparse

from src.core.latency import elapsed_ms
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.infra.tools._common import normalize_relevance_score
from src.infra.tools.docs_search.extraction import extract_doc_content
from src.infra.tools.docs_search.policy import (
    canonicalize_doc_url,
    doc_url_filter_reason,
    is_valid_doc_result,
    normalize_domain,
    normalized_domain_set,
    result_matches_domains,
)
from src.infra.tools.docs_search.url_validation import validate_doc_url


@dataclass(slots=True)
class DocsSearchFilterCounters:
    provider_result_count: int = 0
    filtered_invalid_url_count: int = 0
    filtered_path_prefix_count: int = 0
    filtered_cross_domain_count: int = 0
    filtered_http_error_count: int = 0
    filtered_redirect_policy_count: int = 0
    filtered_url_request_failed_count: int = 0
    filtered_identifier_mismatch_count: int = 0
    validated_url_count: int = 0
    url_validation_ms: int = 0

    def record_url_filter(self, reason: str | None) -> None:
        if reason == "invalid_url":
            self.filtered_invalid_url_count += 1
        elif reason == "path_prefix":
            self.filtered_path_prefix_count += 1

    def record_validation_filter(self, reason: str | None) -> None:
        if reason == "http_error":
            self.filtered_http_error_count += 1
        elif reason == "redirect_policy":
            self.filtered_redirect_policy_count += 1
        elif reason == "request_failed":
            self.filtered_url_request_failed_count += 1


@dataclass(slots=True)
class _DocsResultCandidate:
    result: dict[str, Any]
    original_url: str
    candidates: list[str]
    resolved_url: str = ""


def _validate_candidate_urls(
    candidates: list[_DocsResultCandidate],
    *,
    retrieval_warnings: list[str],
    filter_counters: DocsSearchFilterCounters | None,
) -> None:
    max_candidate_count = max((len(item.candidates) for item in candidates), default=0)
    for candidate_index in range(max_candidate_count):
        jobs: list[tuple[int, str]] = []
        for index, item in enumerate(candidates):
            if item.resolved_url or candidate_index >= len(item.candidates):
                continue
            candidate_url = item.candidates[candidate_index]
            url_filter_reason = doc_url_filter_reason(candidate_url)
            if url_filter_reason is not None:
                if filter_counters is not None:
                    filter_counters.record_url_filter(url_filter_reason)
                continue
            jobs.append((index, candidate_url))

        if not jobs:
            continue

        validation_started = time.perf_counter()
        if len(jobs) == 1:
            validations = [(jobs[0][0], jobs[0][1], validate_doc_url(jobs[0][1]))]
        else:
            max_workers = min(4, len(jobs))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_by_job = {
                    executor.submit(validate_doc_url, candidate_url): (index, candidate_url)
                    for index, candidate_url in jobs
                }
                validation_by_index = {
                    index: (candidate_url, future.result())
                    for future, (index, candidate_url) in future_by_job.items()
                }
            validations = [
                (index, candidate_url, validation_by_index[index][1])
                for index, candidate_url in jobs
            ]
        if filter_counters is not None:
            filter_counters.url_validation_ms += elapsed_ms(validation_started, time.perf_counter())

        for index, _candidate_url, validation in validations:
            item = candidates[index]
            if item.resolved_url:
                continue
            if not validation.ok:
                if filter_counters is not None:
                    filter_counters.record_validation_filter(validation.reason)
                warning = f"url_{validation.reason}_filtered" if validation.reason else "url_validation_filtered"
                if warning not in retrieval_warnings:
                    retrieval_warnings.append(warning)
                continue
            if filter_counters is not None:
                filter_counters.validated_url_count += 1
            item.resolved_url = validation.final_url


def url_domain(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    return normalize_domain(parsed.netloc)


def filter_hits_to_domains(
    hits: list[dict[str, Any]],
    *,
    allowed_domains: list[str],
) -> list[dict[str, Any]]:
    normalized_domains = normalized_domain_set(allowed_domains)
    if not normalized_domains:
        return hits
    return [
        item
        for item in hits
        if url_domain(str(item["evidence"]["snapshot"]["source_uri"])) in normalized_domains
    ]


def _candidate_doc_urls(original_url: str) -> list[str]:
    canonical_url = canonicalize_doc_url(original_url)
    candidates: list[str] = []
    for candidate in (canonical_url, str(original_url or "").strip()):
        if candidate and candidate not in candidates:
            candidates.append(candidate)
    return candidates


def collect_docs_search_hits(
    results: list[dict[str, Any]],
    *,
    allowed_domains: list[str] | None,
    retrieval_warnings: list[str],
    query: str = "",
    filter_counters: DocsSearchFilterCounters | None = None,
) -> tuple[list[SearchHit], list[float]]:
    hits: list[SearchHit] = []
    raw_scores: list[float] = []
    normalized_domains = normalized_domain_set(allowed_domains)
    if filter_counters is not None:
        filter_counters.provider_result_count += len(results)

    candidates: list[_DocsResultCandidate] = []
    for result in results:
        if not isinstance(result, dict):
            if filter_counters is not None:
                filter_counters.filtered_invalid_url_count += 1
            continue
        original_url = str(result.get("url") or "").strip()
        candidates.append(
            _DocsResultCandidate(
                result=result,
                original_url=original_url,
                candidates=_candidate_doc_urls(original_url),
            )
        )

    _validate_candidate_urls(
        candidates,
        retrieval_warnings=retrieval_warnings,
        filter_counters=filter_counters,
    )

    for candidate in candidates:
        result = candidate.result
        original_url = candidate.original_url
        url = candidate.resolved_url
        if not url:
            continue
        title = str(result.get("title") or "").strip() or original_url
        if not is_valid_doc_result(
            url=url,
            title=title,
            snippet=result.get("content"),
        ):
            continue
        if not result_matches_domains(url, normalized_domains):
            if filter_counters is not None:
                filter_counters.filtered_cross_domain_count += 1
            if "cross_library_domain_filtered" not in retrieval_warnings:
                retrieval_warnings.append("cross_library_domain_filtered")
            continue
        normalized_score, raw_score = normalize_relevance_score(
            result.get("score"),
            warnings=retrieval_warnings,
        )
        # Validation may choose a stable alias; the provider's content still
        # belongs to the URL/version it actually returned.
        if not result_matches_domains(original_url, normalized_domains):
            if filter_counters is not None:
                filter_counters.filtered_cross_domain_count += 1
            if "cross_library_domain_filtered" not in retrieval_warnings:
                retrieval_warnings.append("cross_library_domain_filtered")
            continue
        metadata: dict[str, Any] = {"validated_url": url}
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
                query="",
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
        if raw_score is not None:
            raw_scores.append(raw_score)
    return hits, raw_scores


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
