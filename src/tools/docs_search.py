from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..rules import get_rules_config
from ..settings import AppSettings
from ._common import (
    build_evidence_item,
    build_retrieval_payload,
    dedupe_evidence_dicts,
    normalize_relevance_score,
)


TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"


def _docs_search_rules():
    return get_rules_config().docs_search


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


def normalize_include_domains(raw_values: list[str]) -> list[str]:
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


def _normalize_domain(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate


def _normalize_path_prefix(path: str) -> str:
    normalized = "/" + str(path or "").strip().lstrip("/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized == "/":
        return normalized
    return normalized if normalized.endswith("/") else normalized + "/"


def is_allowed_doc_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    domain = parsed.netloc.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    allowed_prefixes = _docs_search_rules().allowed_doc_path_prefixes.get(domain)
    if not allowed_prefixes:
        return False
    normalized_path = _normalize_path_prefix(parsed.path or "/")
    return any(normalized_path.startswith(prefix) for prefix in allowed_prefixes)


def is_valid_doc_result(*, url: str, title: Any, snippet: Any) -> bool:
    if not is_allowed_doc_url(url):
        return False

    combined = " ".join(
        part.strip().lower()
        for part in [str(url or ""), str(title or ""), str(snippet or "")]
        if str(part or "").strip()
    )
    if not combined:
        return False

    return not any(marker in combined for marker in _docs_search_rules().error_page_markers)


def _url_domain(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    return _normalize_domain(parsed.netloc)


def filter_evidence_to_domains(
    evidence: list[dict[str, Any]],
    *,
    allowed_domains: list[str],
) -> list[dict[str, Any]]:
    normalized_domains = _normalized_domain_set(allowed_domains)
    if not normalized_domains:
        return evidence
    return [
        item
        for item in evidence
        if _url_domain(str(item.get("url_or_path") or "")) in normalized_domains
    ]


def _normalized_domain_set(allowed_domains: list[str] | None) -> set[str]:
    if not allowed_domains:
        return set()
    return {_normalize_domain(domain) for domain in allowed_domains if _normalize_domain(domain)}


def _result_matches_domains(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    return _url_domain(url) in allowed_domains


def _tokenize_topic_terms(text: str) -> set[str]:
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.:/-]+", str(text or ""))
        if len(token) >= 2
    }


def _entity_hit_score(query: str, evidence_item: dict[str, Any]) -> float:
    query_terms = _tokenize_topic_terms(query)
    haystack = " ".join(
        [
            str(evidence_item.get("title") or ""),
            str(evidence_item.get("url_or_path") or ""),
            str(evidence_item.get("snippet") or ""),
        ]
    ).lower()
    return float(sum(1 for token in query_terms if token in haystack))


def _path_cluster(value: str) -> str:
    parsed = urlparse(str(value or ""))
    parts = [part for part in str(parsed.path or "").split("/") if part]
    return "/".join(parts[:4]).lower()


def _filter_docs_evidence_by_topic_purity(
    query: str,
    evidence_items: list[dict[str, Any]],
    retrieval_warnings: list[str],
) -> list[dict[str, Any]]:
    if len(evidence_items) <= 1:
        return evidence_items

    ranked = sorted(
        evidence_items,
        key=lambda item: (_entity_hit_score(query, item), float(item.get("score") or 0.0)),
        reverse=True,
    )
    anchor = ranked[0]
    anchor_cluster = _path_cluster(str(anchor.get("url_or_path") or ""))
    kept = [anchor]
    for item in ranked[1:]:
        same_cluster = _path_cluster(str(item.get("url_or_path") or "")) == anchor_cluster
        strong_entity_match = _entity_hit_score(query, item) >= 2.0
        if same_cluster or strong_entity_match:
            kept.append(item)
    if len(kept) < len(evidence_items):
        retrieval_warnings.append("topic_purity_pruned")
    return kept[:2]


def _collect_docs_search_evidence(
    results: list[dict[str, Any]],
    *,
    allowed_domains: list[str] | None,
    retrieval_warnings: list[str],
) -> tuple[list[Any], list[float]]:
    evidence_items: list[Any] = []
    raw_scores: list[float] = []
    normalized_domains = _normalized_domain_set(allowed_domains)
    for result in results:
        if not isinstance(result, dict):
            continue
        url = str(result.get("url") or "").strip()
        if not is_valid_doc_result(
            url=url,
            title=result.get("title"),
            snippet=result.get("content"),
        ):
            continue
        if not _result_matches_domains(url, normalized_domains):
            if "cross_library_domain_filtered" not in retrieval_warnings:
                retrieval_warnings.append("cross_library_domain_filtered")
            continue
        normalized_score, raw_score = normalize_relevance_score(
            result.get("score"),
            warnings=retrieval_warnings,
        )
        evidence_item = build_evidence_item(
            kind="official",
            tool="tavily_search",
            url_or_path=url,
            title=result.get("title"),
            snippet=result.get("content"),
            score=normalized_score,
            metadata={},
            warnings=retrieval_warnings,
        )
        if evidence_item is not None:
            evidence_items.append(evidence_item)
            if raw_score is not None:
                raw_scores.append(raw_score)
    return evidence_items, raw_scores


def request_tavily_search(
    *,
    query: str,
    tavily_api_key: str | None,
    include_domains: list[str],
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"],
    timeout_seconds: int,
    max_results: int = 3,
) -> dict[str, Any]:
    if not tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not configured")

    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json",
        "X-Client-Source": "documate",
    }
    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": "general",
        "include_domains": include_domains,
    }

    try:
        response = requests.post(
            TAVILY_SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise TimeoutError(f"Tavily search timed out after {timeout_seconds}s") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Tavily request failed ({exc})") from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("invalid JSON response from Tavily") from exc

    if response.status_code != 200:
        detail = body.get("detail") if isinstance(body, dict) else None
        error_message = detail.get("error") if isinstance(detail, dict) else None
        if not error_message:
            error_message = f"HTTP {response.status_code}"
        raise RuntimeError(str(error_message))

    if not isinstance(body, dict):
        raise RuntimeError("unexpected response type from Tavily")
    return body


def infer_docs_query_hint(query: str) -> tuple[str, list[str], list[str]] | None:
    lowered = str(query or "").lower()
    for hint in _docs_search_rules().query_hints:
        if any(_query_hint_matches(lowered, identifier, match_mode=hint.match_mode) for identifier in hint.identifiers):
            return hint.library_name, list(hint.domains), list(hint.fallback_queries)
    return None


def _query_hint_matches(query: str, identifier: str, *, match_mode: Literal["contains", "word"]) -> bool:
    normalized_identifier = str(identifier or "").strip().lower()
    if not normalized_identifier:
        return False
    if match_mode == "word":
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(normalized_identifier)}(?![A-Za-z0-9_])", query) is not None
    return normalized_identifier in query


def build_docs_search_tool(settings: AppSettings) -> Any:
    default_domains = list(_docs_search_rules().allowed_doc_path_prefixes.keys())

    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        domains = normalize_include_domains(include_domains or default_domains)
        effective_query = str(query or "").strip()
        fallback_queries: list[str] = []
        hinted_domains: list[str] | None = None
        if include_domains is None:
            query_hint = infer_docs_query_hint(effective_query)
            if query_hint is not None:
                library_name, hinted_domains, fallback_queries = query_hint
                domains = normalize_include_domains(hinted_domains)
                if library_name.lower() not in effective_query.lower():
                    effective_query = f"{effective_query} {library_name}".strip()
        try:
            raw_results = request_tavily_search(
                query=effective_query,
                tavily_api_key=settings.tavily_api_key,
                include_domains=domains,
                search_depth=search_depth,
                timeout_seconds=settings.docs_search_timeout_seconds,
            )
        except Exception as exc:
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message=f"invoke failed ({exc})",
            )

        if not isinstance(raw_results, dict):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="unexpected response type from Tavily",
            )
        results = raw_results.get("results")
        if not isinstance(results, list):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="missing or invalid Tavily results payload",
            )

        evidence_items = []
        retrieval_warnings: list[str] = []
        raw_scores: list[float] = []
        batch_evidence, batch_raw_scores = _collect_docs_search_evidence(
            results,
            allowed_domains=hinted_domains,
            retrieval_warnings=retrieval_warnings,
        )
        evidence_items.extend(batch_evidence)
        raw_scores.extend(batch_raw_scores)

        for fallback_query in fallback_queries:
            if evidence_items:
                break
            try:
                fallback_results = request_tavily_search(
                    query=fallback_query,
                    tavily_api_key=settings.tavily_api_key,
                    include_domains=domains,
                    search_depth=search_depth,
                    timeout_seconds=settings.docs_search_timeout_seconds,
                )
            except Exception:
                continue
            fallback_items = fallback_results.get("results") if isinstance(fallback_results, dict) else None
            if not isinstance(fallback_items, list):
                continue
            batch_evidence, batch_raw_scores = _collect_docs_search_evidence(
                fallback_items,
                allowed_domains=hinted_domains,
                retrieval_warnings=retrieval_warnings,
            )
            evidence_items.extend(batch_evidence)
            raw_scores.extend(batch_raw_scores)

        evidence = dedupe_evidence_dicts(evidence_items)
        evidence = _filter_docs_evidence_by_topic_purity(effective_query, evidence, retrieval_warnings)
        return build_retrieval_payload(
            tool="tavily_search",
            route="docs",
            query=effective_query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no official documentation evidence found",
            raw_relevance_score=max(raw_scores) if raw_scores else None,
            warnings=sorted(set(retrieval_warnings)),
        )

    return StructuredTool.from_function(
        name="tavily_search",
        description=(
            "Search official documentation on the web and return structured evidence items. "
            "Use this for current or official references."
        ),
        func=tavily_search,
        args_schema=TavilyArgs,
    )
