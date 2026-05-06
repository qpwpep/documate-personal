from __future__ import annotations

import time
from typing import Any, Literal

from langchain_core.tools import StructuredTool

from src.core.latency import elapsed_ms
from src.core.evidence import evidence_to_dicts
from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.extraction import should_extract_doc_content
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_query_hint, normalize_include_domains
from src.infra.tools.docs_search.ranking import filter_docs_evidence_by_topic_purity, has_exact_identifier_coverage, has_meaningful_docs_evidence, merge_docs_evidence_items
from src.infra.tools.docs_search.schemas import TavilyArgs
from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_evidence


def build_docs_search_tool(settings: AppSettings) -> Any:
    default_domains = list(docs_search_rules().allowed_doc_path_prefixes.keys())

    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        domains = normalize_include_domains(include_domains or default_domains)
        effective_query = str(query or "").strip()
        fallback_queries: list[str] = []
        hinted_domains: list[str] | None = None
        library_name = ""
        if include_domains is None:
            query_hint = infer_docs_query_hint(effective_query)
            if query_hint is not None:
                library_name, hinted_domains, fallback_queries = query_hint
                domains = normalize_include_domains(hinted_domains)
                if library_name.lower() not in effective_query.lower():
                    effective_query = f"{effective_query} {library_name}".strip()
        include_raw_content = "markdown" if should_extract_doc_content(effective_query) else False
        provider_ms = 0
        post_processing_ms = 0
        include_raw_content_requested = bool(include_raw_content)
        try:
            provider_started = time.perf_counter()
            raw_results = client.request_tavily_search(
                query=effective_query,
                tavily_api_key=settings.tavily_api_key,
                include_domains=domains,
                search_depth=search_depth,
                timeout_seconds=settings.docs_search_timeout_seconds,
                include_raw_content=include_raw_content,
            )
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
        except Exception as exc:
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
            is_timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message=f"invoke failed ({exc})",
                provider_ms=provider_ms,
                include_raw_content_requested=include_raw_content_requested,
                error_code="RETRIEVAL_DOCS_TIMEOUT" if is_timeout else "RETRIEVAL_DOCS_FAILED",
            )

        if not isinstance(raw_results, dict):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="unexpected response type from Tavily",
                provider_ms=provider_ms,
                include_raw_content_requested=include_raw_content_requested,
                error_code="RETRIEVAL_DOCS_FAILED",
            )
        results = raw_results.get("results")
        if not isinstance(results, list):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="missing or invalid Tavily results payload",
                provider_ms=provider_ms,
                include_raw_content_requested=include_raw_content_requested,
                error_code="RETRIEVAL_DOCS_FAILED",
            )

        evidence_items = []
        retrieval_warnings: list[str] = []
        raw_scores: list[float] = []
        filter_counters = DocsSearchFilterCounters()
        post_started = time.perf_counter()
        batch_evidence, batch_raw_scores = collect_docs_search_evidence(
            results,
            allowed_domains=hinted_domains,
            retrieval_warnings=retrieval_warnings,
            query=effective_query,
            filter_counters=filter_counters,
        )
        post_processing_ms += elapsed_ms(post_started, time.perf_counter())
        evidence_items.extend(batch_evidence)
        raw_scores.extend(batch_raw_scores)

        for fallback_query in fallback_queries:
            post_started = time.perf_counter()
            deduped_batch = evidence_to_dicts(merge_docs_evidence_items(evidence_items))
            filtered_batch = filter_docs_evidence_by_topic_purity(
                effective_query,
                deduped_batch,
                retrieval_warnings,
            )
            post_processing_ms += elapsed_ms(post_started, time.perf_counter())
            if has_meaningful_docs_evidence(filtered_batch):
                if has_exact_identifier_coverage(
                    effective_query,
                    filtered_batch,
                    library_name=library_name,
                ):
                    break
            provider_started = time.perf_counter()
            try:
                fallback_include_raw_content = (
                    "markdown"
                    if include_raw_content or should_extract_doc_content(fallback_query)
                    else False
                )
                include_raw_content_requested = include_raw_content_requested or bool(fallback_include_raw_content)
                provider_started = time.perf_counter()
                fallback_results = client.request_tavily_search(
                    query=fallback_query,
                    tavily_api_key=settings.tavily_api_key,
                    include_domains=domains,
                    search_depth=search_depth,
                    timeout_seconds=settings.docs_search_timeout_seconds,
                    include_raw_content=fallback_include_raw_content,
                )
                provider_ms += elapsed_ms(provider_started, time.perf_counter())
            except Exception:
                provider_ms += elapsed_ms(provider_started, time.perf_counter())
                continue
            fallback_items = fallback_results.get("results") if isinstance(fallback_results, dict) else None
            if not isinstance(fallback_items, list):
                continue
            post_started = time.perf_counter()
            batch_evidence, batch_raw_scores = collect_docs_search_evidence(
                fallback_items,
                allowed_domains=hinted_domains,
                retrieval_warnings=retrieval_warnings,
                query=fallback_query,
                filter_counters=filter_counters,
            )
            post_processing_ms += elapsed_ms(post_started, time.perf_counter())
            evidence_items.extend(batch_evidence)
            raw_scores.extend(batch_raw_scores)

        post_started = time.perf_counter()
        evidence = evidence_to_dicts(merge_docs_evidence_items(evidence_items))
        evidence = filter_docs_evidence_by_topic_purity(effective_query, evidence, retrieval_warnings)
        if evidence and not has_meaningful_docs_evidence(evidence):
            retrieval_warnings.append("docs_chrome_only")
            evidence = []
        if evidence and not has_exact_identifier_coverage(
            effective_query,
            evidence,
            library_name=library_name,
        ):
            retrieval_warnings.append("identifier_coverage_incomplete")
            filter_counters.filtered_identifier_mismatch_count += len(evidence)
            evidence = []
        post_processing_ms += elapsed_ms(post_started, time.perf_counter())
        return build_retrieval_payload(
            tool="tavily_search",
            route="docs",
            query=effective_query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no official documentation evidence found",
            raw_score=max(raw_scores) if raw_scores else None,
            provider_ms=provider_ms,
            url_validation_ms=filter_counters.url_validation_ms,
            post_filter_ms=max(0, post_processing_ms - filter_counters.url_validation_ms),
            include_raw_content_requested=include_raw_content_requested,
            provider_result_count=filter_counters.provider_result_count,
            filtered_invalid_url_count=filter_counters.filtered_invalid_url_count,
            filtered_path_prefix_count=filter_counters.filtered_path_prefix_count,
            filtered_cross_domain_count=filter_counters.filtered_cross_domain_count,
            filtered_http_error_count=filter_counters.filtered_http_error_count,
            filtered_redirect_policy_count=filter_counters.filtered_redirect_policy_count,
            filtered_url_request_failed_count=filter_counters.filtered_url_request_failed_count,
            filtered_identifier_mismatch_count=filter_counters.filtered_identifier_mismatch_count,
            validated_url_count=filter_counters.validated_url_count,
            final_evidence_count=len(evidence),
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
