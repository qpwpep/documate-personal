from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from src.core.latency import elapsed_ms
from src.core.evidence import evidence_to_dicts
from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.extraction import should_extract_doc_content
from src.infra.tools.docs_search.normalization import canonicalize_docs_query_text
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_query_hint, normalize_include_domains
from src.infra.tools.docs_search.ranking import filter_docs_evidence_by_topic_purity, has_exact_identifier_coverage, has_meaningful_docs_evidence, merge_docs_evidence_items
from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_evidence


def build_docs_search_tool(settings: AppSettings) -> Callable[..., dict[str, Any]]:
    default_domains = list(docs_search_rules().allowed_doc_path_prefixes.keys())

    def _request_docs_search(query_text: str, domains: list[str], search_depth: str, include_raw_content: Any) -> dict[str, Any]:
        return client.request_tavily_search(
            query=query_text,
            tavily_api_key=settings.tavily_api_key,
            include_domains=domains,
            search_depth=search_depth,  # type: ignore[arg-type]
            timeout_seconds=settings.docs_search_timeout_seconds,
            include_raw_content=include_raw_content,
        )

    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        domains = normalize_include_domains(include_domains or default_domains)
        effective_query = canonicalize_docs_query_text(query)
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
        fallback_plan: list[tuple[str, Any]] = []

        planned_queries = {effective_query}
        for fallback_query in fallback_queries:
            fallback_query = str(fallback_query or "").strip()
            if not fallback_query or fallback_query in planned_queries:
                continue
            fallback_include_raw_content = (
                "markdown"
                if include_raw_content or should_extract_doc_content(fallback_query)
                else False
            )
            planned_queries.add(fallback_query)
            fallback_plan.append((fallback_query, fallback_include_raw_content))

        def request_provider_query(query_text: str, fallback_raw_content: Any) -> dict[str, Any]:
            nonlocal include_raw_content_requested, provider_ms
            include_raw_content_requested = include_raw_content_requested or bool(fallback_raw_content)
            provider_started = time.perf_counter()
            try:
                payload = _request_docs_search(
                    query_text,
                    domains,
                    search_depth,
                    fallback_raw_content,
                )
            finally:
                provider_ms += elapsed_ms(provider_started, time.perf_counter())
            if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                raise RuntimeError("missing or invalid Tavily results payload")
            return payload

        def is_timeout_error(exc: Exception) -> bool:
            message = str(exc).lower()
            return isinstance(exc, TimeoutError) or "timeout" in message or "timed out" in message

        try:
            raw_results = request_provider_query(effective_query, include_raw_content)
        except Exception as exc:
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message=f"invoke failed ({exc})",
                provider_ms=provider_ms,
                include_raw_content_requested=include_raw_content_requested,
                error_code="RETRIEVAL_DOCS_TIMEOUT" if is_timeout_error(exc) else "RETRIEVAL_DOCS_FAILED",
            )

        results = raw_results.get("results")

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

        fallback_errors: list[Exception] = []
        fallback_success_count = 0
        for fallback_query, fallback_include_raw_content in fallback_plan:
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
            try:
                fallback_results = request_provider_query(fallback_query, fallback_include_raw_content)
            except Exception as exc:
                fallback_errors.append(exc)
                retrieval_warnings.append(
                    "docs_fallback_query_timeout" if is_timeout_error(exc) else "docs_fallback_query_failed"
                )
                continue
            fallback_success_count += 1
            fallback_items = fallback_results["results"]
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
        fallback_failure = bool(not evidence and fallback_errors and fallback_success_count == 0)
        fallback_failure_message = ""
        fallback_failure_code: str | None = None
        if fallback_failure:
            fallback_failure_message = f"all fallback queries failed ({fallback_errors[-1]})"
            fallback_failure_code = (
                "RETRIEVAL_DOCS_TIMEOUT"
                if all(is_timeout_error(exc) for exc in fallback_errors)
                else "RETRIEVAL_DOCS_FAILED"
            )
        return build_retrieval_payload(
            tool="tavily_search",
            route="docs",
            query=effective_query,
            evidence=evidence,
            status="success" if evidence else ("error" if fallback_failure else "no_result"),
            message=(
                ""
                if evidence
                else fallback_failure_message or "no official documentation evidence found"
            ),
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
            error_code=fallback_failure_code,
        )

    return tavily_search
