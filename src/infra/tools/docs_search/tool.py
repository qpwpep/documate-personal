from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.core.evidence import SearchHit
from src.core.contracts.debug import DocsSearchRequest, DocsSearchProviderRequest
from src.core.latency import elapsed_ms
from src.core.planner_schema import RetrievalRequirement
from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.policy import docs_search_rules, normalize_include_domains
from src.infra.tools.docs_search.ranking import dedupe_docs_hits, select_docs_evidence
from src.infra.tools.docs_search.requirements import canonical_query, library_domains, reformulate_query, resolve_requirement
from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_hits


def build_docs_search_tool(settings: AppSettings) -> Callable[..., dict[str, Any]]:
    def tavily_search(
        query: str,
        *,
        requirement: RetrievalRequirement,
        k: int = 3,
        completed_requests: list[DocsSearchRequest] | None = None,
        previous_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        requested = resolve_requirement(requirement)
        effective_query = canonical_query(query, requested)
        domains = normalize_include_domains(library_domains(str(requested.library or ""))
                                             or list(docs_search_rules().allowed_doc_path_prefixes))
        candidate_limit = min(20, max(6, k * 3))
        include_raw = "markdown" if requested.symbols or requested.aspects or requested.evidence_needs else False
        completed: dict[str, DocsSearchRequest] = {}
        for request in completed_requests or []:
            key = request.query
            if key not in completed or completed[key].max_results < request.max_results:
                completed[key] = request
        provider_requests: list[DocsSearchProviderRequest] = []
        unsupported_library = bool(requested.library and not library_domains(requested.library))
        candidates = dedupe_docs_hits([hit for hit in previous_hits or [] if hit.evidence.route == "docs"])
        if unsupported_library:
            candidates = []
        warnings: list[str] = []
        counters = DocsSearchFilterCounters()
        provider_ms = post_ms = 0
        assessment = select_docs_evidence(effective_query, candidates, requested, k=k,
                                          allowed_domains=domains, retrieval_warnings=warnings)
        error_code = None
        error_message = ""
        for attempt in range(2):
            if unsupported_library or assessment.answerability == "covered":
                break
            query_text = effective_query if attempt == 0 else reformulate_query(effective_query, requested, assessment.missing)
            key = query_text
            if key in completed and completed[key].max_results >= candidate_limit:
                continue
            started = time.perf_counter()
            try:
                payload = client.request_tavily_search(query=query_text, tavily_api_key=settings.tavily_api_key,
                    include_domains=domains, search_depth="basic", timeout_seconds=settings.docs_search_timeout_seconds,
                    max_results=candidate_limit, include_raw_content=include_raw)
            except Exception as exc:
                timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() or "timeout" in str(exc).lower()
                error_code = "RETRIEVAL_DOCS_TIMEOUT" if timed_out else "RETRIEVAL_DOCS_FAILED"
                error_message = f"invoke failed ({exc})"
                provider_requests.append(DocsSearchProviderRequest(
                    query=query_text, max_results=candidate_limit, status="error"))
                break
            finally:
                provider_ms += elapsed_ms(started, time.perf_counter())
            # Only completed provider requests suppress future searches. A timeout
            # can retain earlier partial evidence without consuming this query.
            completed[key] = DocsSearchRequest(query=query_text, max_results=candidate_limit)
            provider_requests.append(DocsSearchProviderRequest(
                query=query_text, max_results=candidate_limit, status="completed",
                result_count=len(payload["results"])))
            started = time.perf_counter()
            batch, _scores = collect_docs_search_hits(payload["results"], allowed_domains=domains,
                retrieval_warnings=warnings, query=effective_query, filter_counters=counters)
            candidates = dedupe_docs_hits([*candidates, *batch])
            assessment = select_docs_evidence(effective_query, candidates, requested, k=k,
                                              allowed_domains=domains, retrieval_warnings=warnings)
            post_ms += elapsed_ms(started, time.perf_counter())
        hits, answerability, missing = assessment.hits, assessment.answerability, assessment.missing
        if unsupported_library:
            missing = [f"library:{requested.library}"]
        if any(item.startswith("symbol:") for item in missing):
            warnings.append("identifier_coverage_incomplete")
            counters.filtered_identifier_mismatch_count = assessment.rejected_symbols
        status = "success" if hits else "error" if error_code else "no_result"
        payload = build_retrieval_payload(tool="tavily_search", route="docs", query=effective_query,
            hits=[hit.model_copy(update={"rank": i}) for i, hit in enumerate(hits, 1)], status=status,
            message=error_message if error_code else "" if hits else "no official documentation evidence found",
            raw_score=max((h.score.raw for h in hits if h.score.raw is not None), default=None),
            provider_ms=provider_ms, url_validation_ms=counters.url_validation_ms,
            post_filter_ms=max(0, post_ms - counters.url_validation_ms), include_raw_content_requested=bool(include_raw),
            provider_result_count=counters.provider_result_count, filtered_invalid_url_count=counters.filtered_invalid_url_count,
            filtered_path_prefix_count=counters.filtered_path_prefix_count, filtered_cross_domain_count=counters.filtered_cross_domain_count,
            filtered_http_error_count=counters.filtered_http_error_count, filtered_redirect_policy_count=counters.filtered_redirect_policy_count,
            filtered_url_request_failed_count=counters.filtered_url_request_failed_count,
            filtered_identifier_mismatch_count=counters.filtered_identifier_mismatch_count, validated_url_count=counters.validated_url_count,
            final_evidence_count=len(hits), warnings=sorted(set(warnings)), error_code=error_code)
        payload["diagnostics"].update(answerability="unknown" if error_code and not hits else answerability,
            missing_requirements=missing, k=k, candidate_count=len(candidates),
            completed_requests=[request.model_dump(mode="json") for request in completed.values()],
            provider_requests=[request.model_dump(mode="json") for request in provider_requests])
        return payload

    return tavily_search
