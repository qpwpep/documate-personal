from __future__ import annotations

import time
from collections.abc import Callable

from src.core.evidence import SearchHit
from src.core.contracts.retrieval import RetrievalResult
from src.core.contracts.debug import DocsSearchRequest, DocsSearchProviderRequest
from src.core.latency import elapsed_ms
from src.core.planner_schema import RetrievalRequirement
from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_result
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.policy import docs_search_rules, normalize_include_domains
from src.infra.tools.docs_search.ranking import dedupe_docs_hits, select_docs_candidates
from src.infra.tools.docs_search.requirements import canonical_query, library_domains, resolve_requirement
from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_hits


def build_docs_search_tool(settings: AppSettings) -> Callable[..., RetrievalResult]:
    def tavily_search(
        query: str,
        *,
        requirement: RetrievalRequirement,
        k: int = 3,
        completed_requests: list[DocsSearchRequest] | None = None,
        previous_hits: list[SearchHit] | None = None,
        preserved_evidence_ids: list[str] | None = None,
        refresh: bool = False,
    ) -> RetrievalResult:
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
        previous_snapshot_ids = {hit.evidence.snapshot.snapshot_id for hit in candidates}
        # Approval is owned by semantic validation. Only its explicit IDs get
        # ranking priority; other prior candidates remain available for review.
        preserved_ids = set(preserved_evidence_ids or [])
        fresh_snapshot_ids: set[str] = set()
        if unsupported_library:
            candidates = []
        warnings: list[str] = []
        counters = DocsSearchFilterCounters()
        provider_ms = post_ms = 0
        hits = select_docs_candidates(effective_query, candidates, requested, k=k, allowed_domains=domains,
            preserved_evidence_ids=preserved_ids)
        error_code = None
        error_message = ""
        # Search expansion belongs to the retrieval planner after Decisions has
        # reviewed these candidates. This tool never infers missing semantics or
        # reformulates a query from regex coverage labels.
        already_completed = effective_query in completed and completed[effective_query].max_results >= candidate_limit
        if not unsupported_library and (refresh or not already_completed):
            query_text = effective_query
            key = query_text
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
                # A failed refresh must not inherit the successful completion
                # marker from before that attempt and suppress a later retry.
                completed.pop(key, None)
            finally:
                provider_ms += elapsed_ms(started, time.perf_counter())
            if error_code is None:
                # Only completed provider requests suppress future searches. A
                # timeout retains prior candidates without consuming this query.
                completed[key] = DocsSearchRequest(query=query_text, max_results=candidate_limit)
                provider_requests.append(DocsSearchProviderRequest(
                    query=query_text, max_results=candidate_limit, status="completed",
                    result_count=len(payload["results"])))
                started = time.perf_counter()
                batch = collect_docs_search_hits(payload["results"], allowed_domains=domains,
                    retrieval_warnings=warnings, query=effective_query, filter_counters=counters)
                if refresh:
                    fresh_snapshot_ids.update(hit.evidence.snapshot.snapshot_id for hit in batch
                                              if hit.evidence.snapshot.snapshot_id not in previous_snapshot_ids)
                candidates = dedupe_docs_hits([*candidates, *batch])
                hits = select_docs_candidates(effective_query, candidates, requested, k=k,
                    allowed_domains=domains, preferred_snapshot_ids=fresh_snapshot_ids,
                    preserved_evidence_ids=preserved_ids)
                post_ms += elapsed_ms(started, time.perf_counter())
        status = "success" if hits else "error" if error_code else "no_result"
        result = build_retrieval_result(tool="tavily_search", route="docs", query=effective_query,
            hits=[hit.model_copy(update={"rank": i}) for i, hit in enumerate(hits, 1)], status=status,
            message=error_message if error_code else "" if hits else "no official documentation evidence found",
            raw_score=max((h.score.raw for h in hits if h.score.raw is not None), default=None),
            provider_ms=provider_ms, post_filter_ms=post_ms, include_raw_content_requested=bool(include_raw),
            provider_result_count=counters.provider_result_count, filtered_invalid_url_count=counters.filtered_invalid_url_count,
            filtered_path_prefix_count=counters.filtered_path_prefix_count, filtered_cross_domain_count=counters.filtered_cross_domain_count,
            warnings=warnings, error_code=error_code)
        result.diagnostics = result.diagnostics.model_copy(update={
            "answerability": "unknown" if hits or error_code else "missing",
            "missing_requirements": [], "k": k, "candidate_count": len(hits),
            "completed_requests": list(completed.values()),
            "provider_requests": provider_requests,
        })
        return result

    return tavily_search
