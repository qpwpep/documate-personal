from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from src.core.evidence import SearchHit
from src.core.latency import elapsed_ms
from src.core.planner_schema import RetrievalRequirement
from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.extraction import should_extract_doc_content
from src.infra.tools.docs_search.policy import docs_search_rules, normalize_include_domains
from src.infra.tools.docs_search.ranking import dedupe_docs_hits, filter_docs_hits_by_topic_purity
from src.infra.tools.docs_search.requirements import assess_candidates, canonical_query, library_domains, reformulate_query, resolve_requirement
from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_hits


def build_docs_search_tool(settings: AppSettings) -> Callable[..., dict[str, Any]]:
    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
        *,
        requirement: RetrievalRequirement | None = None,
        k: int = 3,
        attempted_queries: list[str] | None = None,
        previous_hits: list[SearchHit] | None = None,
    ) -> dict[str, Any]:
        requested = resolve_requirement(query, requirement)
        effective_query = canonical_query(query, requested)
        if requirement is None and include_domains is None and requested.library and requested.library.casefold() not in effective_query.casefold():
            effective_query += " " + requested.library
        domains = normalize_include_domains(library_domains(str(requested.library or "")) or include_domains
                                             or list(docs_search_rules().allowed_doc_path_prefixes))
        count = max(1, min(10, int(k)))
        include_raw = "markdown" if requested.symbols or requested.aspects or should_extract_doc_content(effective_query) else False
        attempted = list(dict.fromkeys(attempted_queries or []))
        attempted_keys = {" ".join(q.casefold().split()) for q in attempted}
        plan = list(dict.fromkeys([effective_query, reformulate_query(effective_query, requested)]))
        unsupported_library = bool(requested.library and not library_domains(requested.library))
        if unsupported_library:
            plan = []
        if include_domains is not None and requirement is None:
            plan = [effective_query]
        candidates = dedupe_docs_hits([hit for hit in previous_hits or [] if hit.evidence.route == "docs"])
        if unsupported_library:
            candidates = []
        warnings: list[str] = []
        counters = DocsSearchFilterCounters()
        provider_ms = post_ms = 0
        ranked = candidates if requested.symbols else filter_docs_hits_by_topic_purity(effective_query, candidates, warnings, limit=count)
        hits, answerability, missing = assess_candidates(ranked, requested, k=count)
        if answerability == "covered":
            plan = []
        error_code = None
        error_message = ""
        for query_text in plan:
            key = " ".join(query_text.casefold().split())
            if key in attempted_keys:
                continue
            attempted_keys.add(key)
            attempted.append(query_text)
            started = time.perf_counter()
            try:
                payload = client.request_tavily_search(query=query_text, tavily_api_key=settings.tavily_api_key,
                    include_domains=domains, search_depth=search_depth, timeout_seconds=settings.docs_search_timeout_seconds,
                    max_results=count, include_raw_content=include_raw)
                if not isinstance(payload, dict) or not isinstance(payload.get("results"), list):
                    raise RuntimeError("missing or invalid Tavily results payload")
            except Exception as exc:
                timed_out = isinstance(exc, TimeoutError) or "timed out" in str(exc).lower() or "timeout" in str(exc).lower()
                error_code = "RETRIEVAL_DOCS_TIMEOUT" if timed_out else "RETRIEVAL_DOCS_FAILED"
                error_message = f"invoke failed ({exc})"
                break
            finally:
                provider_ms += elapsed_ms(started, time.perf_counter())
            started = time.perf_counter()
            batch, _scores = collect_docs_search_hits(payload["results"], allowed_domains=domains,
                retrieval_warnings=warnings, query=effective_query, filter_counters=counters)
            candidates = dedupe_docs_hits([*candidates, *batch])
            ranked = candidates if requested.symbols else filter_docs_hits_by_topic_purity(effective_query, candidates, warnings, limit=count)
            hits, answerability, missing = assess_candidates(ranked, requested, k=count)
            post_ms += elapsed_ms(started, time.perf_counter())
            if answerability == "covered":
                break
        if unsupported_library:
            missing = [f"library:{requested.library}"]
        if any(item.startswith("symbol:") for item in missing):
            warnings.append("identifier_coverage_incomplete")
            counters.filtered_identifier_mismatch_count = len(candidates)
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
            missing_requirements=missing, attempted_queries=attempted, candidate_count=len(candidates))
        return payload

    return tavily_search
