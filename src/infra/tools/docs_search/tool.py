from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError, wait
import time
from typing import Any, Literal

from langchain_core.tools import StructuredTool

from src.core.latency import elapsed_ms
from src.core.evidence import evidence_to_dicts
from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.extraction import should_extract_doc_content
from src.infra.tools.docs_search.normalization import canonicalize_docs_query_text
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_query_hint, normalize_include_domains
from src.infra.tools.docs_search.ranking import filter_docs_evidence_by_topic_purity, has_exact_identifier_coverage, has_meaningful_docs_evidence, merge_docs_evidence_items
from src.infra.tools.docs_search.schemas import TavilyArgs
from src.infra.tools.docs_search.serialization import DocsSearchFilterCounters, collect_docs_search_evidence


def build_docs_search_tool(settings: AppSettings) -> Any:
    default_domains = list(docs_search_rules().allowed_doc_path_prefixes.keys())

    def _record_tail_hedge(
        raw_results: dict[str, Any],
        *,
        hedge_state: dict[str, Any],
    ) -> None:
        tail_hedge = raw_results.get("_tail_hedge")
        if not isinstance(tail_hedge, dict):
            return
        hedge_state["hedge_started"] = bool(hedge_state["hedge_started"] or tail_hedge.get("hedge_started"))
        hedge_state["hedge_dropped"] = bool(hedge_state["hedge_dropped"] or tail_hedge.get("hedge_dropped"))
        hedge_state["hedge_attempts_started"] = int(hedge_state["hedge_attempts_started"]) + int(
            tail_hedge.get("hedge_attempts_started") or 0
        )
        hedge_state["hedge_attempts_dropped"] = int(hedge_state["hedge_attempts_dropped"]) + int(
            tail_hedge.get("hedge_attempts_dropped") or 0
        )
        winner = str(tail_hedge.get("hedge_winner") or "").strip()
        if winner and not hedge_state.get("hedge_winner"):
            hedge_state["hedge_winner"] = winner

    def _request_docs_search(query_text: str, domains: list[str], search_depth: str, include_raw_content: Any) -> dict[str, Any]:
        return client.request_tavily_search(
            query=query_text,
            tavily_api_key=settings.tavily_api_key,
            include_domains=domains,
            search_depth=search_depth,  # type: ignore[arg-type]
            timeout_seconds=settings.docs_search_timeout_seconds,
            hedge_delay_seconds=settings.docs_search_hedge_delay_seconds,
            hedge_max_attempts=settings.tail_hedge_max_attempts,
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
        hedge_state: dict[str, Any] = {
            "hedge_started": False,
            "hedge_dropped": False,
            "hedge_winner": None,
            "hedge_attempts_started": 0,
            "hedge_attempts_dropped": 0,
        }
        fallback_jobs: list[tuple[str, Any, Future[dict[str, Any]]]] = []
        fallback_plan: list[tuple[str, Any]] = []
        consumed_provider_queries: set[str] = set()
        provider_future_by_query: dict[str, Future[dict[str, Any]]] = {}
        provider_query_by_future: dict[Future[dict[str, Any]], tuple[str, Any]] = {}
        docs_search_executor: ThreadPoolExecutor | None = None

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

        def submit_provider_query(query_text: str, fallback_raw_content: Any) -> Future[dict[str, Any]]:
            nonlocal include_raw_content_requested
            if docs_search_executor is None:
                raise RuntimeError("docs search executor is not initialized")
            existing_future = provider_future_by_query.get(query_text)
            if existing_future is not None:
                return existing_future
            include_raw_content_requested = include_raw_content_requested or bool(fallback_raw_content)
            future = docs_search_executor.submit(
                _request_docs_search,
                query_text,
                domains,
                search_depth,
                fallback_raw_content,
            )
            fallback_jobs.append((query_text, fallback_raw_content, future))
            provider_future_by_query[query_text] = future
            provider_query_by_future[future] = (query_text, fallback_raw_content)
            return future

        def wait_for_first_provider_payload(
            futures: list[Future[dict[str, Any]]],
        ) -> tuple[str, dict[str, Any]]:
            pending = set(futures)
            first_error: Exception | None = None
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    query_text, _raw_content = provider_query_by_future[future]
                    try:
                        payload = future.result()
                    except Exception as exc:
                        if first_error is None:
                            first_error = exc
                        continue
                    if isinstance(payload, dict) and isinstance(payload.get("results"), list):
                        consumed_provider_queries.add(query_text)
                        return query_text, payload
                    if first_error is None:
                        first_error = RuntimeError("missing or invalid Tavily results payload")
            if first_error is not None:
                raise first_error
            raise RuntimeError("missing or invalid Tavily results payload")

        def cancel_pending_fallbacks() -> None:
            for _fallback_query, _fallback_raw_content, fallback_future in fallback_jobs:
                if not fallback_future.done():
                    fallback_future.cancel()
            if docs_search_executor is not None:
                docs_search_executor.shutdown(wait=False, cancel_futures=True)

        try:
            provider_started = time.perf_counter()
            worker_count = max(1, min(2, 1 + len(fallback_plan)))
            docs_search_executor = ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="documate-docs-search",
            )
            primary_future = submit_provider_query(effective_query, include_raw_content)
            fallback_delay = max(0.0, float(settings.docs_search_hedge_delay_seconds or 0.0))
            if fallback_delay > 0 and fallback_plan:
                try:
                    raw_results = primary_future.result(timeout=fallback_delay)
                    if not isinstance(raw_results, dict) or not isinstance(raw_results.get("results"), list):
                        raise RuntimeError("missing or invalid Tavily results payload")
                    initial_query = effective_query
                    consumed_provider_queries.add(effective_query)
                except FutureTimeoutError:
                    race_futures = [primary_future]
                    for fallback_query, fallback_include_raw_content in fallback_plan[:1]:
                        race_futures.append(submit_provider_query(fallback_query, fallback_include_raw_content))
                    initial_query, raw_results = wait_for_first_provider_payload(race_futures)
            else:
                raw_results = primary_future.result()
                if not isinstance(raw_results, dict) or not isinstance(raw_results.get("results"), list):
                    raise RuntimeError("missing or invalid Tavily results payload")
                initial_query = effective_query
                consumed_provider_queries.add(effective_query)
            _record_tail_hedge(raw_results, hedge_state=hedge_state)
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
        except Exception as exc:
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
            cancel_pending_fallbacks()
            is_timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message=f"invoke failed ({exc})",
                provider_ms=provider_ms,
                include_raw_content_requested=include_raw_content_requested,
                hedge_started=bool(hedge_state["hedge_started"]),
                hedge_dropped=bool(hedge_state["hedge_dropped"]),
                hedge_winner=hedge_state.get("hedge_winner"),
                hedge_attempts_started=int(hedge_state["hedge_attempts_started"]),
                hedge_attempts_dropped=int(hedge_state["hedge_attempts_dropped"]),
                error_code="RETRIEVAL_DOCS_TIMEOUT" if is_timeout else "RETRIEVAL_DOCS_FAILED",
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
            query=initial_query,
            filter_counters=filter_counters,
        )
        post_processing_ms += elapsed_ms(post_started, time.perf_counter())
        evidence_items.extend(batch_evidence)
        raw_scores.extend(batch_raw_scores)

        fallback_future_by_query = {
            fallback_query: fallback_future
            for fallback_query, _fallback_include_raw_content, fallback_future in fallback_jobs
        }
        followup_plan = [(effective_query, include_raw_content), *fallback_plan]
        for fallback_query, fallback_include_raw_content in followup_plan:
            if fallback_query in consumed_provider_queries:
                continue
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
            fallback_future = fallback_future_by_query.get(fallback_query)
            if fallback_future is None:
                fallback_future = submit_provider_query(fallback_query, fallback_include_raw_content)
                fallback_future_by_query[fallback_query] = fallback_future
            try:
                fallback_results = fallback_future.result()
                _record_tail_hedge(fallback_results, hedge_state=hedge_state)
                provider_ms = max(provider_ms, elapsed_ms(provider_started, time.perf_counter()))
            except Exception:
                provider_ms = max(provider_ms, elapsed_ms(provider_started, time.perf_counter()))
                continue
            fallback_items = fallback_results.get("results") if isinstance(fallback_results, dict) else None
            if not isinstance(fallback_items, list):
                continue
            consumed_provider_queries.add(fallback_query)
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
        cancel_pending_fallbacks()
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
            hedge_started=bool(hedge_state["hedge_started"]),
            hedge_dropped=bool(hedge_state["hedge_dropped"]),
            hedge_winner=hedge_state.get("hedge_winner"),
            hedge_attempts_started=int(hedge_state["hedge_attempts_started"]),
            hedge_attempts_dropped=int(hedge_state["hedge_attempts_dropped"]),
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
