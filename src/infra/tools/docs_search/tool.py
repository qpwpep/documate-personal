from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import StructuredTool

from src.infra.settings import AppSettings
from src.infra.tools._common import build_retrieval_payload, dedupe_evidence_dicts
from src.infra.tools.docs_search import client
from src.infra.tools.docs_search.policy import docs_search_rules, infer_docs_query_hint, normalize_include_domains
from src.infra.tools.docs_search.ranking import filter_docs_evidence_by_topic_purity, has_meaningful_docs_evidence, merge_docs_evidence_items
from src.infra.tools.docs_search.schemas import TavilyArgs
from src.infra.tools.docs_search.serialization import collect_docs_search_evidence


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
        if include_domains is None:
            query_hint = infer_docs_query_hint(effective_query)
            if query_hint is not None:
                library_name, hinted_domains, fallback_queries = query_hint
                domains = normalize_include_domains(hinted_domains)
                if library_name.lower() not in effective_query.lower():
                    effective_query = f"{effective_query} {library_name}".strip()
        try:
            raw_results = client.request_tavily_search(
                query=effective_query,
                tavily_api_key=settings.tavily_api_key,
                include_domains=domains,
                search_depth=search_depth,
                timeout_seconds=settings.docs_search_timeout_seconds,
            )
        except Exception as exc:
            is_timeout = isinstance(exc, TimeoutError) or "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message=f"invoke failed ({exc})",
                error_code="RETRIEVAL_DOCS_TIMEOUT" if is_timeout else "RETRIEVAL_DOCS_FAILED",
            )

        if not isinstance(raw_results, dict):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="unexpected response type from Tavily",
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
                error_code="RETRIEVAL_DOCS_FAILED",
            )

        evidence_items = []
        retrieval_warnings: list[str] = []
        raw_scores: list[float] = []
        batch_evidence, batch_raw_scores = collect_docs_search_evidence(
            results,
            allowed_domains=hinted_domains,
            retrieval_warnings=retrieval_warnings,
        )
        evidence_items.extend(batch_evidence)
        raw_scores.extend(batch_raw_scores)

        for fallback_query in fallback_queries:
            deduped_batch = dedupe_evidence_dicts(evidence_items)
            filtered_batch = filter_docs_evidence_by_topic_purity(
                effective_query,
                deduped_batch,
                retrieval_warnings,
            )
            if has_meaningful_docs_evidence(filtered_batch):
                break
            try:
                fallback_results = client.request_tavily_search(
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
            batch_evidence, batch_raw_scores = collect_docs_search_evidence(
                fallback_items,
                allowed_domains=hinted_domains,
                retrieval_warnings=retrieval_warnings,
            )
            evidence_items.extend(batch_evidence)
            raw_scores.extend(batch_raw_scores)

        evidence = dedupe_evidence_dicts(merge_docs_evidence_items(evidence_items))
        evidence = filter_docs_evidence_by_topic_purity(effective_query, evidence, retrieval_warnings)
        if evidence and not has_meaningful_docs_evidence(evidence):
            retrieval_warnings.append("docs_chrome_only")
            evidence = []
        return build_retrieval_payload(
            tool="tavily_search",
            route="docs",
            query=effective_query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no official documentation evidence found",
            raw_score=max(raw_scores) if raw_scores else None,
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
