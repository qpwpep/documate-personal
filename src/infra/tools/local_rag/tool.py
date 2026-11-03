from __future__ import annotations

import time
from collections.abc import Callable

from src.core.contracts.retrieval import RetrievalResult, UploadRetriever, UploadSearchMode
from src.core.documents import ParsedDocument
from src.core.evidence import SearchHit
from src.core.latency import elapsed_ms
from src.core.llm_errors import LLMCallError
from src.core.literal_matching import missing_literal_aspects
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools._common import build_retrieval_result
from src.infra.llm_boundary import check_request_deadline
from src.infra.tools.local_rag.requirements import resolve_source_requirement


def _build_search_payload(
    *,
    query: str,
    hits: list[SearchHit],
    mode: UploadSearchMode,
    provider_ms: int = 0,
    requirement: RetrievalRequirement,
) -> RetrievalResult:
    post_started = time.perf_counter()
    normalized_scores = [hit.score.normalized for hit in hits if hit.score.normalized is not None]
    raw_scores = [hit.score.raw for hit in hits if hit.score.raw is not None]
    metric = {"baseline": "l2", "bm25": "bm25", "hybrid": "rrf"}[mode]
    lower_is_better = mode == "baseline"
    best_raw_score = None
    if raw_scores:
        best_raw_score = min(raw_scores) if lower_is_better else max(raw_scores)
    post_filter_ms = elapsed_ms(post_started, time.perf_counter())
    payload = build_retrieval_result(
        tool="upload_search",
        route="upload",
        query=query,
        hits=hits,
        status="success" if hits else "no_result",
        message="" if hits else "no uploaded file evidence found",
        normalized_score=max(normalized_scores) if normalized_scores else None,
        raw_score=best_raw_score,
        provider_ms=provider_ms,
        post_filter_ms=post_filter_ms,
        metric=metric,
        score_direction="lower_is_better" if lower_is_better else "higher_is_better",
    )
    payload.diagnostics.candidate_count = len(hits)
    if (requirement.match == "topic" and (requirement.file_ids or requirement.symbols or requirement.aspects)
            and not (requirement.version or requirement.library or requirement.evidence_needs)):
        # Coverage means literal anchors in the returned ranges, not an exact
        # implementation or inferred semantic support from the complete source.
        missing = []
        for file_id in requirement.file_ids or [None]:
            scoped = [hit.evidence for hit in hits if file_id is None
                      or hit.evidence.element.metadata.get("file_id") == file_id]
            if file_id is not None and not scoped:
                missing.append(f"file:{file_id}")
                continue
            prefix = f"file:{file_id}:" if file_id is not None else ""
            for kind, anchors in (("symbol", requirement.symbols), ("aspect", requirement.aspects)):
                missing.extend(f"{prefix}{kind}:{anchor}" for anchor in missing_literal_aspects(anchors, scoped))
        payload.diagnostics.answerability = (("covered" if not missing else "partial") if hits else "missing")
        payload.diagnostics.missing_requirements = missing
    return payload


def _build_requirement_payload(
    *, query: str, requirement: RetrievalRequirement, source_documents: tuple[ParsedDocument, ...],
) -> RetrievalResult:
    started = time.perf_counter()
    result = resolve_source_requirement(requirement=requirement, source_documents=source_documents)
    payload = build_retrieval_result(
        tool="upload_search", route="upload", query=query, hits=result.hits,
        status="success" if result.hits else "no_result",
        message="" if result.answerability == "covered" else f"uploaded source requirement {result.answerability}",
        post_filter_ms=elapsed_ms(started, time.perf_counter()),
        metric="source_symbol", score_direction="higher_is_better", warnings=result.warnings,
    )
    payload.diagnostics.answerability = result.answerability
    payload.diagnostics.missing_requirements = result.missing_requirements
    payload.diagnostics.candidate_count = result.candidate_count
    return payload


def build_upload_search_tool(*, default_mode: UploadSearchMode = "hybrid") -> Callable[..., RetrievalResult]:
    def upload_search(
        query: str,
        k: int = 4,
        retriever: UploadRetriever | None = None,
        *,
        requirement: RetrievalRequirement,
        mode: UploadSearchMode = default_mode,
    ) -> RetrievalResult:
        if retriever is None:
            return build_retrieval_result(
                tool="upload_search",
                route="upload",
                query=query,
                status="unavailable",
                message="upload retriever is unavailable; attach a supported file first",
            )

        file_ids = requirement.file_ids
        if requirement.match in {"symbol", "definition"}:
            return _build_requirement_payload(query=query, requirement=requirement,
                                              source_documents=retriever.source_documents)

        hits: list[SearchHit] = []
        provider_ms = 0
        provider_started = time.perf_counter()
        try:
            check_request_deadline("retrieval")
            hits = retriever.search(query, k=k, file_ids=file_ids, mode=mode)
            check_request_deadline("retrieval")
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
        except LLMCallError:
            raise
        except Exception as exc:
            check_request_deadline("retrieval")
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
            return build_retrieval_result(
                tool="upload_search",
                route="upload",
                query=query,
                status="error",
                message=f"uploaded file retrieval failed ({exc})",
                provider_ms=provider_ms,
                error_code="LOCAL_RAG_FAILED",
            )

        return _build_search_payload(
            query=query,
            hits=hits,
            mode=mode,
            provider_ms=provider_ms,
            requirement=requirement,
        )

    return upload_search
