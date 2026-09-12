from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from src.core.documents import ParsedDocument
from src.core.latency import elapsed_ms
from src.core.planner_schema import RetrievalRequirement
from src.infra.chroma_store import CHROMA_DISTANCE_METRIC, CHROMA_SCORE_DIRECTION
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.local_rag import client
from src.infra.tools.local_rag.ranking import rank_retrieval_rows
from src.infra.tools.local_rag.requirements import infer_legacy_requirement, resolve_source_requirement
from src.infra.tools.local_rag.serialization import build_local_hit_bundle


def _build_search_payload(
    *,
    query: str,
    docs_with_scores: list[tuple[Any, float | None]],
    provider_ms: int = 0,
    requirement: RetrievalRequirement | None = None,
) -> dict[str, Any]:
    post_started = time.perf_counter()
    ranked_rows = rank_retrieval_rows(docs_with_scores, query=query)
    hits, normalized_scores, raw_scores, retrieval_warnings = build_local_hit_bundle(
        ranked_rows,
        query=query,
    )
    post_filter_ms = elapsed_ms(post_started, time.perf_counter())
    payload = build_retrieval_payload(
        tool="upload_search",
        route="upload",
        query=query,
        hits=hits,
        status="success" if hits else "no_result",
        message="" if hits else "no uploaded file evidence found",
        normalized_score=max(normalized_scores) if normalized_scores else None,
        raw_score=min(raw_scores) if raw_scores else None,
        provider_ms=provider_ms,
        post_filter_ms=post_filter_ms,
        metric=CHROMA_DISTANCE_METRIC,
        score_direction=CHROMA_SCORE_DIRECTION,
        warnings=retrieval_warnings,
    )
    payload["diagnostics"].update(answerability="unknown", missing_requirements=[], candidate_count=len(docs_with_scores))
    if (requirement is not None and requirement.file_ids and requirement.match == "topic"
            and not (requirement.symbols or requirement.aspects or requirement.version or requirement.library)):
        # For a topic constrained only by files, coverage means candidates from
        # every requested source. Additional code/version constraints stay unknown.
        covered_files = {hit.evidence.element.metadata.get("file_id") for hit in hits}
        missing_files = [file_id for file_id in requirement.file_ids if file_id not in covered_files]
        payload["diagnostics"].update(
            answerability="covered" if not missing_files else ("partial" if hits else "missing"),
            missing_requirements=[f"file:{file_id}" for file_id in missing_files])
    return payload


def _build_requirement_payload(
    *, query: str, requirement: RetrievalRequirement, source_documents: tuple[ParsedDocument, ...] | None,
    docs_with_scores: list[tuple[Any, float | None]], provider_ms: int = 0,
) -> dict[str, Any]:
    started = time.perf_counter()
    result = resolve_source_requirement(requirement=requirement, source_documents=source_documents,
                                        candidate_rows=docs_with_scores)
    payload = build_retrieval_payload(
        tool="upload_search", route="upload", query=query, hits=result.hits,
        status="success" if result.hits else "no_result",
        message="" if result.answerability == "covered" else f"uploaded source requirement {result.answerability}",
        provider_ms=provider_ms, post_filter_ms=elapsed_ms(started, time.perf_counter()),
        metric="source_symbol", score_direction="higher_is_better", warnings=result.warnings,
    )
    payload["diagnostics"].update(answerability=result.answerability,
                                  missing_requirements=result.missing_requirements,
                                  candidate_count=result.candidate_count)
    return payload


def build_upload_search_tool() -> Callable[..., dict[str, Any]]:
    def upload_search(
        query: str,
        k: int = 4,
        retriever: Any = None,
        *,
        requirement: RetrievalRequirement | None = None,
    ) -> dict[str, Any]:
        if retriever is None:
            return build_retrieval_payload(
                tool="upload_search",
                route="upload",
                query=query,
                status="unavailable",
                message="upload retriever is unavailable; upload a .py or .ipynb file first",
            )

        if requirement is None or not requirement.specified:
            requirement = infer_legacy_requirement(query)
        file_ids = requirement.file_ids if requirement is not None else []
        upload_files = getattr(retriever, "upload_files", ())
        if set(file_ids).difference(file.file_id for file in upload_files):
            return build_retrieval_payload(
                tool="upload_search", route="upload", query=query, status="error",
                message="requested upload file is not active in this session",
                error_code="UPLOAD_FILE_SCOPE_INVALID")
        registry = getattr(retriever, "source_documents", None)
        source_documents = (tuple(document for document in registry if isinstance(document, ParsedDocument))
                            if isinstance(registry, (tuple, list)) else None)
        if source_documents is None:
            source_document = getattr(retriever, "source_document", None)
            if isinstance(source_document, ParsedDocument):
                source_documents = (source_document,)
        if source_documents is not None and file_ids:
            sources = {file.source_uri for file in upload_files if file.file_id in file_ids}
            source_documents = tuple(document for document in source_documents
                                     if document.snapshot.source_uri in sources)
        if requirement is not None and requirement.symbols and source_documents is not None:
            return _build_requirement_payload(query=query, requirement=requirement,
                                              source_documents=source_documents, docs_with_scores=[])

        docs_with_scores: list[tuple[Any, float | None]] = []
        provider_ms = 0
        provider_started = time.perf_counter()
        try:
            vectorstore = getattr(retriever, "vectorstore", None)
            provider_started = time.perf_counter()
            if vectorstore is not None:
                if file_ids:
                    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k, file_ids=file_ids)
                else:
                    docs_with_scores = client.search_with_raw_scores(vectorstore, query=query, k=k)
            else:
                if file_ids:
                    raise ValueError("Upload retriever does not support scoped retrieval")
                docs = retriever.invoke(query)
                docs_with_scores = [(doc, None) for doc in docs]
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
        except Exception as exc:
            provider_ms += elapsed_ms(provider_started, time.perf_counter())
            return build_retrieval_payload(
                tool="upload_search",
                route="upload",
                query=query,
                status="error",
                message=f"uploaded file retrieval failed ({exc})",
                provider_ms=provider_ms,
                error_code="LOCAL_RAG_FAILED",
            )

        if requirement is not None and requirement.symbols:
            return _build_requirement_payload(query=query, requirement=requirement,
                                              source_documents=None, docs_with_scores=docs_with_scores,
                                              provider_ms=provider_ms)

        return _build_search_payload(
            query=query,
            docs_with_scores=docs_with_scores,
            provider_ms=provider_ms,
            requirement=requirement,
        )

    return upload_search
