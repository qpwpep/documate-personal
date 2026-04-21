from __future__ import annotations

from typing import Annotated, Any

from langchain_core.tools import StructuredTool
from langgraph.prebuilt import InjectedState

from src.infra.chroma_store import CHROMA_DISTANCE_METRIC, CHROMA_SCORE_DIRECTION
from src.infra.settings import AppSettings
from src.infra.tools._common import RagArgs, UploadArgs, build_retrieval_payload
from src.infra.tools.local_rag import client
from src.infra.tools.local_rag.ranking import rank_retrieval_rows
from src.infra.tools.local_rag.serialization import build_local_evidence_bundle


def _build_search_payload(
    *,
    query: str,
    docs_with_scores: list[tuple[Any, float | None]],
    tool_name: str,
    route: str,
    no_result_message: str,
    default_source: str,
) -> dict[str, Any]:
    ranked_rows = rank_retrieval_rows(docs_with_scores, query=query)
    evidence, normalized_scores, raw_scores, retrieval_warnings = build_local_evidence_bundle(
        ranked_rows,
        query=query,
        tool_name=tool_name,
        default_source=default_source,
    )
    return build_retrieval_payload(
        tool=tool_name,
        route=route,
        query=query,
        evidence=evidence,
        status="success" if evidence else "no_result",
        message="" if evidence else no_result_message,
        normalized_score=max(normalized_scores) if normalized_scores else None,
        raw_score=min(raw_scores) if raw_scores else None,
        metric=CHROMA_DISTANCE_METRIC,
        score_direction=CHROMA_SCORE_DIRECTION,
        warnings=retrieval_warnings,
    )


def build_local_rag_tools(settings: AppSettings) -> tuple[Any, Any]:
    def rag_search(query: str, k: int = 4) -> dict[str, Any]:
        if not client.INDEX_PATH.is_dir():
            return build_retrieval_payload(
                tool="rag_search",
                route="local",
                query=query,
                status="unavailable",
                message="local notebook index is unavailable",
            )
        if not settings.openai_api_key:
            return build_retrieval_payload(
                tool="rag_search",
                route="local",
                query=query,
                status="unavailable",
                message="OPENAI_API_KEY is not configured for local retrieval",
            )

        db = client.load_chroma(settings.openai_api_key)
        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            docs_with_scores = client.search_with_raw_scores(db, query=query, k=k)
        except Exception:
            try:
                docs = db.similarity_search(query, k=k)
                docs_with_scores = [(doc, None) for doc in docs]
            except Exception as exc:
                return build_retrieval_payload(
                    tool="rag_search",
                    route="local",
                    query=query,
                    status="error",
                    message=f"local similarity search failed ({exc})",
                )

        return _build_search_payload(
            query=query,
            docs_with_scores=docs_with_scores,
            tool_name="rag_search",
            route="local",
            no_result_message="no local notebook evidence found",
            default_source="notebook",
        )

    def upload_search(
        query: str,
        k: int = 4,
        retriever: Annotated[Any, InjectedState("retriever")] = None,
    ) -> dict[str, Any]:
        if retriever is None:
            return build_retrieval_payload(
                tool="upload_search",
                route="upload",
                query=query,
                status="unavailable",
                message="upload retriever is unavailable; upload a .py or .ipynb file first",
            )

        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            vectorstore = getattr(retriever, "vectorstore", None)
            if vectorstore is not None:
                docs_with_scores = client.search_with_raw_scores(vectorstore, query=query, k=k)
            else:
                docs = retriever.invoke(query)
                docs_with_scores = [(doc, None) for doc in docs]
        except Exception as exc:
            return build_retrieval_payload(
                tool="upload_search",
                route="upload",
                query=query,
                status="error",
                message=f"uploaded file retrieval failed ({exc})",
            )

        return _build_search_payload(
            query=query,
            docs_with_scores=docs_with_scores,
            tool_name="upload_search",
            route="upload",
            no_result_message="no uploaded file evidence found",
            default_source="uploaded",
        )

    rag_search_tool = StructuredTool.from_function(
        name="rag_search",
        description=(
            "Search local .ipynb notebooks (vector index) and return structured evidence items. "
            "Use this when the question is covered by our local documents."
        ),
        func=rag_search,
        args_schema=RagArgs,
    )
    upload_search_tool = StructuredTool.from_function(
        name="upload_search",
        description=(
            "Search only the currently uploaded file context and return structured evidence items. "
            "Use this when user asks about uploaded file content."
        ),
        func=upload_search,
        args_schema=UploadArgs,
    )
    return rag_search_tool, upload_search_tool
