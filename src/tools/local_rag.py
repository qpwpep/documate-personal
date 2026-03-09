from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import nbformat
from langchain_chroma import Chroma
from langchain_core.tools import StructuredTool
from langchain_openai import OpenAIEmbeddings
from langgraph.prebuilt import InjectedState

from ..chunking import chunk_notebook, chunk_python_text
from ..settings import AppSettings
from ._common import (
    RagArgs,
    UploadArgs,
    build_evidence_item,
    build_retrieval_payload,
    dedupe_evidence_dicts,
    to_float_or_none,
)


INDEX_PATH = Path("data/index")


def load_chroma(openai_api_key: str, index_path: Path = INDEX_PATH) -> Chroma:
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=openai_api_key,
    )
    return Chroma(
        embedding_function=embeddings,
        persist_directory=str(index_path),
        collection_name="notebooks",
    )


def extract_text_from_py(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file_obj:
        return file_obj.read()

def extract_upload_session_id(path: str) -> str:
    parts = Path(path).expanduser().parts
    upload_index = -1
    for index, part in enumerate(parts):
        if part.lower() == "uploads":
            upload_index = index

    if upload_index < 0 or upload_index + 1 >= len(parts):
        raise ValueError("Upload path must include uploads/<session_id>/...")

    session_id = str(parts[upload_index + 1]).strip()
    if not session_id or session_id in {".", ".."}:
        raise ValueError("Upload path must include a valid session_id segment")
    return session_id


def build_upload_collection_name(session_id: str) -> str:
    normalized_session_id = re.sub(r"[^0-9A-Za-z_-]+", "-", session_id.strip()).strip("-")
    if not normalized_session_id:
        raise ValueError("session_id cannot be normalized into a collection name")
    return f"upload-session-{normalized_session_id}"


@dataclass
class UploadedRetrieverHandle:
    retriever: Any
    collection_name: str
    _vectorstore: Chroma = field(repr=False)
    _cleaned_up: bool = field(default=False, init=False, repr=False)

    def cleanup(self) -> None:
        if self._cleaned_up:
            return
        self._vectorstore.delete_collection()
        self._cleaned_up = True


def build_temp_retriever(path: str, api_key: str | None = None, k: int = 4) -> UploadedRetrieverHandle:
    session_id = extract_upload_session_id(path)
    collection_name = build_upload_collection_name(session_id)

    path_lower = str(path).lower()
    if path_lower.endswith(".py"):
        docs = chunk_python_text(
            path=path,
            text=extract_text_from_py(path),
            chunk_size=800,
            chunk_overlap=120,
        )
    elif path_lower.endswith(".ipynb"):
        notebook = nbformat.read(path, as_version=4)
        docs = chunk_notebook(
            path=path,
            notebook=notebook,
            chunk_size=800,
            chunk_overlap=120,
        )
    else:
        raise ValueError("Unsupported file type (only .py or .ipynb).")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    vectorstore = Chroma.from_documents(
        docs,
        embedding=embeddings,
        collection_name=collection_name,
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    return UploadedRetrieverHandle(
        retriever=retriever,
        collection_name=collection_name,
        _vectorstore=vectorstore,
    )


def build_local_rag_tools(settings: AppSettings) -> tuple[Any, Any]:
    def rag_search(query: str, k: int = 4) -> dict[str, Any]:
        if not INDEX_PATH.is_dir():
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

        db = load_chroma(settings.openai_api_key)
        docs_with_scores: list[tuple[Any, float | None]] = []
        try:
            raw_docs_with_scores = db.similarity_search_with_relevance_scores(query, k=k)
            for doc, score in raw_docs_with_scores:
                docs_with_scores.append((doc, to_float_or_none(score)))
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

        evidence_items = []
        for doc, score in docs_with_scores:
            if not hasattr(doc, "metadata"):
                continue
            source = doc.metadata.get("source", "notebook")
            evidence_item = build_evidence_item(
                kind="local",
                tool="rag_search",
                url_or_path=str(source),
                snippet=(doc.page_content or "").replace("\n", " "),
                score=score,
                metadata=getattr(doc, "metadata", None),
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)

        evidence = dedupe_evidence_dicts(evidence_items)
        return build_retrieval_payload(
            tool="rag_search",
            route="local",
            query=query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no local notebook evidence found",
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
            if vectorstore is not None and hasattr(vectorstore, "similarity_search_with_relevance_scores"):
                raw_docs_with_scores = vectorstore.similarity_search_with_relevance_scores(query, k=k)
                for doc, score in raw_docs_with_scores:
                    docs_with_scores.append((doc, to_float_or_none(score)))
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

        evidence_items = []
        for doc, score in docs_with_scores:
            if not hasattr(doc, "metadata"):
                continue
            source = doc.metadata.get("source", "uploaded")
            evidence_item = build_evidence_item(
                kind="local",
                tool="upload_search",
                url_or_path=str(source),
                snippet=(doc.page_content or "").replace("\n", " "),
                score=score,
                metadata=getattr(doc, "metadata", None),
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)
        evidence = dedupe_evidence_dicts(evidence_items)
        return build_retrieval_payload(
            tool="upload_search",
            route="upload",
            query=query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no uploaded file evidence found",
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
