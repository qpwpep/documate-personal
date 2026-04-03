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
    normalize_relevance_score,
    to_float_or_none,
)


INDEX_PATH = Path("data/index")
_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b")
_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[가-힣]{2,}")
_KEYWORD_STOPWORDS = {
    "official",
    "docs",
    "documentation",
    "reference",
    "latest",
    "upload",
    "uploaded",
    "current",
    "file",
    "notebook",
    "code",
    "example",
    "examples",
    "find",
    "show",
    "tell",
    "explain",
    "describe",
    "how",
    "what",
    "where",
    "using",
    "used",
    "usage",
    "based",
    "공식",
    "문서",
    "최신",
    "업로드",
    "업로드한",
    "파일",
    "노트북",
    "코드",
    "예제",
    "설명",
    "문법",
    "사용",
    "위치",
    "찾아줘",
    "알려줘",
    "기준",
    "실제",
    "부분",
}
_PARAMETER_HINT_PATTERN = re.compile(
    r"(?i)\b(parameter|parameters|param|params|option|options|arg|args)\b|파라미터|매개변수|옵션|인자"
)


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


def _extract_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    seen_lowered: set[str] = set()
    for token in _IDENTIFIER_PATTERN.findall(str(text or "")):
        lowered = token.lower()
        if lowered in _KEYWORD_STOPWORDS:
            continue
        if lowered not in seen_lowered:
            identifiers.append(token)
            seen_lowered.add(lowered)
    return identifiers


def _extract_keywords(text: str) -> set[str]:
    return {
        token.strip().lower()
        for token in _KEYWORD_PATTERN.findall(str(text or "").lower())
        if len(token.strip()) >= 2 and token.strip().lower() not in _KEYWORD_STOPWORDS
    }


def _parameter_query_score(query: str, content: str) -> int:
    if _PARAMETER_HINT_PATTERN.search(query) is None:
        return 0
    compact = str(content or "")
    return int("=" in compact) + int("(" in compact and ")" in compact)


def _lexical_query_score(query: str, content: str) -> int:
    lowered_content = str(content or "").lower()
    identifier_hits = sum(
        1 for token in _extract_identifiers(query) if token.lower() in lowered_content
    )
    keyword_hits = len(_extract_keywords(query).intersection(_extract_keywords(content)))
    return (identifier_hits * 4) + keyword_hits + _parameter_query_score(query, lowered_content)


def _rank_retrieval_rows(
    docs_with_scores: list[tuple[Any, float | None]],
    *,
    query: str,
) -> list[tuple[Any, float | None]]:
    def _sort_key(item: tuple[Any, float | None]) -> tuple[int, float]:
        doc, score = item
        content = getattr(doc, "page_content", "")
        lexical = _lexical_query_score(query, str(content or ""))
        numeric_score = float(score) if score is not None else float("-inf")
        return (lexical, numeric_score)

    return sorted(docs_with_scores, key=_sort_key, reverse=True)


def _build_query_focused_snippet(text: str, *, query: str, max_length: int = 500) -> str:
    normalized = str(text or "").strip()
    if len(normalized) <= max_length:
        return normalized

    query_identifiers = _extract_identifiers(query)
    lowered_text = normalized.lower()
    best_index: int | None = None
    for token in query_identifiers:
        index = lowered_text.find(token.lower())
        if index < 0:
            continue
        best_index = index if best_index is None else min(best_index, index)

    if best_index is None:
        keywords = sorted(_extract_keywords(query), key=len, reverse=True)
        for keyword in keywords:
            index = lowered_text.find(keyword.lower())
            if index < 0:
                continue
            best_index = index if best_index is None else min(best_index, index)

    if best_index is None:
        return normalized[:max_length]

    line_start = normalized.rfind("\n", 0, best_index)
    if line_start < 0:
        line_start = max(0, best_index - (max_length // 3))
    else:
        line_start += 1
    snippet = normalized[line_start : line_start + max_length]
    return snippet.strip()


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
        docs_with_scores = _rank_retrieval_rows(docs_with_scores, query=query)

        evidence_items = []
        retrieval_warnings: list[str] = []
        raw_scores: list[float] = []
        for doc, score in docs_with_scores:
            if not hasattr(doc, "metadata"):
                continue
            source = doc.metadata.get("source", "notebook")
            normalized_score, raw_score = normalize_relevance_score(
                score,
                warnings=retrieval_warnings,
            )
            evidence_item = build_evidence_item(
                kind="local",
                tool="rag_search",
                url_or_path=str(source),
                snippet=_build_query_focused_snippet(
                    doc.page_content or "",
                    query=query,
                ).replace("\n", " "),
                score=normalized_score,
                metadata=getattr(doc, "metadata", None),
                warnings=retrieval_warnings,
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)
                if raw_score is not None:
                    raw_scores.append(raw_score)

        evidence = dedupe_evidence_dicts(evidence_items)
        return build_retrieval_payload(
            tool="rag_search",
            route="local",
            query=query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no local notebook evidence found",
            raw_relevance_score=max(raw_scores) if raw_scores else None,
            warnings=retrieval_warnings,
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
        docs_with_scores = _rank_retrieval_rows(docs_with_scores, query=query)

        evidence_items = []
        retrieval_warnings: list[str] = []
        raw_scores: list[float] = []
        for doc, score in docs_with_scores:
            if not hasattr(doc, "metadata"):
                continue
            source = doc.metadata.get("source", "uploaded")
            normalized_score, raw_score = normalize_relevance_score(
                score,
                warnings=retrieval_warnings,
            )
            evidence_item = build_evidence_item(
                kind="local",
                tool="upload_search",
                url_or_path=str(source),
                snippet=_build_query_focused_snippet(
                    doc.page_content or "",
                    query=query,
                ).replace("\n", " "),
                score=normalized_score,
                metadata=getattr(doc, "metadata", None),
                warnings=retrieval_warnings,
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)
                if raw_score is not None:
                    raw_scores.append(raw_score)
        evidence = dedupe_evidence_dicts(evidence_items)
        return build_retrieval_payload(
            tool="upload_search",
            route="upload",
            query=query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no uploaded file evidence found",
            raw_relevance_score=max(raw_scores) if raw_scores else None,
            warnings=retrieval_warnings,
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
