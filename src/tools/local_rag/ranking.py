from __future__ import annotations

import re
from typing import Any


IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b")
KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[가-힣]{2,}")
KEYWORD_STOPWORDS = {
    "based",
    "code",
    "current",
    "describe",
    "docs",
    "documentation",
    "example",
    "examples",
    "explain",
    "file",
    "find",
    "how",
    "latest",
    "notebook",
    "official",
    "reference",
    "show",
    "tell",
    "upload",
    "uploaded",
    "usage",
    "used",
    "using",
    "what",
    "where",
    "공식",
    "기준",
    "노트북",
    "문법",
    "문서",
    "부분",
    "사용",
    "설명",
    "실제",
    "알려줘",
    "업로드",
    "업로드한",
    "예제",
    "위치",
    "찾아줘",
    "최신",
    "코드",
    "파일",
}
PARAMETER_HINT_PATTERN = re.compile(
    r"(?i)\b(parameter|parameters|param|params|option|options|arg|args)\b|파라미터|매개변수|옵션|인자"
)


def extract_identifiers(text: str) -> list[str]:
    identifiers: list[str] = []
    seen_lowered: set[str] = set()
    for token in IDENTIFIER_PATTERN.findall(str(text or "")):
        lowered = token.lower()
        if lowered in KEYWORD_STOPWORDS:
            continue
        if lowered not in seen_lowered:
            identifiers.append(token)
            seen_lowered.add(lowered)
    return identifiers


def extract_keywords(text: str) -> set[str]:
    return {
        token.strip().lower()
        for token in KEYWORD_PATTERN.findall(str(text or "").lower())
        if len(token.strip()) >= 2 and token.strip().lower() not in KEYWORD_STOPWORDS
    }


def parameter_query_score(query: str, content: str) -> int:
    if PARAMETER_HINT_PATTERN.search(query) is None:
        return 0
    compact = str(content or "")
    return int("=" in compact) + int("(" in compact and ")" in compact)


def lexical_query_score(query: str, content: str) -> int:
    lowered_content = str(content or "").lower()
    identifier_hits = sum(
        1 for token in extract_identifiers(query) if token.lower() in lowered_content
    )
    keyword_hits = len(extract_keywords(query).intersection(extract_keywords(content)))
    return (identifier_hits * 4) + keyword_hits + parameter_query_score(query, lowered_content)


def rank_retrieval_rows(
    docs_with_scores: list[tuple[Any, float | None]],
    *,
    query: str,
) -> list[tuple[Any, float | None]]:
    def _sort_key(item: tuple[Any, float | None]) -> tuple[int, float]:
        doc, score = item
        content = getattr(doc, "page_content", "")
        lexical = lexical_query_score(query, str(content or ""))
        numeric_score = float(score) if score is not None else float("inf")
        return (-lexical, numeric_score)

    return sorted(docs_with_scores, key=_sort_key)
