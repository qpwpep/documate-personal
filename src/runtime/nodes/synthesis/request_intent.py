from __future__ import annotations


def is_explicit_source_extraction(user_input: str) -> bool:
    query = user_input.lower()
    extraction = any(marker in query for marker in (
        "extract", "quote", "verbatim", "raw code", "code snippet", "인용", "발췌", "추출", "원문", "그대로", "코드 조각",
    ))
    explanation = any(marker in query for marker in (
        "explain", "describe", "summarize", "compare", "설명", "정리", "요약", "비교", "옵션", "매개변수",
    ))
    return extraction and not explanation
