from __future__ import annotations

from typing import Any

from src.infra.chroma_store import normalize_l2_distance
from src.infra.tools._common import build_evidence_item, dedupe_evidence_dicts, to_float_or_none
from src.infra.tools.local_rag.ranking import extract_identifiers, extract_keywords, lexical_query_score


SNIPPET_CHAR_LIMIT = 500
SHORT_DOCUMENT_CHAR_LIMIT = 1200
QUERY_WINDOW_LINE_LIMIT = 5


def score_ranked_rows(
    docs_with_scores: list[tuple[Any, float | None]],
) -> list[tuple[Any, float | None, float | None]]:
    scored_rows: list[tuple[Any, float | None, float | None]] = []
    for doc, score in docs_with_scores:
        raw_score = to_float_or_none(score)
        scored_rows.append((doc, normalize_l2_distance(raw_score), raw_score))
    return scored_rows


def build_query_focused_snippet(text: str, *, query: str, max_length: int = SNIPPET_CHAR_LIMIT) -> str:
    normalized = str(text or "").strip()
    candidate_starts = _build_candidate_starts(normalized, query=query, max_length=max_length)
    if not candidate_starts:
        return normalized if len(normalized) <= max_length else normalized[:max_length]
    if len(normalized) <= max_length and len(normalized.splitlines()) <= QUERY_WINDOW_LINE_LIMIT:
        return normalized
    query_tokens = _query_tokens(query)
    if len(normalized) <= max_length and not _has_query_token_hit(
        text=normalized,
        query_tokens=query_tokens,
    ):
        return normalized

    query_identifiers = extract_identifiers(query)
    best_window = max(
        (
            (
                lexical_query_score(query, _slice_window(normalized, start=start, max_length=max_length)),
                _identifier_hit_count(
                    query_identifiers=query_identifiers,
                    text=_slice_window(normalized, start=start, max_length=max_length),
                ),
                start,
                start,
            )
            for start in candidate_starts
        ),
        key=lambda item: (item[0], item[1], item[2]),
    )
    return _slice_window(normalized, start=best_window[3], max_length=max_length)


def build_local_snippet(
    text: str,
    *,
    query: str,
    metadata: dict[str, Any] | None = None,
    max_length: int = SNIPPET_CHAR_LIMIT,
) -> str:
    normalized = str(text or "").strip()
    if not normalized:
        return ""
    metadata = dict(metadata or {})
    if _should_preserve_full_chunk(metadata=metadata) and _looks_like_code_extraction_query(query):
        return normalized
    return build_query_focused_snippet(normalized, query=query, max_length=max_length)


def _looks_like_code_extraction_query(query: str) -> bool:
    normalized = str(query or "").strip().lower()
    return any(
        marker in normalized
        for marker in (
            "extract",
            "quote",
            "snippet",
            "verbatim",
            "exact",
            "raw code",
            "code snippet",
            "line",
            "cell",
        )
    )


def _should_preserve_full_chunk(*, metadata: dict[str, Any]) -> bool:
    document_chunk_count = _coerce_non_negative_int(metadata.get("document_chunk_count"))
    document_char_count = _coerce_non_negative_int(metadata.get("document_char_count"))
    return document_chunk_count == 1 or (
        document_char_count > 0 and document_char_count <= SHORT_DOCUMENT_CHAR_LIMIT
    )


def _build_candidate_starts(text: str, *, query: str, max_length: int) -> list[int]:
    starts = {0}
    line_starts = _line_start_offsets(text)
    starts.update(line_starts)

    lowered_text = text.lower()
    for token in _query_tokens(query):
        start_index = 0
        lowered_token = token.lower()
        while lowered_token:
            match_index = lowered_text.find(lowered_token, start_index)
            if match_index < 0:
                break
            line_start = _line_start_for_offset(text, match_index)
            starts.add(line_start)
            if match_index - line_start >= max_length:
                starts.add(max(0, match_index - (max_length // 2)))
            start_index = match_index + len(lowered_token)

    return sorted(start for start in starts if 0 <= start < len(text))


def _query_tokens(query: str) -> list[str]:
    identifiers = extract_identifiers(query)
    if identifiers:
        return identifiers
    return sorted(extract_keywords(query), key=len, reverse=True)


def _has_query_token_hit(*, text: str, query_tokens: list[str]) -> bool:
    lowered_text = text.lower()
    return any(str(token or "").lower() in lowered_text for token in query_tokens)


def _identifier_hit_count(*, query_identifiers: list[str], text: str) -> int:
    lowered_text = text.lower()
    return sum(1 for token in query_identifiers if token.lower() in lowered_text)


def _line_start_offsets(text: str) -> list[int]:
    starts = [0]
    for index, char in enumerate(text):
        if char == "\n" and index + 1 < len(text):
            starts.append(index + 1)
    return starts


def _line_start_for_offset(text: str, offset: int) -> int:
    line_start = text.rfind("\n", 0, max(0, offset))
    if line_start < 0:
        return 0
    return line_start + 1


def _slice_window(text: str, *, start: int, max_length: int) -> str:
    line_limited_end = _line_end_after_n_lines(
        text,
        start=start,
        line_limit=QUERY_WINDOW_LINE_LIMIT,
    )
    end = min(len(text), start + max_length, line_limited_end)
    if end < len(text):
        line_end = text.rfind("\n", start + 1, end)
        if line_end > start:
            end = line_end
    snippet = text[start:end].strip()
    if snippet:
        return snippet
    return text[start : min(len(text), start + max_length)].strip()


def _line_end_after_n_lines(text: str, *, start: int, line_limit: int) -> int:
    if line_limit <= 0:
        return min(len(text), start)
    current = max(0, start)
    for _ in range(line_limit):
        next_newline = text.find("\n", current)
        if next_newline < 0:
            return len(text)
        current = next_newline + 1
    return current


def _coerce_non_negative_int(value: Any) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return 0
    return max(0, coerced)


def build_local_evidence_bundle(
    docs_with_scores: list[tuple[Any, float | None]],
    *,
    query: str,
    tool_name: str,
    default_source: str,
) -> tuple[list[dict[str, Any]], list[float], list[float], list[str]]:
    evidence_items = []
    retrieval_warnings: list[str] = []
    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    for doc, score, raw_score in score_ranked_rows(docs_with_scores):
        if not hasattr(doc, "metadata"):
            continue
        source = doc.metadata.get("source", default_source)
        evidence_item = build_evidence_item(
            kind="local",
            tool=tool_name,
            url_or_path=str(source),
            snippet=build_local_snippet(
                doc.page_content or "",
                query=query,
                metadata=getattr(doc, "metadata", None),
            ).replace("\n", " "),
            score=score,
            metadata=getattr(doc, "metadata", None),
            warnings=retrieval_warnings,
        )
        if evidence_item is not None:
            evidence_items.append(evidence_item)
            if raw_score is not None:
                raw_scores.append(raw_score)
            if score is not None:
                normalized_scores.append(score)
    return dedupe_evidence_dicts(evidence_items), normalized_scores, raw_scores, retrieval_warnings
