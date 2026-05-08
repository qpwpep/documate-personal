from __future__ import annotations

import re


_IDENTIFIER_SEGMENT = r"[A-Za-z_][A-Za-z0-9_]*"
_IDENTIFIER_SEGMENT_PATTERN = re.compile(_IDENTIFIER_SEGMENT)
_DOTTED_CHAIN_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_]){_IDENTIFIER_SEGMENT}(?:\s*\.\s*{_IDENTIFIER_SEGMENT})+(?![A-Za-z0-9_])"
)
_SPACED_DOT_PATTERN = re.compile(r"\s+\.\s*|\.\s+")
_TRAILING_IDENTIFIER_DOT_PATTERN = re.compile(
    rf"(?<![A-Za-z0-9_])({_IDENTIFIER_SEGMENT})\.(?=\s|$)"
)


def canonicalize_docs_query_text(text: str) -> str:
    """Normalize API-like dotted tokens before sending a docs search query."""
    normalized = _DOTTED_CHAIN_PATTERN.sub(_replace_spaced_dotted_chain, str(text or ""))
    normalized = _TRAILING_IDENTIFIER_DOT_PATTERN.sub(r"\1", normalized)
    return " ".join(normalized.split())


def normalize_identifier_reference_text(text: str) -> str:
    """Normalize text used for exact identifier extraction and matching."""
    normalized = _DOTTED_CHAIN_PATTERN.sub(_replace_spaced_dotted_chain, str(text or ""))
    normalized = _TRAILING_IDENTIFIER_DOT_PATTERN.sub(r"\1", normalized)
    return " ".join(normalized.split())


def normalize_identifier_token(token: str) -> str:
    return str(token or "").strip().rstrip(".")


def _replace_spaced_dotted_chain(match: re.Match[str]) -> str:
    raw = match.group(0)
    if not _SPACED_DOT_PATTERN.search(raw):
        return raw

    segments = _IDENTIFIER_SEGMENT_PATTERN.findall(raw)
    if len(segments) < 2:
        return raw

    first = segments[0]
    second = segments[1]
    if len(segments) == 2 and _starts_lowercase(first) and second == second.lower():
        return f"{first} {second}"
    if (
        len(segments) == 2
        and not _starts_lowercase(first)
        and not _starts_lowercase(second)
        and min(len(first), len(second)) >= 4
    ):
        return f"{first}{second}"
    if len(segments) == 2:
        return raw
    return ".".join(segments)


def _starts_lowercase(value: str) -> bool:
    return bool(value) and value[0].islower()
