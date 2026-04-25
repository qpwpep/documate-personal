from __future__ import annotations

import re
from functools import lru_cache

from src.core.evidence import EvidenceItem
from src.core.rules import get_rules_config


def validation_rules():
    return get_rules_config().validation


@lru_cache(maxsize=1)
def code_identifier_pattern():
    return re.compile(validation_rules().code_identifier_pattern)


@lru_cache(maxsize=1)
def embedded_code_identifier_pattern():
    return re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


@lru_cache(maxsize=1)
def keyword_pattern():
    return re.compile(validation_rules().keyword_pattern)


def extract_code_identifiers(text: str) -> set[str]:
    matches = set(code_identifier_pattern().findall(str(text or "")))
    matches.update(embedded_code_identifier_pattern().findall(str(text or "")))
    return {
        token.lower()
        for token in matches
        if token and token.lower() not in set(validation_rules().keyword_stopwords)
    }


def extract_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    stopwords = set(validation_rules().keyword_stopwords)
    for token in keyword_pattern().findall(str(text or "").lower()):
        normalized = token.strip().lower()
        if len(normalized) < 2 or normalized in stopwords:
            continue
        keywords.add(normalized)
    return keywords


def combine_evidence_text(items: list[EvidenceItem]) -> str:
    return " ".join(
        part.strip().lower()
        for item in items
        for part in (
            item.title or "",
            item.snippet or "",
            str(item.code_metadata or ""),
            item.url_or_path or "",
            item.document_id or "",
            item.source_id or "",
        )
        if part and part.strip()
    )


def has_exact_identifier_hit(query: str, items: list[EvidenceItem]) -> bool:
    identifiers = extract_code_identifiers(query)
    if not identifiers:
        return False
    combined_text = combine_evidence_text(items)
    return any(identifier in combined_text for identifier in identifiers)


def identifier_overlap_count(query: str, items: list[EvidenceItem]) -> int:
    identifiers = extract_code_identifiers(query)
    if not identifiers:
        return 0
    combined_text = combine_evidence_text(items)
    return sum(1 for identifier in identifiers if identifier in combined_text)


def keyword_overlap_count(query: str, items: list[EvidenceItem]) -> int:
    query_keywords = extract_keywords(query)
    if not query_keywords:
        return 0
    evidence_keywords = extract_keywords(combine_evidence_text(items))
    return len(query_keywords.intersection(evidence_keywords))


def non_identifier_keyword_overlap_count(query: str, items: list[EvidenceItem]) -> int:
    query_identifiers = extract_code_identifiers(query)
    query_keywords = extract_keywords(query) - query_identifiers
    if not query_keywords:
        return 0
    evidence_keywords = extract_keywords(combine_evidence_text(items))
    return len(query_keywords.intersection(evidence_keywords))
