from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse

from ...rules import get_rules_config


def docs_search_rules():
    return get_rules_config().docs_search


def normalize_include_domains(raw_values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in raw_values:
        candidate = value.strip()
        if not candidate:
            continue
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
        domain = (parsed.netloc or parsed.path).strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if domain and domain not in normalized:
            normalized.append(domain)
    return normalized


def normalize_domain(value: str) -> str:
    candidate = str(value or "").strip().lower()
    if candidate.startswith("www."):
        candidate = candidate[4:]
    return candidate


def normalize_path_prefix(path: str) -> str:
    normalized = "/" + str(path or "").strip().lstrip("/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized == "/":
        return normalized
    return normalized if normalized.endswith("/") else normalized + "/"


def is_allowed_doc_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    domain = normalize_domain(parsed.netloc)
    allowed_prefixes = docs_search_rules().allowed_doc_path_prefixes.get(domain)
    if not allowed_prefixes:
        return False
    normalized_path = normalize_path_prefix(parsed.path or "/")
    return any(normalized_path.startswith(prefix) for prefix in allowed_prefixes)


def is_valid_doc_result(*, url: str, title: Any, snippet: Any) -> bool:
    if not is_allowed_doc_url(url):
        return False

    combined = " ".join(
        part.strip().lower()
        for part in [str(url or ""), str(title or ""), str(snippet or "")]
        if str(part or "").strip()
    )
    if not combined:
        return False

    return not any(marker in combined for marker in docs_search_rules().error_page_markers)


def normalized_domain_set(allowed_domains: list[str] | None) -> set[str]:
    if not allowed_domains:
        return set()
    return {normalize_domain(domain) for domain in allowed_domains if normalize_domain(domain)}


def result_matches_domains(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    parsed = urlparse(str(url or "").strip())
    return normalize_domain(parsed.netloc) in allowed_domains


def infer_docs_query_hint(query: str) -> tuple[str, list[str], list[str]] | None:
    lowered = str(query or "").lower()
    for hint in docs_search_rules().query_hints:
        if any(query_hint_matches(lowered, identifier, match_mode=hint.match_mode) for identifier in hint.identifiers):
            return hint.library_name, list(hint.domains), list(hint.fallback_queries)
    return None


def query_hint_matches(query: str, identifier: str, *, match_mode: Literal["contains", "word"]) -> bool:
    normalized_identifier = str(identifier or "").strip().lower()
    if not normalized_identifier:
        return False
    if match_mode == "word":
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(normalized_identifier)}(?![A-Za-z0-9_])", query) is not None
    return normalized_identifier in query
