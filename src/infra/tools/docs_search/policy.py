from __future__ import annotations

import re
from typing import Literal
from urllib.parse import unquote, urlparse

from src.core.rules import get_rules_config
from src.infra.tools.docs_search.normalization import canonicalize_docs_query_text


DocUrlFilterReason = Literal["invalid_url", "path_prefix"]


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
    return doc_url_filter_reason(url) is None


def doc_url_filter_reason(url: str) -> DocUrlFilterReason | None:
    """Check the provider's original source without rewriting its version."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "invalid_url"
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return "invalid_url"
    domain = normalize_domain(parsed.netloc)
    rules = docs_search_rules()
    allowed_prefixes = rules.allowed_doc_path_prefixes.get(domain)
    if not allowed_prefixes:
        return "invalid_url"
    path = unquote(parsed.path or "/")
    if "\\" in path or any(part in {".", ".."} for part in path.split("/")):
        return "path_prefix"
    normalized_path = normalize_path_prefix(path)
    if any(normalized_path.startswith(normalize_path_prefix(prefix)) for prefix in allowed_prefixes):
        return None
    if any(re.match(pattern, normalized_path) for pattern in rules.allowed_doc_path_patterns.get(domain, [])):
        return None
    return "path_prefix"


def normalized_domain_set(allowed_domains: list[str] | None) -> set[str]:
    if not allowed_domains:
        return set()
    return {normalize_domain(domain) for domain in allowed_domains if normalize_domain(domain)}


def result_matches_domains(url: str, allowed_domains: set[str]) -> bool:
    if not allowed_domains:
        return True
    parsed = urlparse(str(url or "").strip())
    return normalize_domain(parsed.netloc) in allowed_domains


def infer_docs_library(query: str) -> str | None:
    lowered = canonicalize_docs_query_text(query).lower()
    best_match: tuple[tuple[int, int, int, int], str] | None = None
    for hint in docs_search_rules().query_hints:
        matched_identifiers = [
            identifier
            for identifier in hint.identifiers
            if query_hint_matches(lowered, identifier, match_mode=hint.match_mode)
        ]
        if not matched_identifiers:
            continue
        library_name = str(hint.library_name or "").strip().lower()
        non_library_matches = [
            identifier
            for identifier in matched_identifiers
            if str(identifier or "").strip().lower() != library_name
        ]
        score = (
            1 if query_hint_matches(lowered, library_name, match_mode="word") else 0,
            1 if non_library_matches else 0,
            len(matched_identifiers),
            max(len(str(identifier or "")) for identifier in matched_identifiers),
        )
        if best_match is None or score > best_match[0]:
            best_match = (score, hint.library_name)
    if best_match is not None:
        return best_match[1]
    return None


def query_hint_matches(query: str, identifier: str, *, match_mode: Literal["contains", "word"]) -> bool:
    normalized_identifier = str(identifier or "").strip().lower()
    if not normalized_identifier:
        return False
    if match_mode == "word":
        return re.search(rf"(?<![A-Za-z0-9_]){re.escape(normalized_identifier)}(?![A-Za-z0-9_])", query) is not None
    return normalized_identifier in query
