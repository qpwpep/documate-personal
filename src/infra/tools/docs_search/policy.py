from __future__ import annotations

import re
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from src.core.rules import get_rules_config


_NUMPY_VERSIONED_DOC_PATH_PATTERN = re.compile(r"^/doc/\d+(?:\.\d+)*/")
_NUMPY_DOC_TITLE_VERSION_PATTERN = re.compile(
    r"(\bNumPy)\s+v?\d+(?:\.\d+)*(?:[A-Za-z0-9.+-]*)?(\s+Manual\b)",
    re.IGNORECASE,
)


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


def canonicalize_doc_url(url: str) -> str:
    candidate = str(url or "").strip()
    if not candidate:
        return ""

    parsed = urlparse(candidate)
    domain = normalize_domain(parsed.netloc)
    if domain == "numpy.org":
        stable_path = _NUMPY_VERSIONED_DOC_PATH_PATTERN.sub("/doc/stable/", parsed.path)
        if stable_path != parsed.path:
            return urlunparse(parsed._replace(path=stable_path))
    return candidate


def canonicalize_doc_title(*, title: Any, original_url: str, canonical_url: str) -> Any:
    title_text = str(title).strip() if title else ""
    if not title_text:
        return title

    original = str(original_url or "").strip()
    canonical = str(canonical_url or "").strip()
    if original == canonical:
        return title

    parsed = urlparse(canonical)
    if normalize_domain(parsed.netloc) == "numpy.org" and parsed.path.startswith("/doc/stable/"):
        normalized_title = _NUMPY_DOC_TITLE_VERSION_PATTERN.sub(r"\1\2", title_text)
        return " ".join(normalized_title.split())
    return title


def is_allowed_doc_url(url: str) -> bool:
    parsed = urlparse(canonicalize_doc_url(url))
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
    best_match: tuple[tuple[int, int, int], tuple[str, list[str], list[str]]] | None = None
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
            1 if non_library_matches else 0,
            len(matched_identifiers),
            max(len(str(identifier or "")) for identifier in matched_identifiers),
        )
        candidate = (hint.library_name, list(hint.domains), list(hint.fallback_queries))
        if best_match is None or score > best_match[0]:
            best_match = (score, candidate)
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
