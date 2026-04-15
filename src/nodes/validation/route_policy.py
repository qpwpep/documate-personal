from __future__ import annotations

from typing import Any

from ...contracts import RetrievalDiagnostic
from ...evidence import EvidenceItem
from .lexical import (
    extract_code_identifiers,
    extract_keywords,
    has_exact_identifier_hit,
    identifier_overlap_count,
    keyword_overlap_count,
    non_identifier_keyword_overlap_count,
)


HYBRID_LOCAL_MIN_NORMALIZED_SCORE = 0.15


def clamp_score(value: Any) -> float | None:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, min(1.0, normalized))


def is_hybrid_compare_routes(required_routes: list[str] | None) -> bool:
    routes = {str(route or "").strip() for route in (required_routes or []) if str(route or "").strip()}
    return "docs" in routes and bool(routes.intersection({"upload", "local"}))


def route_max_score(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return max(scores)


def route_score_avg(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def route_has_strong_lexical_match(query: str, items: list[EvidenceItem]) -> bool:
    return has_exact_identifier_hit(query, items) or keyword_overlap_count(query, items) >= 2


def route_has_hybrid_local_lexical_match(query: str, items: list[EvidenceItem]) -> bool:
    identifier_hits = identifier_overlap_count(query, items)
    return bool(
        identifier_hits >= 2
        or (identifier_hits >= 1 and non_identifier_keyword_overlap_count(query, items) >= 1)
    )


def resolve_validation_query(
    *,
    route: str,
    route_query: str,
    user_input: str,
    required_routes: list[str] | None = None,
) -> str:
    normalized_query = str(route_query or "").strip()
    if (
        route in {"upload", "local"}
        and is_hybrid_compare_routes(required_routes)
        and not extract_code_identifiers(normalized_query)
        and len(extract_keywords(normalized_query)) < 2
    ):
        normalized_user_input = str(user_input or "").strip()
        if normalized_user_input:
            return normalized_user_input
    return normalized_query or str(user_input or "").strip()


def route_normalized_score(
    items: list[EvidenceItem],
    diagnostics: list[RetrievalDiagnostic] | None = None,
) -> float | None:
    diagnostic_scores = [
        clamp_score(item.normalized_score)
        for item in (diagnostics or [])
        if clamp_score(item.normalized_score) is not None
    ]
    if diagnostic_scores:
        return max(diagnostic_scores)

    return route_max_score(items)


def route_passes_validation(
    route: str,
    query: str,
    items: list[EvidenceItem],
    *,
    required_routes: list[str] | None = None,
    diagnostics: list[RetrievalDiagnostic] | None = None,
    user_input: str | None = None,
) -> bool:
    if not items:
        return False
    effective_query = resolve_validation_query(
        route=route,
        route_query=query,
        user_input=user_input or query,
        required_routes=required_routes,
    )
    normalized_score = route_normalized_score(items, diagnostics)
    if route == "docs":
        return bool(
            (normalized_score is not None and normalized_score >= 0.5)
            or route_has_strong_lexical_match(effective_query, items)
        )
    if route in {"upload", "local"} and is_hybrid_compare_routes(required_routes):
        return bool(
            (normalized_score is not None and normalized_score >= HYBRID_LOCAL_MIN_NORMALIZED_SCORE)
            or route_has_hybrid_local_lexical_match(effective_query, items)
        )
    query_identifiers = extract_code_identifiers(effective_query)
    query_keywords = extract_keywords(effective_query)
    if not query_identifiers and len(query_keywords) < 2:
        return True
    return bool(
        has_exact_identifier_hit(effective_query, items)
        or keyword_overlap_count(effective_query, items) >= 2
        or (normalized_score is not None and normalized_score > 0.0)
    )


def route_query_for_validation(
    route: str,
    diagnostics: list[RetrievalDiagnostic],
    fallback_query: str,
) -> str:
    for item in diagnostics:
        if str(item.route or "").strip() != route:
            continue
        query = str(item.query or "").strip()
        if query:
            return query
    return fallback_query


def route_error_statuses(diagnostics: list[RetrievalDiagnostic]) -> set[str]:
    return {
        str(item.status or "").strip()
        for item in diagnostics
        if str(item.status or "").strip()
    }


def route_has_warning(diagnostics: list[RetrievalDiagnostic]) -> bool:
    return any(item.warnings for item in diagnostics)
