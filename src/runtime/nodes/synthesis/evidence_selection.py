from __future__ import annotations

import re

from src.core.contracts.routes import route_for_tool
from src.core.evidence import EvidenceItem
from src.core.planner_schema import PlannerOutput


_ASCII_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z][A-Za-z0-9._-]{1,}\b")
_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[가-힣]{2,}")
_QUERY_STOPWORDS = {
    "uploaded",
    "upload",
    "notebook",
    "file",
    "current",
    "this",
    "show",
    "find",
    "code",
    "example",
    "examples",
    "usage",
    "official",
    "docs",
    "documentation",
    "the",
}
_PARAMETER_HINTS = ("parameter", "parameters", "param", "파라미터", "매개변수", "옵션")
_IMPORT_ONLY_PATTERN = re.compile(
    r"^\s*(?:from\s+\S+\s+import\s+.+|import\s+\S+(?:\s+as\s+\S+)?)\s*$",
    flags=re.I,
)


def route_for_evidence(item: EvidenceItem) -> str:
    return route_for_tool(str(item.tool or ""))


def _extract_identifier_tokens(text: str) -> list[str]:
    identifiers: list[str] = []
    seen_lowered: set[str] = set()
    for token in _ASCII_IDENTIFIER_PATTERN.findall(str(text or "")):
        lowered = token.lower()
        if lowered in _QUERY_STOPWORDS:
            continue
        if lowered not in seen_lowered:
            identifiers.append(token)
            seen_lowered.add(lowered)
    return identifiers


def _extract_keyword_tokens(text: str) -> set[str]:
    return {
        token.strip().lower()
        for token in _KEYWORD_PATTERN.findall(str(text or "").lower())
        if len(token.strip()) >= 2 and token.strip().lower() not in _QUERY_STOPWORDS
    }


def _is_import_only_snippet(text: str) -> bool:
    compact = " ".join(str(text or "").replace("\r", "\n").split()).strip()
    if not compact or "=" in compact:
        return False
    return _IMPORT_ONLY_PATTERN.match(compact) is not None


def _code_metadata_search_text(candidate: EvidenceItem) -> str:
    metadata = candidate.code_metadata
    if not isinstance(metadata, dict):
        return ""

    parts: list[str] = []
    for option in metadata.get("option_literals") or []:
        option_text = " ".join(str(option or "").split()).strip()
        if option_text:
            parts.append(option_text)

    for call in metadata.get("calls") or []:
        if not isinstance(call, dict):
            continue
        call_name = str(call.get("call_name") or "").strip()
        if call_name:
            parts.append(call_name)
        kwargs = call.get("kwargs")
        if isinstance(kwargs, dict):
            for key, value in kwargs.items():
                key_text = str(key or "").strip()
                value_text = str(value or "").strip()
                if key_text:
                    parts.append(key_text)
                if key_text and value_text:
                    parts.append(f"{key_text}={value_text}")
                elif value_text:
                    parts.append(value_text)
    return " ".join(parts)


def _score_evidence_candidate(
    *,
    user_input: str,
    candidate: EvidenceItem,
) -> tuple[int, int, int, int, int, float]:
    code_metadata_text = _code_metadata_search_text(candidate)
    combined_text = " ".join(
        part.strip()
        for part in (
            candidate.title or "",
            candidate.snippet or "",
            candidate.url_or_path or "",
            code_metadata_text,
        )
        if part and part.strip()
    )
    lowered_text = combined_text.lower()
    identifier_hits = sum(
        1 for token in _extract_identifier_tokens(user_input) if token.lower() in lowered_text
    )
    keyword_hits = len(_extract_keyword_tokens(user_input).intersection(_extract_keyword_tokens(combined_text)))
    code_metadata_hits = len(
        _extract_keyword_tokens(user_input).intersection(_extract_keyword_tokens(code_metadata_text))
    )
    parameter_boost = 0
    if any(hint in user_input.lower() for hint in _PARAMETER_HINTS) and "=" in combined_text and ("(" in combined_text or code_metadata_text):
        parameter_boost = 1
    non_import = 0 if _is_import_only_snippet(combined_text) else 1
    numeric_score = float(candidate.score) if candidate.score is not None else float("-inf")
    return (parameter_boost, identifier_hits, keyword_hits, code_metadata_hits, non_import, numeric_score)


def _has_strong_query_match(*, user_input: str, candidate: EvidenceItem) -> bool:
    parameter_boost, identifier_hits, keyword_hits, _code_metadata_hits, non_import, _ = _score_evidence_candidate(
        user_input=user_input,
        candidate=candidate,
    )
    if identifier_hits > 0:
        return True
    if keyword_hits >= 2 and non_import > 0:
        return True
    return bool(parameter_boost > 0 and keyword_hits >= 1)


def _select_best_evidence_for_query(
    *,
    user_input: str,
    candidates: list[EvidenceItem],
) -> EvidenceItem | None:
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda item: _score_evidence_candidate(user_input=user_input, candidate=item),
    )


def _select_top_evidence_items(
    *,
    user_input: str,
    evidence_items: list[EvidenceItem],
    limit: int,
) -> list[EvidenceItem]:
    ranked = sorted(
        evidence_items,
        key=lambda item: _score_evidence_candidate(user_input=user_input, candidate=item),
        reverse=True,
    )
    return ranked[: max(0, limit)]


def _extend_unique(selected: list[EvidenceItem], candidates: list[EvidenceItem]) -> None:
    seen_ids = {
        str(item.source_id or "").strip()
        for item in selected
        if str(item.source_id or "").strip()
    }
    for candidate in candidates:
        source_id = str(candidate.source_id or "").strip()
        if source_id and source_id in seen_ids:
            continue
        selected.append(candidate)
        if source_id:
            seen_ids.add(source_id)


def select_primary_evidence_items(
    *,
    user_input: str,
    evidence_items: list[EvidenceItem],
    planner_output: PlannerOutput,
) -> list[EvidenceItem]:
    if not evidence_items:
        return []

    if planner_output.use_retrieval and planner_output.tasks:
        requested_routes: list[str] = []
        for task in planner_output.tasks:
            route = str(task.route or "")
            if route and route not in requested_routes:
                requested_routes.append(route)

        is_hybrid_routes = len(requested_routes) > 1 and "docs" in requested_routes and any(
            route in {"upload", "local"} for route in requested_routes
        )
        if len(requested_routes) == 1 and requested_routes[0] in {"upload", "local"}:
            route = requested_routes[0]
            route_matches = [item for item in evidence_items if route_for_evidence(item) == route]
            if route_matches:
                return _select_top_evidence_items(
                    user_input=user_input,
                    evidence_items=route_matches,
                    limit=2,
                )

        selected: list[EvidenceItem] = []
        seen_routes: set[str] = set()
        for task in planner_output.tasks:
            route = str(task.route or "")
            route_matches = [item for item in evidence_items if route_for_evidence(item) == route]
            if route in seen_routes and not is_hybrid_routes:
                continue

            if is_hybrid_routes:
                strong_route_matches = [
                    item
                    for item in route_matches
                    if _has_strong_query_match(user_input=user_input, candidate=item)
                ]
                if not strong_route_matches:
                    seen_routes.add(route)
                    continue
                route_top_matches = _select_top_evidence_items(
                    user_input=user_input,
                    evidence_items=strong_route_matches,
                    limit=1,
                )
                _extend_unique(selected, route_top_matches)
                seen_routes.add(route)
                continue

            match = _select_best_evidence_for_query(
                user_input=user_input,
                candidates=route_matches,
            )
            if match is not None:
                selected.append(match)
                seen_routes.add(route)
        if is_hybrid_routes:
            return selected[:2]
        if selected:
            return selected[:2]

    return _select_top_evidence_items(
        user_input=user_input,
        evidence_items=evidence_items,
        limit=2,
    )


def select_grounded_fallback_evidence_items(
    *,
    user_input: str,
    evidence_items: list[EvidenceItem],
    planner_output: PlannerOutput,
) -> list[EvidenceItem]:
    primary_items = select_primary_evidence_items(
        user_input=user_input,
        evidence_items=evidence_items,
        planner_output=planner_output,
    )
    if primary_items:
        return primary_items

    requested_routes: list[str] = []
    for task in planner_output.tasks or []:
        route = str(task.route or "")
        if route and route not in requested_routes:
            requested_routes.append(route)

    is_hybrid_routes = len(requested_routes) > 1 and "docs" in requested_routes and any(
        route in {"upload", "local"} for route in requested_routes
    )
    if is_hybrid_routes:
        selected: list[EvidenceItem] = []
        for route in requested_routes:
            route_matches = [item for item in evidence_items if route_for_evidence(item) == route]
            strong_route_matches = [
                item
                for item in route_matches
                if _has_strong_query_match(user_input=user_input, candidate=item)
            ]
            if not strong_route_matches:
                continue
            match = _select_best_evidence_for_query(
                user_input=user_input,
                candidates=strong_route_matches,
            )
            if match is not None:
                _extend_unique(selected, [match])
        return selected[:2]

    return _select_top_evidence_items(
        user_input=user_input,
        evidence_items=evidence_items,
        limit=2,
    )
