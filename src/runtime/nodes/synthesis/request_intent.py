from __future__ import annotations

from src.core.evidence import EvidenceItem
from src.core.planner_schema import PlannerOutput
from src.runtime.nodes.synthesis.evidence_selection import _extract_identifier_tokens


_EXTRACTION_HINTS = (
    "extract",
    "quote",
    "snippet",
    "verbatim",
    "exact",
    "raw code",
    "code snippet",
    "cell",
    "line",
    "인용",
    "발췌",
    "추출",
    "원문",
    "그대로",
    "코드 조각",
)
_EXPLAINER_HINTS = (
    "explain",
    "describe",
    "summarize",
    "parameter",
    "parameters",
    "option",
    "options",
    "compare",
    "설명",
    "정리",
    "요약",
    "파라미터",
    "매개변수",
    "옵션",
    "비교",
)


def _looks_like_extraction_request(user_input: str) -> bool:
    normalized = str(user_input or "").strip().lower()
    return any(hint in normalized for hint in _EXTRACTION_HINTS)


def _looks_like_explainer_request(user_input: str) -> bool:
    normalized = str(user_input or "").strip().lower()
    return any(hint in normalized for hint in _EXPLAINER_HINTS)


def _query_identifiers(user_input: str) -> set[str]:
    return {token.lower() for token in _extract_identifier_tokens(user_input)}


def _evidence_contains_identifier(user_input: str, evidence_items: list[EvidenceItem]) -> bool:
    identifiers = _query_identifiers(user_input)
    if not identifiers:
        return False
    combined_text = " ".join(
        part.lower()
        for item in evidence_items
        for part in (str(item.snippet or ""), str(item.title or ""), str(item.url_or_path or ""))
    )
    return any(identifier in combined_text for identifier in identifiers)


def should_use_deterministic_grounded_direct(
    *,
    user_input: str,
    planner_output: PlannerOutput,
    evidence_items: list[EvidenceItem],
) -> bool:
    if not planner_output.use_retrieval or not planner_output.tasks:
        return False
    if not 1 <= len(evidence_items) <= 2:
        return False
    selected_routes = {task.route for task in planner_output.tasks}
    evidence_kinds = {str(item.kind or "").strip().lower() for item in evidence_items}
    if "official" in evidence_kinds:
        return False
    if _looks_like_explainer_request(user_input):
        return False
    if not _looks_like_extraction_request(user_input):
        return False
    if not _evidence_contains_identifier(user_input, evidence_items):
        return False
    return selected_routes == {"upload"} and len(evidence_items) == 1
