from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

from langchain_core.messages import AIMessage

from ...answer_schema import (
    AgentResponsePayloadModel,
    ClaimItem,
    SynthesisOutput,
    average_claim_confidence,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    normalize_confidence,
    render_payload_from_claims,
)
from ...contracts.routes import route_for_tool
from ...evidence import EvidenceItem
from ...planner_schema import PlannerOutput
from ..session import extract_text_content
from .prompt_builder import parse_plain_summary_segments


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


@dataclass(slots=True)
class RenderedSynthesisPayload:
    payload: AgentResponsePayloadModel
    final_answer: str
    synthesis_output: SynthesisOutput


def build_structured_synthesizer(llm_synthesizer: Any) -> Any:
    if hasattr(llm_synthesizer, "with_structured_output"):
        try:
            return llm_synthesizer.with_structured_output(
                SynthesisOutput,
                method="json_schema",
                include_raw=False,
                strict=True,
            )
        except Exception:
            return llm_synthesizer
    return llm_synthesizer


def coerce_synthesis_output(raw_value: Any) -> SynthesisOutput:
    if isinstance(raw_value, SynthesisOutput):
        return raw_value
    if isinstance(raw_value, dict):
        try:
            return SynthesisOutput.model_validate(raw_value)
        except Exception:
            return SynthesisOutput(answer=str(raw_value))

    content = extract_text_content(getattr(raw_value, "content", raw_value))
    stripped = str(content or "").strip()
    if not stripped:
        return SynthesisOutput(answer="", claims=[], confidence=None)

    try:
        return SynthesisOutput.model_validate_json(stripped)
    except Exception:
        return SynthesisOutput(answer=stripped, claims=[], confidence=None)


def coerce_structured_synthesis_result(
    result: Any,
) -> tuple[Any, AIMessage | None, Exception | None]:
    if isinstance(result, AIMessage):
        return result, result, None
    if not isinstance(result, dict):
        return result, None, None

    if not {"raw", "parsed", "parsing_error"}.intersection(result.keys()):
        return result, None, None

    raw_message = result.get("raw")
    parsed = result.get("parsed")
    parsing_error = result.get("parsing_error")

    if isinstance(parsed, SynthesisOutput):
        parsed = parsed.model_dump(mode="json")

    if not isinstance(raw_message, AIMessage):
        raw_message = None

    if parsing_error is not None and isinstance(parsing_error, Exception):
        return parsed, raw_message, parsing_error
    if parsing_error is not None:
        return parsed, raw_message, RuntimeError(str(parsing_error))
    return parsed, raw_message, None


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


def _score_evidence_candidate(*, user_input: str, candidate: EvidenceItem) -> tuple[int, int, int, int, float]:
    combined_text = " ".join(
        part.strip()
        for part in (candidate.title or "", candidate.snippet or "", candidate.url_or_path or "")
        if part and part.strip()
    )
    lowered_text = combined_text.lower()
    identifier_hits = sum(
        1 for token in _extract_identifier_tokens(user_input) if token.lower() in lowered_text
    )
    keyword_hits = len(_extract_keyword_tokens(user_input).intersection(_extract_keyword_tokens(combined_text)))
    parameter_boost = 0
    if any(hint in user_input.lower() for hint in _PARAMETER_HINTS) and "=" in combined_text and "(" in combined_text:
        parameter_boost = 1
    non_import = 0 if _is_import_only_snippet(combined_text) else 1
    numeric_score = float(candidate.score) if candidate.score is not None else float("-inf")
    return (parameter_boost, identifier_hits, keyword_hits, non_import, numeric_score)


def _has_strong_query_match(*, user_input: str, candidate: EvidenceItem) -> bool:
    parameter_boost, identifier_hits, keyword_hits, non_import, _ = _score_evidence_candidate(
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


def build_plain_summary_attach_payload(
    *,
    content: str,
    evidence_items: list[EvidenceItem],
) -> AgentResponsePayloadModel | None:
    if not evidence_items:
        return None

    limited_evidence = evidence_items[:2]
    segments = parse_plain_summary_segments(content, limit=len(limited_evidence))
    if not segments:
        return None

    adopted_pairs = list(zip(segments[: len(limited_evidence)], limited_evidence))
    if not adopted_pairs:
        return None

    claims: list[ClaimItem] = []
    for segment, evidence_item in adopted_pairs:
        source_id = str(evidence_item.source_id or "").strip()
        if not source_id:
            continue
        claims.append(
            ClaimItem(
                text=segment,
                evidence_ids=[source_id],
                confidence=normalize_confidence(evidence_item.score, clamp=True),
            )
        )

    if not claims:
        return None

    confidence = average_claim_confidence(claims)
    payload = render_payload_from_claims(
        claims=claims,
        evidence_items=limited_evidence,
        confidence=confidence,
    )
    payload.confidence = confidence
    return payload


def build_korean_template_summary_payload(
    *,
    content: str,
    evidence_items: list[EvidenceItem],
) -> AgentResponsePayloadModel | None:
    payload = build_plain_summary_attach_payload(content=content, evidence_items=evidence_items)
    if payload is None:
        return None
    rendered_answer = str(content or "").strip()
    if rendered_answer:
        payload.answer = rendered_answer
    return payload


def _looks_like_extraction_request(user_input: str) -> bool:
    normalized = str(user_input or "").strip().lower()
    return any(hint in normalized for hint in _EXTRACTION_HINTS)


def _looks_like_explainer_request(user_input: str) -> bool:
    normalized = str(user_input or "").strip().lower()
    return any(hint in normalized for hint in _EXPLAINER_HINTS)


def _query_identifiers(user_input: str) -> set[str]:
    return {
        token.lower()
        for token in _ASCII_IDENTIFIER_PATTERN.findall(str(user_input or ""))
        if token and token.lower() not in _QUERY_STOPWORDS
    }


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


def build_local_fallback_payload(
    *,
    evidence_items: list[EvidenceItem],
    retrieval_required: bool,
    generic_answer: str,
) -> AgentResponsePayloadModel:
    if evidence_items:
        return build_deterministic_grounded_payload(
            evidence_items=evidence_items,
            fallback_answer=generic_answer,
        )
    if retrieval_required:
        return build_empty_response_payload(answer="")
    return build_empty_response_payload(answer=generic_answer)


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
    return selected_routes in ({"upload"}, {"local"}) and len(evidence_items) == 1


def render_synthesis_payload(
    synthesis_output: SynthesisOutput,
    evidence_items: list[EvidenceItem],
) -> RenderedSynthesisPayload:
    payload_confidence = synthesis_output.confidence
    if payload_confidence is None:
        payload_confidence = average_claim_confidence(synthesis_output.claims)

    if synthesis_output.claims:
        payload = render_payload_from_claims(
            claims=synthesis_output.claims,
            evidence_items=evidence_items,
            confidence=payload_confidence,
            sections=synthesis_output.sections,
        )
        return RenderedSynthesisPayload(
            payload=payload,
            final_answer=payload.answer,
            synthesis_output=synthesis_output,
        )

    fallback_answer = str(synthesis_output.answer or "").strip()
    payload = build_empty_response_payload(
        answer=fallback_answer,
        confidence=payload_confidence,
        sections=synthesis_output.sections,
    )
    return RenderedSynthesisPayload(
        payload=payload,
        final_answer=payload.answer,
        synthesis_output=synthesis_output,
    )
