from __future__ import annotations

from dataclasses import dataclass
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
    "cell",
    "line",
    "find",
    "show",
    "where",
    "locate",
    "인용",
    "추출",
    "줄",
    "셀",
    "코드",
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
                include_raw=True,
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

    if not isinstance(raw_message, AIMessage):
        raw_message = None

    if parsing_error is not None and isinstance(parsing_error, Exception):
        return parsed, raw_message, parsing_error
    if parsing_error is not None:
        return parsed, raw_message, RuntimeError(str(parsing_error))
    return parsed, raw_message, None


def route_for_evidence(item: EvidenceItem) -> str:
    return route_for_tool(str(item.tool or ""))


def select_primary_evidence_items(
    *,
    evidence_items: list[EvidenceItem],
    planner_output: PlannerOutput,
) -> list[EvidenceItem]:
    if not evidence_items:
        return []

    if planner_output.use_retrieval and planner_output.tasks:
        selected: list[EvidenceItem] = []
        seen_routes: set[str] = set()
        for task in planner_output.tasks:
            route = str(task.route or "")
            if route in seen_routes:
                continue
            match = next((item for item in evidence_items if route_for_evidence(item) == route), None)
            if match is not None:
                selected.append(match)
                seen_routes.add(route)
        if selected:
            return selected[:2]

    return evidence_items[:2]


def select_grounded_fallback_evidence_items(
    *,
    evidence_items: list[EvidenceItem],
    planner_output: PlannerOutput,
) -> list[EvidenceItem]:
    return select_primary_evidence_items(
        evidence_items=evidence_items,
        planner_output=planner_output,
    ) or evidence_items[:2]


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
    return any(hint in normalized for hint in _EXTRACTION_HINTS) or any(
        hint in normalized for hint in ("원문", "발췌", "추출", "그대로", "코드", "셀")
    )


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
    if not _looks_like_extraction_request(user_input):
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
    )
    return RenderedSynthesisPayload(
        payload=payload,
        final_answer=payload.answer,
        synthesis_output=synthesis_output,
    )
