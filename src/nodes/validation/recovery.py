from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import AIMessage

from ...answer_schema import (
    AgentResponsePayloadModel,
    AnswerSection,
    ClaimItem,
    SynthesisOutput,
    average_claim_confidence,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    clean_grounded_text,
    normalize_answer_sections,
    normalize_confidence,
    render_payload_from_claims,
    render_sections_text,
    resolve_answer_text,
    summarize_grounded_text,
)
from ...request_contracts import infer_answer_contract
from ...contracts import GraphState, ResponseState
from ...contracts.debug import RetryReason
from ...contracts.routes import route_for_tool
from ..retry import build_followup_from_routes
from .evidence_validator import ValidationAssessment, ValidationSnapshot


def build_response_payload_updates(
    payload: AgentResponsePayloadModel,
    *,
    attempt: int,
) -> GraphState:
    synthesis_output = SynthesisOutput(
        answer=payload.answer,
        claims=payload.claims,
        confidence=payload.confidence,
        sections=payload.sections,
    )
    return {
        "messages": [AIMessage(content=payload.answer)],
        "response": ResponseState(
            final_answer=payload.answer,
            payload=payload,
            synthesis_output=synthesis_output,
            synthesis_attempt=attempt,
        ),
    }


def build_followup_updates(answer: str, *, attempt: int) -> GraphState:
    return build_response_payload_updates(
        build_empty_response_payload(answer=answer),
        attempt=attempt,
    )


def _heading_for_section_kind(kind: str, *, snapshot: ValidationSnapshot) -> str:
    headings = {
        "summary": "요약",
        "checklist": "체크리스트",
        "steps": "단계별 안내",
        "official_docs": "공식 문서",
        "upload_code": "업로드 코드" if "upload" in snapshot.required_routes else "로컬 코드",
        "comparison": "비교",
        "interpretation_a": "해석 A",
        "interpretation_b": "해석 B",
    }
    return headings.get(str(kind or "").strip(), str(kind or "").strip())


def _ordered_unique_lines(lines: list[str]) -> list[str]:
    normalized_lines: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = str(line or "").strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        normalized_lines.append(normalized)
    return normalized_lines


def _render_claim_lines(
    *,
    claims: list[ClaimItem],
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
) -> list[str]:
    return _ordered_unique_lines(
        [
            _summarize_claim(claim=claim, snapshot=snapshot, payload=payload)
            for claim in claims
        ]
    )


def _fallback_route_line(
    *,
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
    routes: set[str],
) -> str:
    evidence_item = _top_evidence_item_for_routes(snapshot=snapshot, routes=routes)
    if evidence_item is None:
        return ""
    route = route_for_tool(str(evidence_item.tool or "")) or next(iter(routes), "")
    claim = _claim_from_evidence_item(evidence_item=evidence_item, route=route)
    if claim is None:
        return ""
    return _summarize_claim(claim=claim, snapshot=snapshot, payload=payload)


def _section_body_from_claims(
    *,
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
    routes: set[str] | None = None,
    mode: str = "paragraph",
) -> str:
    selected_claims = payload.claims
    if routes is not None:
        selected_claims = _claims_for_routes(claims=payload.claims, snapshot=snapshot, routes=routes)

    lines = _render_claim_lines(claims=selected_claims, snapshot=snapshot, payload=payload)
    if not lines and routes is not None:
        fallback_line = _fallback_route_line(snapshot=snapshot, payload=payload, routes=routes)
        if fallback_line:
            lines = [fallback_line]
    if not lines:
        return ""
    if mode == "checklist":
        return "\n".join(f"- {line}" for line in lines)
    if mode == "steps":
        return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))
    return "\n".join(lines)


def _comparison_section_body(
    *,
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
) -> str:
    docs_body = _section_body_from_claims(
        snapshot=snapshot,
        payload=payload,
        routes={"docs"},
    )
    local_body = _section_body_from_claims(
        snapshot=snapshot,
        payload=payload,
        routes={"upload", "local"},
    )
    comparison_lines = _ordered_unique_lines([docs_body, local_body])
    if not comparison_lines:
        return ""
    local_route = "upload" if "upload" in snapshot.required_routes else "local"
    comparison_lines.append(_build_hybrid_limit_sentence(local_route))
    return "\n".join(comparison_lines)


def _repair_required_sections(
    *,
    payload: AgentResponsePayloadModel,
    snapshot: ValidationSnapshot,
) -> AgentResponsePayloadModel:
    answer_contract = infer_answer_contract(snapshot.user_input, snapshot.required_routes)
    normalized_sections = normalize_answer_sections(payload.sections)
    if not answer_contract.required_sections:
        normalized_answer = resolve_answer_text(answer=payload.answer, sections=normalized_sections)
        return payload.model_copy(update={"answer": normalized_answer, "sections": normalized_sections})

    existing_by_kind = {section.kind: section for section in normalized_sections}
    repaired_sections: list[AnswerSection] = []
    base_answer = resolve_answer_text(answer=payload.answer, sections=normalized_sections)
    for kind in answer_contract.required_sections:
        section = existing_by_kind.get(kind)
        if section is not None:
            repaired_sections.append(section)
            continue

        body = ""
        if kind == "summary":
            body = base_answer or _section_body_from_claims(
                snapshot=snapshot,
                payload=payload,
            )
        elif kind == "checklist":
            body = _section_body_from_claims(
                snapshot=snapshot,
                payload=payload,
                mode="checklist",
            )
        elif kind == "steps":
            body = _section_body_from_claims(
                snapshot=snapshot,
                payload=payload,
                mode="steps",
            )
        elif kind == "official_docs":
            body = _section_body_from_claims(
                snapshot=snapshot,
                payload=payload,
                routes={"docs"},
            )
        elif kind == "upload_code":
            body = _section_body_from_claims(
                snapshot=snapshot,
                payload=payload,
                routes={"upload", "local"},
            )
        elif kind == "comparison":
            body = _comparison_section_body(snapshot=snapshot, payload=payload)

        body = str(body or "").strip()
        if not body:
            continue
        repaired_sections.append(
            AnswerSection(
                kind=kind,
                heading=_heading_for_section_kind(kind, snapshot=snapshot),
                body=body,
            )
        )

    extra_sections = [
        section
        for section in normalized_sections
        if section.kind not in {item.kind for item in repaired_sections}
    ]
    final_sections = normalize_answer_sections([*repaired_sections, *extra_sections])
    final_answer = resolve_answer_text(answer=payload.answer, sections=final_sections)
    return payload.model_copy(update={"answer": final_answer, "sections": final_sections})


def _route_by_source_id(snapshot: ValidationSnapshot) -> dict[str, str]:
    return {
        str(item.source_id or "").strip(): route_for_tool(str(item.tool or ""))
        for item in snapshot.parsed_evidence
        if str(item.source_id or "").strip()
    }


def _claims_cover_required_routes(
    *,
    claims: list[Any],
    snapshot: ValidationSnapshot,
) -> bool:
    required_routes = {
        str(route or "").strip()
        for route in snapshot.required_routes
        if str(route or "").strip()
    }
    if not required_routes:
        return True

    route_by_source_id = _route_by_source_id(snapshot)
    covered_routes: set[str] = set()
    for claim in claims:
        for evidence_id in getattr(claim, "evidence_ids", []) or []:
            route = route_by_source_id.get(str(evidence_id or "").strip())
            if route:
                covered_routes.add(route)
    return required_routes.issubset(covered_routes)


def _is_hybrid_retrieval_request(snapshot: ValidationSnapshot) -> bool:
    required_routes = {
        str(route or "").strip()
        for route in snapshot.required_routes
        if str(route or "").strip()
    }
    return "docs" in required_routes and bool(required_routes.intersection({"upload", "local"}))


def _ensure_sentence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


def _claim_citation_label(
    *,
    claim: ClaimItem,
    payload: AgentResponsePayloadModel,
) -> str:
    evidence_order = {
        str(item.source_id or "").strip(): index + 1
        for index, item in enumerate(payload.evidence)
        if str(item.source_id or "").strip()
    }
    labels = [
        f"[{evidence_order[evidence_id]}]"
        for evidence_id in claim.evidence_ids
        if evidence_id in evidence_order
    ]
    return " ".join(labels)


def _claim_route(
    *,
    claim: ClaimItem,
    snapshot: ValidationSnapshot,
) -> str:
    route_by_source_id = _route_by_source_id(snapshot)
    for evidence_id in claim.evidence_ids:
        route = route_by_source_id.get(str(evidence_id or "").strip())
        if route:
            return route
    return ""


def _summarize_claim(
    *,
    claim: ClaimItem,
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
) -> str:
    route = _claim_route(claim=claim, snapshot=snapshot)
    cleaned = clean_grounded_text(claim.text)
    if not cleaned:
        return ""
    if route == "docs":
        sentence = f"공식 문서 기준으로 {_ensure_sentence(cleaned)}"
    elif route == "upload":
        sentence = f"업로드 파일에서는 {_ensure_sentence(cleaned)}"
    elif route == "local":
        sentence = f"로컬 자료에서는 {_ensure_sentence(cleaned)}"
    else:
        sentence = _ensure_sentence(cleaned)

    labels = _claim_citation_label(claim=claim, payload=payload)
    return f"{sentence} {labels}".strip()


def _claim_from_evidence_item(
    *,
    evidence_item: Any,
    route: str,
) -> ClaimItem | None:
    source_id = str(getattr(evidence_item, "source_id", "") or "").strip()
    if not source_id:
        return None
    grounded_text = (
        clean_grounded_text(getattr(evidence_item, "snippet", "") or "")
        or clean_grounded_text(getattr(evidence_item, "title", "") or "")
        or clean_grounded_text(getattr(evidence_item, "url_or_path", "") or "")
    )
    if not grounded_text:
        return None
    if route == "docs":
        text = f"공식 문서 기준으로 {_ensure_sentence(grounded_text)}"
    elif route == "upload":
        text = f"업로드 파일에서는 {_ensure_sentence(grounded_text)}"
    else:
        text = f"로컬 자료에서는 {_ensure_sentence(grounded_text)}"
    return ClaimItem(
        text=text,
        evidence_ids=[source_id],
        confidence=normalize_confidence(getattr(evidence_item, "score", None), clamp=True),
    )


def _top_evidence_item_for_routes(
    *,
    snapshot: ValidationSnapshot,
    routes: set[str],
) -> Any | None:
    candidates = [
        item
        for item in snapshot.parsed_evidence
        if route_for_tool(str(item.tool or "")) in routes
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: normalize_confidence(getattr(item, "score", None), clamp=True) or 0.0,
        reverse=True,
    )
    for candidate in candidates:
        if summarize_grounded_text(getattr(candidate, "snippet", "") or ""):
            return candidate
        if summarize_grounded_text(getattr(candidate, "title", "") or ""):
            return candidate
    return None


def _claims_for_routes(
    *,
    claims: list[ClaimItem],
    snapshot: ValidationSnapshot,
    routes: set[str],
) -> list[ClaimItem]:
    return [
        claim
        for claim in claims
        if _claim_route(claim=claim, snapshot=snapshot) in routes
    ]


def _build_hybrid_limit_sentence(local_route: str) -> str:
    if local_route == "upload":
        return "근거는 공식 문서 1건과 업로드 파일 1건만 반영했습니다."
    return "근거는 공식 문서 1건과 로컬 자료 1건만 반영했습니다."


def _normalize_claim_for_route(
    *,
    claim: ClaimItem,
    route: str,
) -> ClaimItem:
    cleaned = clean_grounded_text(claim.text).rstrip(".!?")
    cleaned = re.sub(r"^(?:공식 문서 기준으로(?:는)?|(?:반면\s+)?업로드 파일에서는|(?:반면\s+)?로컬 자료에서는|(?:반면\s+)?업로드 예시에서는)\s+", "", cleaned)
    if route == "docs":
        text = f"공식 문서 기준으로 {cleaned}."
    elif route == "upload":
        text = f"업로드 파일에서는 {cleaned}."
    else:
        text = f"로컬 자료에서는 {cleaned}."
    return ClaimItem(
        text=text,
        evidence_ids=list(claim.evidence_ids),
        confidence=claim.confidence,
    )


def _strip_route_prefix(text: str) -> str:
    return re.sub(
        r"^(?:공식 문서 기준으로(?:는)?|(?:반면\s+)?업로드 파일에서는|(?:반면\s+)?로컬 자료에서는|(?:반면\s+)?업로드 예시에서는)\s+",
        "",
        str(text or "").strip(),
    )


def _rewrite_filtered_hybrid_payload(
    *,
    payload: AgentResponsePayloadModel,
    snapshot: ValidationSnapshot,
) -> AgentResponsePayloadModel:
    if not _is_hybrid_retrieval_request(snapshot):
        return payload

    docs_claim = next(iter(_claims_for_routes(claims=payload.claims, snapshot=snapshot, routes={"docs"})), None)
    local_claim = next(
        iter(_claims_for_routes(claims=payload.claims, snapshot=snapshot, routes={"upload", "local"})),
        None,
    )
    if docs_claim is None:
        official_item = _top_evidence_item_for_routes(snapshot=snapshot, routes={"docs"})
        if official_item is not None:
            docs_claim = _claim_from_evidence_item(evidence_item=official_item, route="docs")
    else:
        docs_claim = _normalize_claim_for_route(claim=docs_claim, route="docs")
    if local_claim is None:
        local_item = _top_evidence_item_for_routes(snapshot=snapshot, routes={"upload", "local"})
        if local_item is not None:
            local_route = route_for_tool(str(local_item.tool or ""))
            local_claim = _claim_from_evidence_item(evidence_item=local_item, route=local_route)
    else:
        local_route = _claim_route(claim=local_claim, snapshot=snapshot) or "upload"
        local_claim = _normalize_claim_for_route(claim=local_claim, route=local_route)

    if docs_claim is not None:
        cleaned = summarize_grounded_text(_strip_route_prefix(docs_claim.text))
        if cleaned:
            docs_claim = ClaimItem(
                text=f"공식 문서 기준으로는 {cleaned}",
                evidence_ids=docs_claim.evidence_ids,
                confidence=docs_claim.confidence,
            )
    if local_claim is not None:
        cleaned = summarize_grounded_text(_strip_route_prefix(local_claim.text))
        if cleaned:
            local_route = _claim_route(claim=local_claim, snapshot=snapshot)
            if local_route == "upload":
                prefix = "반면 업로드 파일에서는"
            else:
                prefix = "반면 로컬 자료에서는"
            local_claim = ClaimItem(
                text=f"{prefix} {cleaned}",
                evidence_ids=local_claim.evidence_ids,
                confidence=local_claim.confidence,
            )

    rebuilt_claims = [claim for claim in (docs_claim, local_claim) if claim is not None]
    if not rebuilt_claims:
        return payload

    rebuilt_payload = render_payload_from_claims(
        claims=rebuilt_claims,
        evidence_items=snapshot.parsed_evidence,
        confidence=average_claim_confidence(rebuilt_claims),
    )
    local_route = _claim_route(claim=local_claim, snapshot=snapshot) if local_claim is not None else "upload"
    rebuilt_payload.answer = f"{rebuilt_payload.answer} {_build_hybrid_limit_sentence(local_route)}".strip()
    rebuilt_payload.confidence = average_claim_confidence(rebuilt_claims)
    return rebuilt_payload


def _build_route_balanced_hybrid_payload(
    snapshot: ValidationSnapshot,
) -> AgentResponsePayloadModel | None:
    if not _is_hybrid_retrieval_request(snapshot):
        return None

    official_item = _top_evidence_item_for_routes(snapshot=snapshot, routes={"docs"})
    local_item = _top_evidence_item_for_routes(snapshot=snapshot, routes={"upload", "local"})
    if official_item is None or local_item is None:
        return None

    return build_deterministic_grounded_payload(
        evidence_items=[official_item, local_item],
        max_claims=2,
        fallback_answer="",
    )


def apply_validation_outcome(
    *,
    snapshot: ValidationSnapshot,
    assessment: ValidationAssessment,
    attempt: int,
    needs_retry: bool,
) -> GraphState:
    updates: GraphState = {}
    retry_reason: RetryReason | None = assessment.retry_reason
    if retry_reason is None or needs_retry:
        return updates

    next_payload: AgentResponsePayloadModel | None = None
    if assessment.has_grounded_response_payload and snapshot.response_payload is not None:
        next_payload = snapshot.response_payload.model_copy(deep=True)
    elif assessment.valid_claims:
        filtered_confidence = average_claim_confidence(assessment.valid_claims)
        next_payload = render_payload_from_claims(
            claims=assessment.valid_claims,
            evidence_items=snapshot.parsed_evidence,
            confidence=filtered_confidence,
        )
        next_payload.confidence = filtered_confidence
        if snapshot.response_payload is not None and snapshot.response_payload.sections:
            next_payload = next_payload.model_copy(update={"sections": snapshot.response_payload.sections})
    elif retry_reason == "missing_sections" and snapshot.response_payload is not None:
        next_payload = snapshot.response_payload.model_copy(deep=True)
    elif snapshot.retrieval_required and snapshot.parsed_evidence:
        docs_valid_claims = _claims_for_routes(
            claims=assessment.valid_claims,
            snapshot=snapshot,
            routes={"docs"},
        )
        local_valid_claims = _claims_for_routes(
            claims=assessment.valid_claims,
            snapshot=snapshot,
            routes={"upload", "local"},
        )
        next_payload = None
        if not docs_valid_claims and not local_valid_claims:
            next_payload = _build_route_balanced_hybrid_payload(snapshot)
        next_payload = next_payload or build_deterministic_grounded_payload(
            evidence_items=snapshot.parsed_evidence,
            fallback_answer="",
        )
    else:
        followup_answer = build_followup_from_routes(snapshot.planner_output, retry_reason)
        updates.update(
            build_followup_updates(
                followup_answer,
                attempt=attempt,
            )
        )
        return updates

    if next_payload is None:
        return updates

    if snapshot.response_payload is not None and snapshot.response_payload.sections and not next_payload.sections:
        next_payload = next_payload.model_copy(update={"sections": snapshot.response_payload.sections})

    if _is_hybrid_retrieval_request(snapshot) and (
        retry_reason in {"unsupported_claims", "missing_route_coverage"}
        or bool(assessment.missing_route_coverage)
    ):
        next_payload = _rewrite_filtered_hybrid_payload(
            payload=next_payload,
            snapshot=snapshot,
        )

    next_payload = _repair_required_sections(
        payload=next_payload,
        snapshot=snapshot,
    )
    updates.update(
        build_response_payload_updates(
            next_payload,
            attempt=attempt,
        )
    )
    return updates
