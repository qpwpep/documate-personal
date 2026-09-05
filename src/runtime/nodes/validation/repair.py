from __future__ import annotations

from src.core.answer_schema.models import AgentResponsePayloadModel, AnswerSection, ClaimItem, normalize_answer_sections
from src.core.answer_schema.rendering import resolve_answer_text
from src.core.answer_schema.text_cleaning import clean_grounded_text
from src.core.contracts.routes import route_for_tool
from src.core.request_contracts import infer_answer_contract
from src.runtime.nodes.validation.evidence_validator import ValidationSnapshot
from src.runtime.nodes.validation.hybrid_rewrite import claim_from_evidence_item, claims_for_routes, top_evidence_item_for_routes
from src.runtime.nodes.validation.messages_ko import hybrid_limit_sentence, section_heading
from src.runtime.nodes.validation.option_literals import contains_option_literal, extract_uploaded_option_literals


def _ensure_sentence(text: str) -> str:
    stripped = str(text or "").strip()
    if not stripped:
        return ""
    if stripped[-1] in ".!?":
        return stripped
    return f"{stripped}."


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


def _summarize_claim(
    *,
    claim: ClaimItem,
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
) -> str:
    cleaned = clean_grounded_text(claim.text)
    if not cleaned:
        return ""

    labels = _claim_citation_label(claim=claim, payload=payload)
    return f"{_ensure_sentence(cleaned)} {labels}".strip()


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
    evidence_item = top_evidence_item_for_routes(snapshot=snapshot, routes=routes)
    if evidence_item is None:
        return ""
    route = route_for_tool(str(evidence_item.tool or "")) or next(iter(routes), "")
    claim = claim_from_evidence_item(evidence_item=evidence_item, route=route)
    if claim is None:
        return ""
    return _summarize_claim(claim=claim, snapshot=snapshot, payload=payload)


def _local_options_text(snapshot: ValidationSnapshot) -> str:
    options = extract_uploaded_option_literals(snapshot.parsed_evidence)
    if not options:
        return ""
    return ", ".join(options[:4])


def _section_body_from_claims(
    *,
    snapshot: ValidationSnapshot,
    payload: AgentResponsePayloadModel,
    routes: set[str] | None = None,
    mode: str = "paragraph",
) -> str:
    selected_claims = payload.claims
    if routes is not None:
        selected_claims = claims_for_routes(claims=payload.claims, snapshot=snapshot, routes=routes)

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
        routes={"upload"},
    )
    local_options_text = _local_options_text(snapshot)
    if local_options_text and not contains_option_literal(local_body, extract_uploaded_option_literals(snapshot.parsed_evidence)):
        local_body = _ordered_unique_lines([local_options_text, local_body])
        local_body = "\n".join(local_body)
    comparison_lines = _ordered_unique_lines(
        [
            f"공식 문서 옵션/기본값: {docs_body}" if docs_body else "",
            f"업로드 코드 실제 설정: {local_body}" if local_body else "",
        ]
    )
    if not comparison_lines:
        return ""
    comparison_lines.append(hybrid_limit_sentence())
    return "\n".join(comparison_lines)


def repair_required_sections(
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
                routes={"upload"},
            )
            options_text = _local_options_text(snapshot)
            if options_text and not contains_option_literal(body, extract_uploaded_option_literals(snapshot.parsed_evidence)):
                body = "\n".join(_ordered_unique_lines([f"업로드 코드의 실제 설정: {options_text}.", body]))
        elif kind == "comparison":
            body = _comparison_section_body(snapshot=snapshot, payload=payload)

        body = str(body or "").strip()
        if not body:
            continue
        repaired_sections.append(
            AnswerSection(
                kind=kind,
                heading=section_heading(kind),
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


__all__ = [
    "repair_required_sections",
]
