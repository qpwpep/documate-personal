from __future__ import annotations

import time
from typing import Any

from src.core.answer_schema import build_grounded_response, finalize_answer, text_document
from src.core.contracts import GraphState
from src.core.latency import elapsed_ms, make_stage_latency_event
from src.core.request_contracts import check_answer_contract
from src.runtime.nodes.synthesis.models import SynthesisContext
from src.runtime.nodes.synthesis.state import build_synthesis_updates


def maybe_short_circuit_synthesis(
    *, state: GraphState, debug: Any, context: SynthesisContext, stage_started: float,
) -> GraphState | None:
    packet = []
    kind = "draft"
    contract = context.request_contract
    if contract is None or contract.failure is not None:
        result = finalize_answer(text_document("요청의 조건을 확정하지 못했습니다. 다시 요청해 주세요."), [])
        mode = "request_contract_unavailable"
        kind = "failure"
    elif context.guided_followup and context.planner_blocked:
        result = finalize_answer(text_document(context.guided_followup), [])
        mode = "guided_followup"
        kind = "clarification"
    elif contract.can_cancel_pending():
        result = finalize_answer(text_document("보류 중인 전달 요청을 취소했습니다."), [])
        mode = "pending_cancelled"
        kind = "clarification"
    elif contract.can_acknowledge():
        save_forbidden = contract.actions.save_text.intent == "forbidden"
        slack_forbidden = contract.actions.slack_notify.intent == "forbidden"
        confirmation = (
            "알겠습니다. 파일로 저장하거나 Slack으로 전송하지 않겠습니다." if save_forbidden and slack_forbidden else
            "알겠습니다. 파일로 저장하지 않겠습니다." if save_forbidden else
            "알겠습니다. Slack으로 전송하지 않겠습니다." if slack_forbidden else "알겠습니다."
        )
        result = finalize_answer(text_document(confirmation), [])
        mode = "request_acknowledged"
        kind = "clarification"
    elif not contract.can_prepare_body():
        question = next((item.question for item in contract.missing_info if item.slot in {
            "subject", "input_reference", "answer_reference", "pending_request",
        }), None)
        if not question and contract.body.kind == "unresolved":
            question = contract.body.question
        if not question and contract.missing_info:
            question = contract.missing_info[0].question
        result = finalize_answer(text_document(question or "요청의 조건을 확정하지 못했습니다. 다시 요청해 주세요."), [])
        mode = "body_request_unresolved"
        kind = "clarification"
    elif contract.body.kind in {"copy_answer", "transform_answer"} and context.source_response is None:
        result = finalize_answer(text_document("전달할 원래 답변을 확정하지 못했습니다. 사용할 본문을 지정해 주세요."), [])
        mode = "body_reference_unresolved"
        kind = "clarification"
    elif contract.body.kind == "copy_input":
        result = finalize_answer(text_document(contract.body.source.text), [])
        if not check_answer_contract(contract.answer, result.content).valid:
            result = finalize_answer(text_document("선택한 원문이 요청의 필수·금지 조건을 충족하지 않아 그대로 사용할 수 없습니다."), [])
            kind = "failure"
            mode = "copy_input_contract_conflict"
        else:
            mode = "copy_bound_input"
    elif contract.body.kind == "copy_answer":
        previous = context.source_response
        if not check_answer_contract(
            contract.answer, previous.content, evidence=[citation.evidence for citation in previous.citations],
        ).valid:
            result = finalize_answer(text_document("원래 답변이 이번 요청의 필수·금지 조건을 충족하지 않아 그대로 사용할 수 없습니다."), [])
            mode = "reuse_contract_conflict"
            kind = "failure"
        else:
            packet = [citation.evidence for citation in previous.citations]
            result = finalize_answer(previous.content, packet, retrieval_required=previous.retrieval_required, issues=previous.issues)
            mode = "reuse_bound_answer"
    elif (
        context.retrieval_required and len(context.hits) == 1
        and len(context.planner_output.tasks) == 1
        and context.hits[0].evidence.route == "upload"
        and contract.body.kind == "extract"
        and any(d.requirement_id == context.hits[0].requirement_id and d.answerability == "covered"
                for d in debug.retrieval_diagnostics)
    ):
        packet = [context.hits[0].evidence]
        result = build_grounded_response(packet, message="요청한 자료의 원문 발췌입니다.")
        check = check_answer_contract(contract.answer, result.content, evidence=packet)
        if not check.valid or check.unchecked_semantic:
            return None
        result = finalize_answer(result.content, packet, retrieval_required=True)
        mode = "deterministic_grounded_direct"
    else:
        return None
    return build_synthesis_updates(
        debug=debug, result=result, evidence_packet=packet, attempt=context.attempt,
        latency_trace=[make_stage_latency_event(
            stage="synthesis", attempt=context.attempt,
            latency_ms=elapsed_ms(stage_started, time.perf_counter()), status=mode,
        )],
        retrieval_errors=context.parse_errors,
        planner_errors=context.planner_parse_errors,
        kind=kind,
        request_id=contract.request_id if contract else None,
        contract_revision=contract.revision if contract else 0,
        body_kind=contract.body.kind if contract else "unresolved",
        evidence_source=context.evidence_source,
        evidence_requirement_map={item.id: [hit.requirement_id for hit in context.hits if hit.evidence.id == item.id and hit.requirement_id]
                                  for item in packet},
    )
