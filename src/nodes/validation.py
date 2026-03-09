from __future__ import annotations

import logging
from langchain_core.messages import AIMessage

from ..answer_schema import (
    AgentResponsePayloadModel,
    average_claim_confidence,
    build_empty_response_payload,
    filter_claims_by_evidence,
    render_payload_from_claims,
)
from ..evidence import EvidenceItem, parse_evidence_payload
from ..logging_utils import log_event
from .retry import build_followup_from_routes, build_retry_update, contains_tool_error
from .state import (
    LOW_SCORE_THRESHOLD,
    RetryReason,
    State,
    coerce_planner_output,
    coerce_retry_context,
    safe_list,
    slice_from_index,
)


logger = logging.getLogger(__name__)


def _coerce_response_payload(raw_payload: object) -> AgentResponsePayloadModel | None:
    if raw_payload is None:
        return None
    try:
        return AgentResponsePayloadModel.model_validate(raw_payload)
    except Exception:
        return None


def _payload_to_state_dict(payload: AgentResponsePayloadModel) -> dict[str, object]:
    return payload.model_dump(mode="json")


def _build_followup_updates(answer: str) -> State:
    payload = build_empty_response_payload(answer=answer)
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "response_payload": _payload_to_state_dict(payload),
    }


def _coerce_evidence_list(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if isinstance(item, EvidenceItem)]


def make_validate_evidence_node(verbose: bool):
    def validate_evidence(state: State) -> State:
        local_errors: list[str] = []
        parse_errors: list[str] = []

        planner_output = coerce_planner_output(state.get("planner_output"), local_errors)
        retry_context = coerce_retry_context(state.get("retry_context"))
        guided_followup = str(state.get("guided_followup") or "").strip()
        if guided_followup:
            needs_retry, next_retry_context, _ = build_retry_update(
                retry_context=retry_context,
                retry_reason="blocked_missing_upload",
                planner_output=planner_output,
                retrieval_errors=[],
                score_avg=None,
            )
            updates: State = {
                "needs_retry": needs_retry,
                "retry_context": next_retry_context,
            }
            updates.update(_build_followup_updates(guided_followup))
            return updates

        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        retrieval_error_start_index = int(retry_context.get("retrieval_error_start_index", 0))
        retrieval_diagnostic_start_index = int(
            retry_context.get("retrieval_diagnostic_start_index", 0)
        )

        current_attempt_evidence_payload = slice_from_index(
            safe_list(state.get("retrieved_evidence")),
            evidence_start_index,
        )
        parsed_evidence = _coerce_evidence_list(
            parse_evidence_payload(
                current_attempt_evidence_payload,
                context="retrieved_evidence",
                errors=parse_errors,
            )
        )
        local_errors.extend(parse_errors)

        current_attempt_retrieval_errors = [
            str(error)
            for error in slice_from_index(
                safe_list(state.get("retrieval_errors")),
                retrieval_error_start_index,
            )
            if str(error).strip()
        ]
        current_attempt_retrieval_diagnostics = [
            item
            for item in slice_from_index(
                safe_list(state.get("retrieval_diagnostics")),
                retrieval_diagnostic_start_index,
            )
            if isinstance(item, dict)
        ]

        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
        has_valid_evidence = len(parsed_evidence) > 0
        response_payload = _coerce_response_payload(state.get("response_payload"))

        score_values = [float(item.score) for item in parsed_evidence if item.score is not None]
        score_avg = (sum(score_values) / len(score_values)) if score_values else None
        low_score = bool(
            retrieval_required
            and score_avg is not None
            and score_avg < LOW_SCORE_THRESHOLD
        )
        blocked_missing_upload = bool(
            retrieval_required
            and any(
                str(item.get("route") or "") == "upload"
                and str(item.get("status") or "") == "unavailable"
                for item in current_attempt_retrieval_diagnostics
            )
        )
        tool_error = bool(
            retrieval_required
            and not blocked_missing_upload
            and (
                any(str(item.get("status") or "") == "error" for item in current_attempt_retrieval_diagnostics)
                or contains_tool_error(current_attempt_retrieval_errors)
                or contains_tool_error(parse_errors)
            )
        )

        valid_claims = []
        invalid_claims = []
        if retrieval_required and response_payload is not None:
            valid_claims, invalid_claims = filter_claims_by_evidence(
                claims=response_payload.claims,
                evidence_items=parsed_evidence,
            )

        unsupported_claims = bool(
            retrieval_required
            and response_payload is not None
            and (
                (response_payload.answer.strip() and not response_payload.claims)
                or bool(invalid_claims)
            )
        )

        retry_reason: RetryReason | None = None
        if blocked_missing_upload:
            retry_reason = "blocked_missing_upload"
        elif tool_error:
            retry_reason = "tool_error"
        elif retrieval_required and not has_valid_evidence:
            retry_reason = "no_evidence"
        elif unsupported_claims:
            retry_reason = "unsupported_claims"
        elif low_score:
            retry_reason = "low_score"

        needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
            retry_context=retry_context,
            retry_reason=retry_reason,
            planner_output=planner_output,
            retrieval_errors=current_attempt_retrieval_errors + parse_errors,
            score_avg=score_avg,
        )

        if retry_reason is not None:
            local_errors.append(
                "validate_evidence: retry_reason="
                f"{retry_reason}, score_avg={score_avg}, feedback={retrieval_feedback}"
            )

        if verbose:
            log_event(
                logger,
                logging.INFO,
                "validate_evidence",
                retrieval_required=retrieval_required,
                evidence_count=len(parsed_evidence),
                needs_retry=needs_retry,
                retry_reason=retry_reason,
            )

        updates: State = {
            "needs_retry": needs_retry,
            "retry_context": next_retry_context,
        }
        if retry_reason is not None and not needs_retry:
            if retry_reason == "unsupported_claims" and valid_claims:
                filtered_confidence = average_claim_confidence(valid_claims)
                filtered_payload = render_payload_from_claims(
                    claims=valid_claims,
                    evidence_items=parsed_evidence,
                    confidence=filtered_confidence,
                )
                filtered_payload.confidence = filtered_confidence
                updates["messages"] = [AIMessage(content=filtered_payload.answer)]
                updates["final_answer"] = filtered_payload.answer
                updates["response_payload"] = _payload_to_state_dict(filtered_payload)
            else:
                followup_answer = build_followup_from_routes(planner_output, retry_reason)
                updates.update(_build_followup_updates(followup_answer))
        if local_errors:
            updates["validation_errors"] = local_errors
        return updates

    return validate_evidence
