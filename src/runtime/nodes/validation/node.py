from __future__ import annotations

import logging

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.infra.logging_utils import log_event
from src.core.evidence import parse_evidence_payload
from src.core.sequence_utils import slice_from_index
from src.runtime.nodes.retry import build_retry_update
from src.runtime.nodes.validation.evidence_validator import ValidationAssessment, ValidationSnapshot, assess_validation, build_validation_snapshot, coerce_evidence_list
from src.runtime.nodes.validation.policy import apply_validation_outcome, build_followup_updates


logger = logging.getLogger(__name__)


def _collect_validation_snapshot(state: GraphState) -> tuple[ValidationSnapshot, list[str]]:
    local_errors: list[str] = []
    parse_errors: list[str] = []
    runtime = get_runtime_state(state)
    planner = get_planner_state(state)
    retrieval = get_retrieval_state(state)
    response = get_response_state(state)
    debug = get_debug_state(state)
    retry_context = get_retry_state(state)

    planner_output = parse_planner_output(planner.output, local_errors)
    evidence_start_index = int(retry_context.evidence_start_index)
    retrieval_error_start_index = int(retry_context.retrieval_error_start_index)
    retrieval_diagnostic_start_index = int(retry_context.retrieval_diagnostic_start_index)

    current_attempt_evidence_payload = slice_from_index(
        retrieval.evidence_log,
        evidence_start_index,
    )
    parsed_evidence = coerce_evidence_list(
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
            debug.retrieval_errors,
            retrieval_error_start_index,
        )
        if str(error).strip()
    ]
    current_attempt_retrieval_diagnostics = [
        item
        for item in slice_from_index(
            debug.retrieval_diagnostics,
            retrieval_diagnostic_start_index,
        )
        if item is not None
    ]

    snapshot = build_validation_snapshot(
        user_input=runtime.user_input,
        planner_output=planner_output,
        parsed_evidence=parsed_evidence,
        current_attempt_retrieval_errors=[*current_attempt_retrieval_errors, *parse_errors],
        current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics,
        response_payload=response.payload,
    )
    return snapshot, local_errors


def _assess_route_failures(snapshot: ValidationSnapshot) -> ValidationAssessment:
    return assess_validation(snapshot)


def _decide_retry_outcome(
    *,
    snapshot: ValidationSnapshot,
    assessment: ValidationAssessment,
    state: GraphState,
) -> tuple[bool, object, str, list[str]]:
    retry_context = get_retry_state(state)
    needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
        retry_context=retry_context,
        retry_reason=assessment.retry_reason,
        planner_output=snapshot.planner_output,
        retrieval_errors=snapshot.current_attempt_retrieval_errors,
        score_avg=assessment.score_avg,
        failed_routes=assessment.failed_routes,
        current_attempt_evidence=snapshot.parsed_evidence,
        current_attempt_retrieval_diagnostics=snapshot.current_attempt_retrieval_diagnostics,
    )

    local_errors: list[str] = []
    if assessment.retry_reason is not None:
        local_errors.append(
            "validate_evidence: retry_reason="
            f"{assessment.retry_reason}, failed_routes={sorted(assessment.failed_routes)}, "
            f"score_avg={assessment.score_avg}, feedback={retrieval_feedback}"
        )
    return needs_retry, next_retry_context, retrieval_feedback, local_errors


def _apply_validation_outcome(
    *,
    snapshot: ValidationSnapshot,
    assessment: ValidationAssessment,
    attempt: int,
    needs_retry: bool,
) -> GraphState:
    return apply_validation_outcome(
        snapshot=snapshot,
        assessment=assessment,
        attempt=attempt,
        needs_retry=needs_retry,
    )


def make_validate_evidence_node(verbose: bool):
    def validate_evidence(state: GraphState) -> GraphState:
        planner = get_planner_state(state)
        response = get_response_state(state)
        debug = get_debug_state(state)
        retry_context = get_retry_state(state)
        guided_followup = str(planner.guided_followup or "").strip()

        planner_output = parse_planner_output(planner.output, [])
        if guided_followup:
            needs_retry, next_retry_context, _ = build_retry_update(
                retry_context=retry_context,
                retry_reason="blocked_missing_upload",
                planner_output=planner_output,
                retrieval_errors=[],
                score_avg=None,
            )
            _ = needs_retry
            updates: GraphState = {
                "retry": next_retry_context,
            }
            updates.update(build_followup_updates(guided_followup, attempt=response.synthesis_attempt))
            return updates

        snapshot, local_errors = _collect_validation_snapshot(state)
        assessment = _assess_route_failures(snapshot)
        needs_retry, next_retry_context, retrieval_feedback, retry_errors = _decide_retry_outcome(
            snapshot=snapshot,
            assessment=assessment,
            state=state,
        )
        local_errors.extend(retry_errors)

        if verbose:
            log_event(
                logger,
                logging.INFO,
                "validate_evidence",
                retrieval_required=snapshot.retrieval_required,
                evidence_count=len(snapshot.parsed_evidence),
                needs_retry=needs_retry,
                retry_reason=assessment.retry_reason,
            )

        _ = retrieval_feedback
        updates: GraphState = {
            "retry": next_retry_context,
        }
        updates.update(
            _apply_validation_outcome(
                snapshot=snapshot,
                assessment=assessment,
                attempt=response.synthesis_attempt,
                needs_retry=needs_retry,
            )
        )
        if local_errors:
            updates["debug"] = debug.model_copy(
                update={"validation_errors": [*debug.validation_errors, *local_errors]}
            )
        return updates

    return validate_evidence
