from __future__ import annotations

import logging

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.response import get_response_state
from src.infra.logging_utils import log_event
from src.runtime.nodes.retry import build_retry_update
from src.runtime.nodes.validation.evidence_validator import ValidationAssessment, ValidationSnapshot, assess_validation, collect_validation_snapshot
from src.runtime.nodes.validation.policy import apply_validation_outcome


logger = logging.getLogger(__name__)


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


def make_post_synthesis_validation_node(verbose: bool):
    def post_synthesis_validation(state: GraphState) -> GraphState:
        response = get_response_state(state)
        debug = get_debug_state(state)
        retry_context = get_retry_state(state)

        snapshot, local_errors = collect_validation_snapshot(state)
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
                "post_synthesis_validation",
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
            error_codes = list(debug.error_codes)
            for code in assessment.error_codes:
                if code not in error_codes:
                    error_codes.append(code)
            updates["debug"] = debug.model_copy(
                update={
                    "validation_errors": [*debug.validation_errors, *local_errors],
                    "validation_events": [*debug.validation_events, *local_errors],
                    "error_codes": error_codes,
                }
            )
        return updates

    return post_synthesis_validation
