from __future__ import annotations

import logging

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.response import get_response_state
from src.infra.logging_utils import log_event
from src.runtime.nodes.retry import build_followup_from_routes, build_retry_update
from src.runtime.nodes.validation.evidence_validator import assess_retrieval_quality, collect_validation_snapshot
from src.runtime.nodes.validation.policy import build_followup_updates


logger = logging.getLogger(__name__)


def make_pre_synthesis_validation_node(verbose: bool):
    def pre_synthesis_validation(state: GraphState) -> GraphState:
        planner = get_planner_state(state)
        response = get_response_state(state)
        debug = get_debug_state(state)
        retry_context = get_retry_state(state)
        guided_followup = str(planner.guided_followup or "").strip()

        if guided_followup:
            planner_output = parse_planner_output(planner.output, [])
            needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
                retry_context=retry_context,
                retry_reason="blocked_missing_upload",
                planner_output=planner_output,
                retrieval_errors=[],
                score_avg=None,
                failed_routes={"upload"},
            )
            _ = needs_retry
            updates: GraphState = {
                "retry": next_retry_context,
            }
            updates.update(build_followup_updates(guided_followup, attempt=response.synthesis_attempt))
            updates["debug"] = debug.model_copy(
                update={
                    "validation_errors": [
                        *debug.validation_errors,
                        "pre_synthesis_validation: retry_reason=blocked_missing_upload, "
                        f"failed_routes=['upload'], score_avg=None, feedback={retrieval_feedback}",
                    ],
                    "validation_events": [
                        *debug.validation_events,
                        "pre_synthesis_validation: retry_reason=blocked_missing_upload, "
                        f"failed_routes=['upload'], score_avg=None, feedback={retrieval_feedback}",
                    ],
                }
            )
            return updates

        snapshot, local_errors = collect_validation_snapshot(state)
        assessment = assess_retrieval_quality(snapshot)
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

        if assessment.retry_reason is not None:
            local_errors.append(
                "pre_synthesis_validation: retry_reason="
                f"{assessment.retry_reason}, failed_routes={sorted(assessment.failed_routes)}, "
                f"score_avg={assessment.score_avg}, feedback={retrieval_feedback}"
            )

        if verbose:
            log_event(
                logger,
                logging.INFO,
                "pre_synthesis_validation",
                retrieval_required=snapshot.retrieval_required,
                evidence_count=len(snapshot.parsed_evidence),
                needs_retry=needs_retry,
                retry_reason=assessment.retry_reason,
            )

        updates: GraphState = {
            "retry": next_retry_context,
        }
        if assessment.retry_reason is not None and not needs_retry:
            followup_answer = build_followup_from_routes(
                snapshot.planner_output,
                assessment.retry_reason,
            )
            updates.update(
                build_followup_updates(
                    followup_answer,
                    attempt=response.synthesis_attempt,
                )
            )
        if local_errors:
            updates["debug"] = debug.model_copy(
                update={
                    "validation_errors": [*debug.validation_errors, *local_errors],
                    "validation_events": [*debug.validation_events, *local_errors],
                }
            )
        return updates

    return pre_synthesis_validation
