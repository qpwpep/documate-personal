from __future__ import annotations

from langchain_core.messages import AIMessage

from ..evidence import parse_evidence_payload
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
            return {
                "needs_retry": needs_retry,
                "retry_context": next_retry_context,
                "messages": [AIMessage(content=guided_followup)],
                "final_answer": guided_followup,
            }

        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        retrieval_error_start_index = int(retry_context.get("retrieval_error_start_index", 0))
        retrieval_diagnostic_start_index = int(
            retry_context.get("retrieval_diagnostic_start_index", 0)
        )

        current_attempt_evidence_payload = slice_from_index(
            safe_list(state.get("retrieved_evidence")),
            evidence_start_index,
        )
        parsed_evidence = parse_evidence_payload(
            current_attempt_evidence_payload,
            context="retrieved_evidence",
            errors=parse_errors,
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

        retry_reason: RetryReason | None = None
        if blocked_missing_upload:
            retry_reason = "blocked_missing_upload"
        elif tool_error:
            retry_reason = "tool_error"
        elif retrieval_required and not has_valid_evidence:
            retry_reason = "no_evidence"
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
            print(
                f"[validate_evidence] required={retrieval_required} "
                f"evidence={len(parsed_evidence)} retry={needs_retry} reason={retry_reason}"
            )

        updates: State = {
            "needs_retry": needs_retry,
            "retry_context": next_retry_context,
        }
        if retry_reason is not None and not needs_retry:
            followup_answer = build_followup_from_routes(planner_output, retry_reason)
            updates["messages"] = [AIMessage(content=followup_answer)]
            updates["final_answer"] = followup_answer
        if local_errors:
            updates["validation_errors"] = local_errors
        return updates

    return validate_evidence
