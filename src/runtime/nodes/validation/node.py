from __future__ import annotations

import json
import logging
from dataclasses import replace

from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.response import get_response_state
from src.infra.logging_utils import log_event
from src.runtime.nodes.retry import build_retry_update
from src.runtime.nodes.validation.assessment import assess_validation
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.policy import add_retrieval_gap_issues, apply_validation_outcome
from src.runtime.nodes.validation.snapshot import collect_validation_snapshot, detect_packet_coverage_gaps
from src.runtime.nodes.validation.semantic import DecisionEvaluator, assess_post_semantics


logger = logging.getLogger(__name__)


def _deterministic_issues(assessment: ValidationAssessment) -> dict[str, list[str]]:
    """Return server-owned check codes without source text or model-provided refs."""
    if assessment.checked_result is None:
        return {}
    return {
        check.unit_id: list(dict.fromkeys(issue.split(":", 1)[0] for issue in check.issues))
        for check in assessment.checked_result.checks
        if check.unit_id in assessment.invalid_unit_paths and check.issues
    }


def _decide_retry_outcome(
    *,
    snapshot: ValidationSnapshot,
    assessment: ValidationAssessment,
    state: GraphState,
) -> tuple[bool, object, str, list[str]]:
    retry_context = get_retry_state(state)
    retrieved_missing, packet_omitted = detect_packet_coverage_gaps(snapshot)
    # A compact omission can still recover using a complete normal packet.
    terminal_missing = set(snapshot.terminal_failed_requirement_ids)
    repairable_gaps = (set(retrieved_missing) | set(packet_omitted)) - terminal_missing
    normal_missing = set(snapshot.normal_evidence_missing_requirement_ids) - terminal_missing
    content_repair_available = not (repairable_gaps and normal_missing)
    needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
        retry_context=retry_context,
        retry_reason=assessment.retry_reason,
        planner_output=snapshot.planner_output,
        score_avg=assessment.score_avg,
        failed_routes=assessment.failed_routes,
        failed_requirement_ids=assessment.failed_requirement_ids,
        current_attempt_hits=snapshot.parsed_hits,
        current_attempt_retrieval_diagnostics=snapshot.current_attempt_retrieval_diagnostics,
        request_contract=snapshot.request_contract,
        content_repair_available=content_repair_available,
    )
    if retrieved_missing or packet_omitted:
        retrieval_feedback += (
            f" Missing from retrieved source ranges: {retrieved_missing}."
            f" Omitted from the evidence packet by range/item/text limits: {packet_omitted}."
        )
        if not content_repair_available:
            retrieval_feedback += " The normal evidence packet also lacks required coverage; repeating generation cannot restore it."
        next_retry_context = next_retry_context.model_copy(update={"retrieval_feedback": retrieval_feedback})
    if assessment.missing_content or assessment.forbidden_content:
        retrieval_feedback += (
            f" Missing required: {assessment.missing_content}."
            f" Forbidden present: {assessment.forbidden_content}."
            " Keep the confirmed request contract unchanged when repairing the body."
        )
        failed_conditions = [
            {"path": f"answer.content[{index}]",
             **requirement.model_dump(include={"kind", "mode", "description"})}
            for index, requirement in enumerate(snapshot.request_contract.answer.content)
            if f"semantic_content_condition:{index}" in assessment.missing_content
        ] if snapshot.request_contract is not None else []
        if failed_conditions:
            retrieval_feedback += (
                " Failed confirmed content conditions: "
                + json.dumps(failed_conditions, ensure_ascii=False) + "."
            )
        failed_sentence_counts = [
            {"scope_id": index, **requirement.model_dump(exclude={"evidence_ids"})}
            for index, requirement in enumerate(snapshot.request_contract.answer.sentence_counts, 1)
            if "sentence_scope" in assessment.missing_content
            or f"sentence_count:{index}" in assessment.missing_content
        ] if snapshot.request_contract is not None else []
        if failed_sentence_counts:
            retrieval_feedback += (
                " Failed confirmed sentence conditions: "
                + json.dumps(failed_sentence_counts, ensure_ascii=False) + "."
            )
        next_retry_context = next_retry_context.model_copy(update={"retrieval_feedback": retrieval_feedback})
    if "inline_code" in assessment.missing_content or "semantic_format_fulfillment" in assessment.missing_content:
        retrieval_feedback += (
            " Check the requested formatting on every intended portion of the actual body."
            " For code inside a sentence, preserve the literal code in Markdown code spans within the same"
            " paragraph content unit; standalone blocks and plain text do not satisfy that layout."
        )
        next_retry_context = next_retry_context.model_copy(update={"retrieval_feedback": retrieval_feedback})
    if assessment.invalid_unit_paths:
        deterministic_issues = _deterministic_issues(assessment)
        retrieval_feedback += (
            f" Unsupported displayed units: {sorted(assessment.invalid_unit_paths)}."
            " Recheck each claim against only its own cited source excerpts; remove or repair unsupported claims."
        )
        if deterministic_issues:
            retrieval_feedback += (
                " Deterministic check issues by displayed unit: "
                + json.dumps(deterministic_issues, sort_keys=True)
                + "."
            )
        issue_codes = {issue for issues in deterministic_issues.values() for issue in issues}
        if "excerpt_does_not_match_source" in issue_codes:
            retrieval_feedback += (
                " For a partial quotation, preserve the supported literal text and use basis=source with its source ref."
                " basis=excerpt requires text identical to the entire selected excerpt of exactly one referenced source,"
                " including whitespace; a one-line quotation from a longer selection is not an exact excerpt."
            )
        if any(issue.startswith("source_location_") for issue in issue_codes):
            retrieval_feedback += (
                f" Use a separate location-only unit with text={SOURCE_LOCATION_PLACEHOLDER!r}."
                " Preserve requested literal code and explanations in separate units with source_location=null."
                " Bind source_location.ref to its only refs entry and select element-character start/end offsets"
                " inside that evidence selection using source_lines. The server generates physical line numbers."
                " Do not put line numbers in free prose, count excerpt lines, or guess a source range."
                " If the source has no exact mapping, remove the unsupported location claim and state that limitation."
            )
        next_retry_context = next_retry_context.model_copy(update={"retrieval_feedback": retrieval_feedback})

    local_errors: list[str] = []
    if assessment.retry_reason is not None:
        local_errors.append(
            "validate_evidence: retry_reason="
            f"{assessment.retry_reason}, failed_routes={sorted(assessment.failed_routes)}, "
            f"score_avg={assessment.score_avg}, feedback={retrieval_feedback}"
        )
    return needs_retry, next_retry_context, retrieval_feedback, local_errors


def make_post_synthesis_validation_node(
    verbose: bool, *, decisions_client: DecisionEvaluator, pass_threshold: float = 0.8,
    max_turns: int = 6,
):
    def post_synthesis_validation(state: GraphState) -> GraphState:
        response = get_response_state(state)
        debug = get_debug_state(state)
        retry_context = get_retry_state(state)

        if response.problem is not None:
            if retry_context.terminal_failed_requirement_ids:
                snapshot, _local_errors = collect_validation_snapshot(state, max_turns=max_turns)
                response = response.model_copy(update={
                    "result": add_retrieval_gap_issues(snapshot, response.result),
                })
            # A grounded partial response preserves its technical failure and cannot
            # trigger another semantic repair window or become an executable answer.
            return {
                "response": response,
                "retry": retry_context.model_copy(update={
                    "needs_retry": False, "retry_reason": None, "failed_routes": [],
                    "failed_requirement_ids": [], "retrieval_feedback": "",
                }),
            }

        snapshot, local_errors = collect_validation_snapshot(state, max_turns=max_turns)
        assessment = assess_validation(snapshot)
        assess_post_semantics(
            snapshot, assessment, decisions_client=decisions_client,
            threshold=pass_threshold, attempt=max(1, response.synthesis_attempt),
        )
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
                evidence_count=len(snapshot.parsed_hits),
                needs_retry=needs_retry,
                retry_reason=assessment.retry_reason,
                missing_content=assessment.missing_content,
                deterministic_issues=_deterministic_issues(assessment),
                semantic_failures={
                    name: round(value, 4) for name, value in assessment.semantic_probabilities.items()
                    if value < pass_threshold
                },
            )

        _ = retrieval_feedback
        updates: GraphState = {
            "retry": next_retry_context,
        }
        outcome = apply_validation_outcome(
            snapshot=snapshot,
            assessment=assessment,
            attempt=response.synthesis_attempt,
            needs_retry=needs_retry,
        )
        updates.update(outcome.updates)
        semantic_assessments = [("generated", assessment)]
        final_response = outcome.response_to_validate
        if final_response is not None:
            # Filtering changes the displayed body and invalidates its previous
            # semantic verdict. Recheck that exact revision before promotion.
            revised_snapshot = replace(
                snapshot, response_result=final_response.result,
                evidence_packet=final_response.evidence_packet,
                evidence_requirement_map=final_response.evidence_requirement_map,
                decision_cache=assessment.decision_cache or snapshot.decision_cache,
            )
            revised_assessment = assess_validation(revised_snapshot)
            assess_post_semantics(
                revised_snapshot, revised_assessment, decisions_client=decisions_client,
                threshold=pass_threshold, attempt=max(1, response.synthesis_attempt),
            )
            semantic_assessments.append(("filtered", revised_assessment))
            # A second rejection must remove the newly rejected claims as well.
            # Stop bounded recovery with an explicit failure;
            # do not publish an unreviewed third revision as a validated answer.
            revised_outcome = apply_validation_outcome(
                snapshot=revised_snapshot, assessment=revised_assessment,
                attempt=response.synthesis_attempt, needs_retry=False, allow_filtered_answer=False,
            )
            if revised_assessment.retry_reason is None and final_response.problem is not None:
                # Validation of the retained body cannot resolve the original
                # source-location or retrieval failure that made it partial.
                revised_outcome.updates["response"] = revised_outcome.updates["response"].model_copy(update={
                    "kind": final_response.kind, "problem": final_response.problem,
                })
            updates.update(revised_outcome.updates)
        final_assessment = semantic_assessments[-1][1]
        if final_assessment.decision_cache:
            updates["retrieval"] = get_retrieval_state(state).model_copy(update={
                "decision_cache": final_assessment.decision_cache,
            })
        semantic_events = []
        for revision, semantic_assessment in semantic_assessments:
            if not semantic_assessment.semantic_probabilities and not semantic_assessment.semantic_refusals:
                continue
            failures = {name: round(value, 4) for name, value in semantic_assessment.semantic_probabilities.items()
                        if value < pass_threshold}
            semantic_events.append(
                f"post_synthesis_validation: decisions revision={revision}, threshold={pass_threshold}, "
                f"predicates={len(semantic_assessment.semantic_probabilities) + len(semantic_assessment.semantic_refusals)}, "
                f"failed={failures}, refused={sorted(semantic_assessment.semantic_refusals)}"
            )
        if local_errors or semantic_events:
            error_codes = list(debug.error_codes)
            for code in assessment.error_codes:
                if code not in error_codes:
                    error_codes.append(code)
            updates["debug"] = debug.model_copy(
                update={
                    "validation_errors": [*debug.validation_errors, *local_errors],
                    "validation_events": [*debug.validation_events, *semantic_events, *local_errors],
                    "error_codes": error_codes,
                }
            )
        return updates

    return post_synthesis_validation
