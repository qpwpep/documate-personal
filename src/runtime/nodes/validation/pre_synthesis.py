from __future__ import annotations

import logging
import json
from dataclasses import replace

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.llm_errors import LLMCallError, make_problem
from src.infra.logging_utils import log_event
from src.runtime.nodes.retry import build_followup_from_routes, build_retry_update, retrieval_subject
from src.runtime.nodes.validation.evidence_validator import assess_retrieval_quality, collect_validation_snapshot
from src.runtime.nodes.validation.models import ValidationAssessment
from src.runtime.nodes.validation.policy import build_followup_updates
from src.runtime.nodes.validation.semantic import DecisionEvaluator, assess_pre_semantics
from src.runtime.nodes.synthesis.evidence_selection import tasks_for_hit, upload_file_id


logger = logging.getLogger(__name__)


def make_pre_synthesis_validation_node(
    verbose: bool, *, decisions_client: DecisionEvaluator, pass_threshold: float = 0.8,
    max_turns: int = 6,
):
    def pre_synthesis_validation(state: GraphState) -> GraphState:
        planner = get_planner_state(state)
        response = get_response_state(state)
        debug = get_debug_state(state)
        retry_context = get_retry_state(state)
        guided_followup = str(planner.guided_followup or "").strip()
        runtime = get_runtime_state(state)
        contract = runtime.request_contract
        if contract is None or contract.failure is not None or planner.diagnostics.reason == "planner_unavailable":
            raise LLMCallError(make_problem("internal_error", "pre_synthesis_validation"))
        if contract.can_acknowledge() or contract.can_cancel_pending():
            # These server-resolved instructions create no deliverable body.
            # Preserve their deterministic acknowledgment/cancellation path.
            return {"retry": retry_context.model_copy(update={
                "needs_retry": False, "retry_reason": None, "failed_routes": [],
                "failed_requirement_ids": [], "retrieval_feedback": "",
            })}
        stamp = {"request_id": contract.request_id,
                 "contract_revision": contract.revision,
                 "body_kind": contract.body.kind,
                 "evidence_source": None}
        if (contract.can_prepare_body()
                and planner.diagnostics.reason != "upload_retriever_missing"):
            guided_followup = ""

        if guided_followup:
            if planner.diagnostics.reason != "upload_retriever_missing":
                updates = {"retry": retry_context.model_copy(update={
                    "needs_retry": False, "retry_reason": None, "failed_routes": [],
                    "failed_requirement_ids": [], "retrieval_feedback": "",
                })}
                updates.update(build_followup_updates(guided_followup, attempt=response.synthesis_attempt, **stamp))
                return updates
            needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
                retry_context=retry_context,
                retry_reason="blocked_missing_upload",
                planner_output=planner.output,
                score_avg=None,
                failed_routes={"upload"},
                request_contract=contract,
            )
            _ = needs_retry
            updates: GraphState = {
                "retry": next_retry_context,
            }
            updates.update(build_followup_updates(guided_followup, attempt=response.synthesis_attempt, **stamp))
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

        snapshot, local_errors = collect_validation_snapshot(state, max_turns=max_turns)
        # Source rejection can change the document scope used by static coverage.
        # Keep semantic failures separate until static quality is assessed against
        # the accepted sources, so obsolete raw-source failures cannot survive.
        assessment = ValidationAssessment()
        assess_pre_semantics(
            snapshot, assessment, decisions_client=decisions_client,
            threshold=pass_threshold, attempt=retry_context.attempt + 1,
        )
        # Keep the raw observations for diagnostics, but all downstream source
        # consumers use the recorded per-requirement, per-source verdicts.
        retrieval = get_retrieval_state(state).model_copy(update={
            "evidence_verdicts": assessment.evidence_verdicts,
        })
        snapshot = replace(snapshot, parsed_hits=assessment.accepted_hits)
        accepted_quality = assess_retrieval_quality(snapshot)
        assessment.failed_requirement_ids.update(accepted_quality.failed_requirement_ids)
        assessment.failed_routes.update(accepted_quality.failed_routes)
        if accepted_quality.retry_reason is not None:
            assessment.retry_reason = accepted_quality.retry_reason
        request_level_failure = (
            "request_ready" in assessment.semantic_refusals
            or (assessment.request_readiness_failed and not assessment.failed_requirement_ids)
        )
        needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
            retry_context=retry_context,
            retry_reason=None if request_level_failure else assessment.retry_reason,
            planner_output=snapshot.planner_output,
            score_avg=assessment.score_avg,
            failed_routes=assessment.failed_routes,
            failed_requirement_ids=assessment.failed_requirement_ids,
            current_attempt_hits=snapshot.parsed_hits,
            current_attempt_retrieval_diagnostics=snapshot.current_attempt_retrieval_diagnostics,
            request_contract=snapshot.request_contract,
        )
        if request_level_failure:
            # A global readiness failure is not evidence that any accepted
            # source is bad. Re-searching those sources cannot repair it.
            needs_retry = False
            next_retry_context = next_retry_context.model_copy(update={
                "needs_retry": False, "content_repair_attempt": retry_context.content_repair_attempt,
                "retry_scope": "refresh_routes", "retry_reason": "missing_content",
                "preserved_hits": list(assessment.accepted_hits),
            })

        if assessment.semantic_probabilities or assessment.semantic_refusals:
            failures = {name: round(value, 4) for name, value in assessment.semantic_probabilities.items()
                        if value < pass_threshold}
            semantic_event = (
                f"pre_synthesis_validation: decisions threshold={pass_threshold}, "
                f"predicates={len(assessment.semantic_probabilities) + len(assessment.semantic_refusals)}, "
                f"failed={failures}, refused={sorted(assessment.semantic_refusals)}"
            )
            evidence_event = "pre_synthesis_validation: evidence_verdicts=" + json.dumps(
                [verdict.model_dump(mode="json") for verdict in assessment.evidence_verdicts],
                ensure_ascii=False,
            )
            updates_debug_events = [*debug.validation_events, semantic_event, evidence_event]
        else:
            updates_debug_events = list(debug.validation_events)

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
                evidence_count=len(snapshot.parsed_hits),
                rejected_evidence_count=sum(item.status != "passed" for item in assessment.evidence_verdicts),
                failed_requirement_ids=sorted(assessment.failed_requirement_ids),
                needs_retry=needs_retry,
                retry_reason=assessment.retry_reason,
            )

        updates: GraphState = {
            "retry": next_retry_context,
            "retrieval": retrieval,
        }
        if request_level_failure:
            updates.update(build_followup_updates(
                "요청의 내용과 조건에 맞는 답변을 준비할 수 있는지 확인하지 못했습니다. 요청을 더 구체적으로 알려 주세요.",
                attempt=response.synthesis_attempt, kind="failure",
                problem=make_problem(
                    "model_refusal" if "request_ready" in assessment.semantic_refusals else "evidence_insufficient",
                    "pre_synthesis_validation",
                ), **stamp,
            ))
        elif assessment.retry_reason is not None and not needs_retry:
            usable_hits = [hit for hit in snapshot.parsed_hits if tasks_for_hit(hit, snapshot.planner_output)]
            if usable_hits:
                # Preserve the failed obligations while allowing a grounded partial
                # body. Post-validation keeps it ineligible for delivery.
                updates["retry"] = next_retry_context.model_copy(update={
                    "terminal_failed_requirement_ids": sorted(
                        set(next_retry_context.terminal_failed_requirement_ids) | assessment.failed_requirement_ids
                    ),
                    "terminal_retrieval_reason": assessment.retry_reason,
                })
                if local_errors:
                    updates["debug"] = debug.model_copy(update={
                        "validation_errors": [*debug.validation_errors, *local_errors],
                        "validation_events": [*updates_debug_events, *local_errors],
                    })
                return updates
            followup_answer = build_followup_from_routes(
                snapshot.planner_output,
                assessment.retry_reason,
            )
            if not snapshot.planner_output.tasks:
                followup_answer = "요청에 필요한 내용을 충분히 확인하지 못해 답변을 준비하지 못했습니다. 요청할 내용과 조건을 더 구체적으로 알려 주세요."
            if assessment.retry_reason == "no_evidence" and assessment.failed_requirement_ids:
                notices = []
                for task in snapshot.planner_output.tasks:
                    if task.requirement_id not in assessment.failed_requirement_ids:
                        continue
                    diagnostics = [d for d in snapshot.current_attempt_retrieval_diagnostics if d.requirement_id == task.requirement_id]
                    subjects = retrieval_subject(task)
                    if task.requirement.file_ids:
                        observed_files = {upload_file_id(hit.evidence) for hit in snapshot.parsed_hits
                                          if hit.requirement_id == task.requirement_id}
                        missing = [item for diagnostic in diagnostics for item in diagnostic.missing_requirements]
                        file_ids = [file_id for file_id in task.requirement.file_ids
                                    if file_id not in observed_files or any(
                                        item == f"file:{file_id}" or item.startswith(f"file:{file_id}:") for item in missing)]
                        names = {item.file_id: item.name for item in runtime.upload_files}
                        labels = ", ".join(f"{names.get(file_id, file_id)} ({file_id})"
                                           for file_id in file_ids or task.requirement.file_ids)
                        notices.append(f"첨부 파일 {labels}에서 {subjects}에 필요한 근거를 충분히 확인하지 못했습니다.")
                        continue
                    if task.route == "upload" and any(d.answerability == "missing" for d in diagnostics):
                        label = "정의" if task.requirement.match == "definition" else "대상"
                        notices.append(f"현재 업로드 파일에서 {subjects} {label}를 찾지 못했습니다.")
                    else:
                        version = f" {task.requirement.version} 버전" if task.requirement.version else ""
                        source = "공식 문서" if task.route == "docs" else "업로드 파일"
                        notices.append(f"{source}에서 {subjects}{version}에 필요한 근거를 충분히 확인하지 못했습니다.")
                followup_answer = " ".join(notices) or followup_answer
            updates.update(
                build_followup_updates(
                    followup_answer,
                    attempt=response.synthesis_attempt,
                    kind="failure",
                    problem=make_problem(
                        "provider_unavailable" if assessment.retry_reason == "tool_error" else "evidence_insufficient",
                        "retrieval" if snapshot.planner_output.tasks else "pre_synthesis_validation",
                    ),
                    **stamp,
                )
            )
        if local_errors or assessment.semantic_probabilities or assessment.semantic_refusals:
            updates["debug"] = debug.model_copy(
                update={
                    "validation_errors": [*debug.validation_errors, *local_errors],
                    "validation_events": [*updates_debug_events, *local_errors],
                }
            )
        return updates

    return pre_synthesis_validation
