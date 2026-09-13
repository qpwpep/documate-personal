from __future__ import annotations

import logging

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.infra.logging_utils import log_event
from src.runtime.nodes.retry import build_followup_from_routes, build_retry_update
from src.runtime.nodes.validation.evidence_validator import assess_retrieval_quality, collect_validation_snapshot
from src.runtime.nodes.validation.policy import build_followup_updates
from src.runtime.nodes.synthesis.evidence_selection import upload_file_id


logger = logging.getLogger(__name__)


def make_pre_synthesis_validation_node(verbose: bool):
    def pre_synthesis_validation(state: GraphState) -> GraphState:
        planner = get_planner_state(state)
        response = get_response_state(state)
        debug = get_debug_state(state)
        retry_context = get_retry_state(state)
        guided_followup = str(planner.guided_followup or "").strip()
        runtime = get_runtime_state(state)
        contract = runtime.request_contract
        stamp = {"request_id": contract.request_id if contract else None,
                 "contract_revision": contract.revision if contract else 0,
                 "body_kind": contract.body.kind if contract else "unresolved",
                 "evidence_source": None}
        if (contract is not None and contract.can_prepare_body()
                and planner.diagnostics.reason not in {"upload_retriever_missing", "upload_file_scope_invalid", "planner_unavailable"}):
            guided_followup = ""

        if guided_followup:
            if planner.diagnostics.reason != "upload_retriever_missing":
                updates = {"retry": retry_context.model_copy(update={
                    "needs_retry": False, "retry_reason": None, "failed_routes": [],
                    "failed_requirement_ids": [], "retrieval_feedback": "",
                })}
                updates.update(build_followup_updates(guided_followup, attempt=response.synthesis_attempt, **stamp))
                return updates
            planner_output = parse_planner_output(planner.output, [])
            planner_unavailable = planner.diagnostics.reason == "planner_unavailable"
            needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
                retry_context=retry_context,
                retry_reason=None if planner_unavailable else "blocked_missing_upload",
                planner_output=planner_output,
                retrieval_errors=[],
                score_avg=None,
                failed_routes=set() if planner_unavailable else {"upload"},
                request_contract=contract,
            )
            _ = needs_retry
            updates: GraphState = {
                "retry": next_retry_context,
            }
            updates.update(build_followup_updates(guided_followup, attempt=response.synthesis_attempt, **stamp))
            if planner_unavailable:
                return updates
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
            failed_requirement_ids=assessment.failed_requirement_ids,
            current_attempt_hits=snapshot.parsed_hits,
            current_attempt_retrieval_diagnostics=snapshot.current_attempt_retrieval_diagnostics,
            request_contract=snapshot.request_contract,
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
                evidence_count=len(snapshot.parsed_hits),
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
            if assessment.retry_reason == "no_evidence" and assessment.failed_requirement_ids:
                notices = []
                for task in snapshot.planner_output.tasks:
                    if task.requirement_id not in assessment.failed_requirement_ids:
                        continue
                    diagnostics = [d for d in snapshot.current_attempt_retrieval_diagnostics if d.requirement_id == task.requirement_id]
                    subjects = ", ".join(task.requirement.symbols) or task.requirement.library or task.query
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
                    **stamp,
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
