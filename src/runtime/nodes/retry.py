from __future__ import annotations

import json
from collections.abc import Collection
from typing import Any

from src.core.contracts import GraphState, RetrievalDiagnostic
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.debug import RETRYABLE_REASONS, RetryReason, RetryState
from src.core.evidence import SearchHit
from src.core.planner_schema import PlannerOutput, PlannerRouteName, RetrievalTask
from src.core.request_contracts import RequestContract


CONTENT_REPAIR_REASONS: set[RetryReason] = {
    "unresolved_references", "missing", "missing_route_coverage", "missing_content",
}


def format_retry_context_for_planner(state: GraphState, retry_context: RetryState) -> str:
    """Supply fixed retrieval meaning and observed requests as reference data."""
    runtime = get_runtime_state(state)
    contract = runtime.request_contract
    failed_ids = set(retry_context.failed_requirement_ids)
    current_tasks = {task.requirement_id: task for task in get_planner_state(state).output.tasks}
    diagnostics = [item for item in get_debug_state(state).retrieval_diagnostics
                   if item.requirement_id in failed_ids][-16:]
    prior_requests = [{
        "requirement_id": d.requirement_id,
        "route": d.route,
        "query": d.query[:600],
        "k": d.k,
        "status": d.status,
        "answerability": d.answerability,
        "candidate_count": d.candidate_count,
        "provider_result_count": d.provider_result_count,
        "missing_requirements": d.missing_requirements[:16],
        "warnings": d.warnings[:12],
        "completed_requests": [request.model_dump(mode="json") for request in d.completed_requests],
        "provider_requests": [request.model_dump(mode="json") for request in d.provider_requests],
    } for d in diagnostics]
    body = contract.body_request or contract.body if contract is not None else None
    context = {
        "attempt": retry_context.attempt,
        "max_retries": retry_context.max_retries,
        "reason": retry_context.retry_reason,
        "score_avg": retry_context.score_avg,
        "feedback": retry_context.retrieval_feedback[:1000],
        "resolved_request": getattr(body, "instruction", ""),
        "latest_user_request": runtime.user_input,
        "failed_tasks": [task.model_copy(update={
            "query": current_tasks[task.requirement_id].query,
            "k": current_tasks[task.requirement_id].k,
        }).model_dump(mode="json") if task.requirement_id in current_tasks else task.model_dump(mode="json")
            for task in retry_context.original_tasks if task.requirement_id in failed_ids],
        "prior_requests": prior_requests,
    }
    return "[Retry Context]\n" + json.dumps(context, ensure_ascii=False)


def contains_tool_error(errors: list[str]) -> bool:
    if not errors:
        return False
    keywords = (
        "failed",
        "error",
        "unavailable",
        "invalid json",
        "payload must",
        "timeout",
    )
    for error in errors:
        lowered = str(error).lower()
        if any(keyword in lowered for keyword in keywords):
            return True
    return False


def build_retrieval_feedback(
    reason: RetryReason,
    *,
    planner_output: PlannerOutput,
    retrieval_errors: list[str],
    score_avg: float | None,
) -> str:
    if reason == "blocked_missing_upload":
        return "uploaded file context is missing; ask the user to upload the file first."
    if reason == "tool_error":
        if any(
            "upload" in str(error).lower() and "unavailable" in str(error).lower()
            for error in retrieval_errors
        ):
            return "upload retriever unavailable; ask the user to upload the file again."
        return "retrieval tool error detected; use the reported failure to revise query/k while you preserve the route and evidence requirements."
    if reason == "no_evidence":
        selected_routes = ", ".join(task.route for task in planner_output.tasks) if planner_output.tasks else "none"
        return f"query too narrow or domain mismatch on routes: {selected_routes}"
    if reason == "unresolved_references":
        return "displayed content referenced evidence not supplied to synthesis; use only the available evidence packet."
    if reason in {"missing", "missing_route_coverage"}:
        return "selected retrieval routes were found, but the answer did not preserve grounded coverage for every required route."
    if reason == "missing_content":
        return "the answer violated its confirmed content or format constraints; repair the body while preserving all required and forbidden conditions."
    if score_avg is not None:
        return f"low retrieval relevance(avg_score={score_avg:.3f}); revise query/k and preserve the route and evidence requirements."
    return "low retrieval relevance; revise query/k and preserve the route and evidence requirements."


def build_missing_upload_followup() -> str:
    return "확인할 파일을 먼저 업로드한 뒤 다시 질문해 주세요."


def build_route_specific_followup(
    planner_output: PlannerOutput,
    reason: RetryReason,
) -> str:
    routes = {task.route for task in planner_output.tasks}
    if reason == "blocked_missing_upload":
        return build_missing_upload_followup()
    if reason == "tool_error":
        if routes == {"docs"}:
            return "공식 문서 조회 중 문제가 있었습니다. 라이브러리명이나 API 이름을 더 구체적으로 알려 주세요."
        if routes == {"upload"}:
            return "업로드 파일 검색 중 문제가 있었습니다. 파일을 다시 올리거나 찾을 함수명을 더 구체적으로 알려 주세요."
        return "검색 경로에서 문제가 있었습니다. 확인할 API 이름이나 비교 대상을 더 구체적으로 알려 주세요."
    if reason == "unresolved_references":
        return "근거로 확인할 코드 위치나 함수명을 더 구체적으로 알려 주시면, 확인 가능한 내용만 다시 정리하겠습니다."
    if reason in {"missing", "missing_route_coverage", "missing_content"}:
        return "요청한 구조(예: 비교/체크리스트/해석 분기)와 근거 범위를 유지할 수 있도록, 확인할 API나 비교 대상을 조금 더 구체적으로 알려 주세요."
    if routes == {"docs"}:
        return "공식 문서에서 찾을 라이브러리명이나 API 이름을 더 구체적으로 알려 주세요."
    if routes == {"upload"}:
        return "업로드한 파일에서 찾을 함수명이나 코드 위치를 더 구체적으로 알려 주세요."
    if routes == {"docs", "upload"}:
        return "공식 문서와 업로드 파일에서 함께 확인할 API나 함수명을 더 구체적으로 알려 주세요."
    return "찾고 싶은 대상과 범위를 조금 더 구체적으로 알려 주세요."


def build_followup_from_routes(
    planner_output: PlannerOutput,
    reason: RetryReason,
) -> str:
    if reason == "blocked_missing_upload":
        return build_missing_upload_followup()
    if reason == "unresolved_references":
        return (
            "현재 답변 초안의 근거 매핑이 충분하지 않아, 확인 가능한 근거만으로 다시 답하기 어렵습니다. "
            "확인이 필요한 코드 위치나 함수명을 더 구체적으로 알려 주세요."
        )
    route_specific_followup = build_route_specific_followup(planner_output, reason)
    return f"현재 확인 가능한 근거를 찾지 못했습니다. {route_specific_followup}"


def current_retrieval_attempt(retry_context: RetryState) -> int:
    return retry_context.attempt + 1


def _is_repair_retry_candidate(selected_routes: set[str], retry_reason: RetryReason | None) -> bool:
    if retry_reason not in CONTENT_REPAIR_REASONS:
        return False
    return bool(selected_routes) and selected_routes.issubset({"docs", "upload"})


def _can_refresh_search(task: RetrievalTask, diagnostics: list[RetrievalDiagnostic]) -> bool:
    if task.route == "docs":
        return True
    requirement = task.requirement
    # Only topic/aspect vector searches can establish coverage through new
    # candidates. Source symbol absence and unsupported source constraints
    # cannot be repaired by changing the query.
    if requirement.match != "topic" or requirement.symbols or requirement.library or requirement.version or requirement.evidence_needs:
        return False
    return not any(
        item.requirement_id == task.requirement_id
        and (item.status == "unavailable" or item.error_code == "UPLOAD_FILE_SCOPE_INVALID")
        for item in diagnostics
    )


def build_retry_update(
    *,
    retry_context: RetryState,
    retry_reason: RetryReason | None,
    planner_output: PlannerOutput,
    retrieval_errors: list[str],
    score_avg: float | None,
    failed_routes: Collection[PlannerRouteName] | None = None,
    request_contract: RequestContract | None = None,
    failed_requirement_ids: set[str] | list[str] | None = None,
    current_attempt_hits: list[SearchHit] | None = None,
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic] | None = None,
    content_repair_available: bool = True,
) -> tuple[bool, RetryState, str]:
    max_retries = max(0, retry_context.max_retries)
    used_retries = retry_context.attempt

    needs_retry = False
    retrieval_feedback = ""

    next_retry_context = retry_context.model_copy(
        update={
            "needs_retry": False,
            "max_retries": max_retries,
            "score_avg": score_avg,
        }
    )

    selected_routes = {task.route for task in planner_output.tasks}
    failed_route_set = set(failed_routes or ())
    failed_ids = set(failed_requirement_ids or [])
    if retry_reason is not None and retry_reason not in CONTENT_REPAIR_REASONS and not failed_ids:
        # Resolve route-wide provider failures to concrete server-owned tasks.
        failed_ids = {task.requirement_id for task in planner_output.tasks
                      if task.route in (failed_route_set or selected_routes)}
    # A request that needs no retrieval can still repair a generated or
    # transformed body. Re-enter synthesis, never reinterpret the request.
    contract = request_contract
    repair_without_retrieval = (
        not selected_routes and retry_reason in CONTENT_REPAIR_REASONS
        and contract is not None and contract.can_prepare_body()
        and contract.body.kind in {"compose", "transform_input", "transform_answer"}
    )
    reuse_hits_only = repair_without_retrieval or (
        _is_repair_retry_candidate(selected_routes, retry_reason) and bool(current_attempt_hits)
    )

    if retry_reason is not None:
        retrieval_feedback = build_retrieval_feedback(
            retry_reason,
            planner_output=planner_output,
            retrieval_errors=retrieval_errors,
            score_avg=score_avg,
        )
        # Content repair is bounded to one extra synthesis, even if retrieval
        # retries have a larger configured budget. An explicit zero disables it.
        retry_limit = min(max_retries, 1) if retry_reason in CONTENT_REPAIR_REASONS else max_retries
        if retry_reason in RETRYABLE_REASONS and used_retries < retry_limit:
            if retry_reason in CONTENT_REPAIR_REASONS:
                needs_retry = reuse_hits_only and content_repair_available
            else:
                retryable_ids = {
                    task.requirement_id for task in planner_output.tasks
                    if task.requirement_id in failed_ids
                    and _can_refresh_search(task, current_attempt_retrieval_diagnostics or [])
                }
                needs_retry = bool(retryable_ids)
                if needs_retry:
                    failed_ids = retryable_ids
                    failed_route_set = {task.route for task in planner_output.tasks
                                                if task.requirement_id in failed_ids}
        if needs_retry:
            used_retries += 1
        retry_update: dict[str, Any] = {
            "retry_reason": retry_reason,
            "retrieval_feedback": retrieval_feedback,
            "failed_routes": (
                [] if reuse_hits_only
                else sorted(failed_route_set or selected_routes)
            ),
            "retry_scope": "reuse_hits_resynthesize" if reuse_hits_only else "refresh_routes",
            "failed_requirement_ids": sorted(failed_ids),
        }
        if reuse_hits_only:
            retry_update["preserved_hits"] = [hit.model_dump(mode="json") for hit in (current_attempt_hits or [])]
            retry_update["preserved_retrieval_diagnostics"] = [
                item.model_copy(deep=True)
                for item in (current_attempt_retrieval_diagnostics or [])
            ]
        elif failed_ids:
            retry_update["preserved_hits"] = [hit.model_dump(mode="json") for hit in (current_attempt_hits or []) if hit.requirement_id not in failed_ids]
            retry_update["preserved_retrieval_diagnostics"] = [d.model_copy(deep=True) for d in (current_attempt_retrieval_diagnostics or []) if d.requirement_id not in failed_ids]
        else:
            retry_update["preserved_hits"] = []
            retry_update["preserved_retrieval_diagnostics"] = []
    else:
        retry_update = {
            "retrieval_feedback": "",
            "retry_reason": None,
            "failed_routes": [],
            "failed_requirement_ids": [],
            "preserved_hits": [],
            "preserved_retrieval_diagnostics": [],
            "retry_scope": "refresh_routes",
        }

    retry_update["attempt"] = used_retries
    retry_update["needs_retry"] = needs_retry
    next_retry_context = next_retry_context.model_copy(update=retry_update)
    return needs_retry, next_retry_context, retrieval_feedback
