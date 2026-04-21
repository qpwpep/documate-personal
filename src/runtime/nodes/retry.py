from __future__ import annotations

from typing import Any

from src.core.contracts import GraphState, RetrievalDiagnostic
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.debug import DEFAULT_MAX_RETRIES, RETRYABLE_REASONS, RetryReason, RetryState
from src.core.contracts.routes import normalize_routes, route_for_tool
from src.core.evidence import EvidenceItem, evidence_to_dicts
from src.core.planner_schema import PlannerOutput

def _normalize_failed_routes(value: set[str] | list[str] | tuple[str, ...] | None) -> list[str]:
    return normalize_routes(value)


def _preserve_successful_route_payload(
    *,
    current_attempt_evidence: list[EvidenceItem],
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic],
    failed_routes: set[str],
) -> tuple[list[dict[str, Any]], list[RetrievalDiagnostic]]:
    preserved_evidence = evidence_to_dicts(
        [
            item
            for item in current_attempt_evidence
            if route_for_tool(item.tool) and route_for_tool(item.tool) not in failed_routes
        ]
    )
    preserved_diagnostics = [
        item.model_copy(deep=True)
        for item in current_attempt_retrieval_diagnostics
        if str(item.route or "").strip() not in failed_routes
    ]
    return preserved_evidence, preserved_diagnostics


def format_retry_context_for_planner(state: GraphState, retry_context: RetryState) -> str | None:
    attempt = int(retry_context.attempt)
    if attempt <= 0:
        return None

    max_retries = int(retry_context.max_retries or DEFAULT_MAX_RETRIES)
    retry_reason = str(retry_context.retry_reason or "no_evidence")
    score_avg = retry_context.score_avg
    score_text = f"{score_avg:.3f}" if isinstance(score_avg, (int, float)) else "n/a"

    previous_output = get_planner_state(state).output
    if previous_output.use_retrieval and previous_output.tasks:
        previous_routes = ", ".join(task.route for task in previous_output.tasks)
    else:
        previous_routes = "none"
    failed_routes = ", ".join(retry_context.failed_routes) or "none"

    return (
        "[Retry Context]\n"
        f"attempt={attempt}/{max_retries}\n"
        f"reason={retry_reason}\n"
        f"previous_routes={previous_routes}\n"
        f"failed_routes={failed_routes}\n"
        f"score_avg={score_text}\n"
        "Use a shorter, route-specific query."
    )


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
            return "upload retriever unavailable; switch to docs/local routes."
        return "retrieval tool error detected; broaden query and simplify route strategy."
    if reason == "no_evidence":
        selected_routes = ", ".join(task.route for task in planner_output.tasks) if planner_output.tasks else "none"
        return f"query too narrow or domain mismatch on routes: {selected_routes}"
    if reason == "unsupported_claims":
        return "generated claims referenced unsupported evidence ids; keep only grounded claims."
    if reason in {"missing", "missing_route_coverage"}:
        return "selected retrieval routes were found, but the answer did not preserve grounded coverage for every required route."
    if reason == "missing_sections":
        return "the answer did not satisfy the requested response structure; regenerate using the required sections only."
    if score_avg is not None:
        return f"low evidence confidence(avg_score={score_avg:.3f}); broaden query or switch route."
    return "low evidence confidence; broaden query or switch route."


def build_missing_upload_followup() -> str:
    return "업로드한 파일을 확인하려면 `.py` 또는 `.ipynb` 파일을 먼저 올린 뒤 다시 질문해 주세요."


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
        if routes == {"local"}:
            return "로컬 예제 검색 중 문제가 있었습니다. 찾고 싶은 함수명이나 노트북 주제를 더 구체적으로 알려 주세요."
        return "검색 경로에서 문제가 있었습니다. 확인할 API 이름이나 비교 대상을 더 구체적으로 알려 주세요."
    if reason == "unsupported_claims":
        return "근거로 확인할 코드 위치나 함수명을 더 구체적으로 알려 주시면, 확인 가능한 내용만 다시 정리하겠습니다."
    if reason in {"missing", "missing_route_coverage", "missing_sections"}:
        return "요청한 구조(예: 비교/체크리스트/해석 분기)와 근거 범위를 유지할 수 있도록, 확인할 API나 비교 대상을 조금 더 구체적으로 알려 주세요."
    if routes == {"docs"}:
        return "공식 문서에서 찾을 라이브러리명이나 API 이름을 더 구체적으로 알려 주세요."
    if routes == {"upload"}:
        return "업로드한 파일에서 찾을 함수명이나 코드 위치를 더 구체적으로 알려 주세요."
    if routes == {"local"}:
        return "로컬 예제에서 찾을 함수명이나 노트북 주제를 더 구체적으로 알려 주세요."
    if routes == {"docs", "upload"}:
        return "공식 문서와 업로드 파일에서 함께 확인할 API나 함수명을 더 구체적으로 알려 주세요."
    if routes == {"docs", "local"}:
        return "공식 문서와 로컬 예제에서 함께 확인할 API나 함수명을 더 구체적으로 알려 주세요."
    return "찾고 싶은 대상과 범위를 조금 더 구체적으로 알려 주세요."


def build_followup_from_routes(
    planner_output: PlannerOutput,
    reason: RetryReason,
) -> str:
    if reason == "blocked_missing_upload":
        return build_missing_upload_followup()
    if reason == "unsupported_claims":
        return (
            "현재 답변 초안의 근거 매핑이 충분하지 않아, 확인 가능한 근거만으로 다시 답하기 어렵습니다. "
            "확인이 필요한 코드 위치나 함수명을 더 구체적으로 알려 주세요."
        )
    route_specific_followup = build_route_specific_followup(planner_output, reason)
    return f"현재 확인 가능한 근거를 찾지 못했습니다. {route_specific_followup}"


def current_retrieval_attempt(retry_context: RetryState) -> int:
    return int(retry_context.attempt) + 1


def _is_repair_retry_candidate(selected_routes: set[str], retry_reason: RetryReason | None) -> bool:
    if retry_reason not in {"unsupported_claims", "missing", "missing_route_coverage", "missing_sections"}:
        return False
    return selected_routes in ({"docs"}, {"docs", "upload"})


def build_retry_update(
    *,
    retry_context: RetryState,
    retry_reason: RetryReason | None,
    planner_output: PlannerOutput,
    retrieval_errors: list[str],
    score_avg: float | None,
    failed_routes: set[str] | list[str] | tuple[str, ...] | None = None,
    current_attempt_evidence: list[EvidenceItem] | None = None,
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic] | None = None,
) -> tuple[bool, RetryState, str]:
    max_retries = int(retry_context.max_retries or DEFAULT_MAX_RETRIES)
    used_retries = int(retry_context.attempt)

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
    normalized_failed_routes = set(_normalize_failed_routes(failed_routes))
    reuse_evidence_only = _is_repair_retry_candidate(selected_routes, retry_reason)

    if retry_reason is not None:
        retrieval_feedback = build_retrieval_feedback(
            retry_reason,
            planner_output=planner_output,
            retrieval_errors=retrieval_errors,
            score_avg=score_avg,
        )
        if retry_reason in RETRYABLE_REASONS and used_retries < max_retries:
            if reuse_evidence_only:
                needs_retry = True
            elif selected_routes == {"docs"}:
                needs_retry = True
            elif selected_routes == {"docs", "upload"} and normalized_failed_routes == {"docs"}:
                needs_retry = True
        if needs_retry:
            needs_retry = True
            used_retries += 1
        retry_update: dict[str, Any] = {
            "retry_reason": retry_reason,
            "retrieval_feedback": retrieval_feedback,
            "failed_routes": (
                        []
                        if reuse_evidence_only
                        else _normalize_failed_routes(normalized_failed_routes or selected_routes)
                    ),
                    "retry_scope": (
                        "reuse_evidence_resynthesize"
                        if reuse_evidence_only
                        else "refresh_routes"
                    ),
        }
        if reuse_evidence_only:
            retry_update["preserved_evidence"] = evidence_to_dicts(current_attempt_evidence or [])
            retry_update["preserved_retrieval_diagnostics"] = [
                item.model_copy(deep=True)
                for item in (current_attempt_retrieval_diagnostics or [])
            ]
        elif selected_routes == {"docs", "upload"} and normalized_failed_routes == {"docs"}:
            preserved_evidence, preserved_diagnostics = _preserve_successful_route_payload(
                current_attempt_evidence=current_attempt_evidence or [],
                current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics or [],
                failed_routes=normalized_failed_routes,
            )
            retry_update["preserved_evidence"] = preserved_evidence
            retry_update["preserved_retrieval_diagnostics"] = preserved_diagnostics
        else:
            retry_update["preserved_evidence"] = []
            retry_update["preserved_retrieval_diagnostics"] = []
    else:
        retry_update = {
            "retrieval_feedback": "",
            "retry_reason": None,
            "failed_routes": [],
            "preserved_evidence": [],
            "preserved_retrieval_diagnostics": [],
            "retry_scope": "refresh_routes",
        }

    retry_update["attempt"] = used_retries
    retry_update["needs_retry"] = needs_retry
    next_retry_context = next_retry_context.model_copy(update=retry_update)
    return needs_retry, next_retry_context, retrieval_feedback
