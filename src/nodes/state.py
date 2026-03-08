from __future__ import annotations

import json
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from ..planner_schema import PlannerOutput

RetryReason = Literal[
    "no_evidence",
    "low_score",
    "tool_error",
    "blocked_missing_upload",
    "unsupported_claims",
]
PlannerStatus = Literal["llm", "heuristic_fallback", "fallback_no_routes"]
PlannerOverrideReason = Literal[
    "missing_required_retrieval",
    "missing_required_routes",
    "upload_retriever_missing",
]
LOW_SCORE_THRESHOLD = 0.5
DEFAULT_MAX_RETRIES = 1
RETRYABLE_REASONS: set[RetryReason] = {"no_evidence", "low_score", "unsupported_claims"}
ROUTE_ORDER: tuple[str, ...] = ("docs", "upload", "local")


class RetryContext(TypedDict, total=False):
    attempt: int
    max_retries: int
    retry_reason: RetryReason
    retrieval_feedback: str
    evidence_start_index: int
    retrieval_error_start_index: int
    retrieval_diagnostic_start_index: int
    score_avg: float | None


class PlannerDiagnostic(TypedDict, total=False):
    status: PlannerStatus
    reason: str | None
    fallback_routes: list[str]
    intent_required: bool
    required_routes: list[str]
    override_applied: bool
    override_reason: PlannerOverrideReason | None


class RetrievalDiagnostic(TypedDict, total=False):
    tool: str
    route: str
    status: str
    message: str
    query: str
    attempt: int


def merge_string_lists(current: list[str] | None, update: list[str] | None) -> list[str]:
    merged = list(current or [])
    if update:
        merged.extend(update)
    return merged


def merge_dict_lists(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged = list(current or [])
    if update:
        merged.extend(update)
    return merged


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def slice_from_index(items: list[Any], start_index: int) -> list[Any]:
    if start_index < 0:
        start_index = 0
    if start_index >= len(items):
        return []
    return items[start_index:]


def coerce_retry_context(value: Any) -> RetryContext:
    context: RetryContext = {
        "attempt": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retrieval_feedback": "",
        "evidence_start_index": 0,
        "retrieval_error_start_index": 0,
        "retrieval_diagnostic_start_index": 0,
        "score_avg": None,
    }
    if not isinstance(value, dict):
        return context

    attempt = value.get("attempt")
    if isinstance(attempt, int) and attempt >= 0:
        context["attempt"] = attempt

    max_retries = value.get("max_retries")
    if isinstance(max_retries, int) and max_retries >= 0:
        context["max_retries"] = max_retries

    retry_reason = value.get("retry_reason")
    if retry_reason in {
        "no_evidence",
        "low_score",
        "tool_error",
        "blocked_missing_upload",
        "unsupported_claims",
    }:
        context["retry_reason"] = retry_reason

    retrieval_feedback = value.get("retrieval_feedback")
    if retrieval_feedback is not None:
        context["retrieval_feedback"] = str(retrieval_feedback).strip()

    evidence_start_index = value.get("evidence_start_index")
    if isinstance(evidence_start_index, int) and evidence_start_index >= 0:
        context["evidence_start_index"] = evidence_start_index

    retrieval_error_start_index = value.get("retrieval_error_start_index")
    if isinstance(retrieval_error_start_index, int) and retrieval_error_start_index >= 0:
        context["retrieval_error_start_index"] = retrieval_error_start_index

    retrieval_diagnostic_start_index = value.get("retrieval_diagnostic_start_index")
    if isinstance(retrieval_diagnostic_start_index, int) and retrieval_diagnostic_start_index >= 0:
        context["retrieval_diagnostic_start_index"] = retrieval_diagnostic_start_index

    score_avg = value.get("score_avg")
    if isinstance(score_avg, (int, float)):
        context["score_avg"] = float(score_avg)
    elif score_avg is None:
        context["score_avg"] = None

    return context


def coerce_planner_output(raw: Any, errors: list[str]) -> PlannerOutput:
    if isinstance(raw, PlannerOutput):
        return raw
    try:
        return PlannerOutput.model_validate(raw)
    except Exception as exc:
        errors.append(f"planner: output validation failed ({exc})")
        return PlannerOutput.fallback()


def build_tool_message(tool_name: str, payload: Any, index: int) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=tool_name,
        tool_call_id=f"{tool_name}-{index}",
    )


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    final_answer: Optional[str]
    retriever: Optional[Any]
    memory_summary: Optional[str]
    planner_output: PlannerOutput
    planner_status: PlannerStatus
    planner_diagnostics: PlannerDiagnostic
    guided_followup: Optional[str]
    retrieved_evidence: Annotated[list[dict[str, Any]], merge_dict_lists]
    retrieval_diagnostics: Annotated[list[dict[str, Any]], merge_dict_lists]
    retrieval_errors: Annotated[list[str], merge_string_lists]
    synthesis_errors: Annotated[list[str], merge_string_lists]
    validation_errors: Annotated[list[str], merge_string_lists]
    action_errors: Annotated[list[str], merge_string_lists]
    latency_trace: Annotated[list[dict[str, Any]], merge_dict_lists]
    synthesis_attempt: int
    synthesis_output: dict[str, Any]
    response_payload: dict[str, Any]
    needs_retry: bool
    retry_context: RetryContext
