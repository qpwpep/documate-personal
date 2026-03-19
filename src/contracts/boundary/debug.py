from __future__ import annotations

from typing import Any

from ..debug import (
    DebugPayload,
    LLMCallMetadata,
    RetryState,
    TokenUsage,
    json_safe_deep_copy,
)
from ..graph_state import DebugState
from ..routes import normalize_routes
from .planner import parse_planner_diagnostic
from .retrieval import parse_retrieval_diagnostic, parse_retrieval_diagnostics


def parse_retry_state(value: Any) -> RetryState:
    if isinstance(value, RetryState):
        return value

    retry_state = RetryState()
    if not isinstance(value, dict):
        return retry_state

    attempt = value.get("attempt")
    if isinstance(attempt, int) and attempt >= 0:
        retry_state.attempt = attempt

    max_retries = value.get("max_retries")
    if isinstance(max_retries, int) and max_retries >= 0:
        retry_state.max_retries = max_retries

    retry_reason = value.get("retry_reason")
    if retry_reason in {
        "no_evidence",
        "low_score",
        "tool_error",
        "blocked_missing_upload",
        "unsupported_claims",
    }:
        retry_state.retry_reason = retry_reason

    retry_state.needs_retry = bool(value.get("needs_retry", retry_state.needs_retry))

    retrieval_feedback = value.get("retrieval_feedback")
    if retrieval_feedback is not None:
        retry_state.retrieval_feedback = str(retrieval_feedback).strip()

    evidence_start_index = value.get("evidence_start_index")
    if isinstance(evidence_start_index, int) and evidence_start_index >= 0:
        retry_state.evidence_start_index = evidence_start_index

    retrieval_error_start_index = value.get("retrieval_error_start_index")
    if isinstance(retrieval_error_start_index, int) and retrieval_error_start_index >= 0:
        retry_state.retrieval_error_start_index = retrieval_error_start_index

    retrieval_diagnostic_start_index = value.get("retrieval_diagnostic_start_index")
    if isinstance(retrieval_diagnostic_start_index, int) and retrieval_diagnostic_start_index >= 0:
        retry_state.retrieval_diagnostic_start_index = retrieval_diagnostic_start_index

    score_avg = value.get("score_avg")
    if isinstance(score_avg, (int, float)):
        retry_state.score_avg = float(score_avg)
    elif score_avg is None and "score_avg" in value:
        retry_state.score_avg = None

    failed_routes = value.get("failed_routes")
    if isinstance(failed_routes, list):
        retry_state.failed_routes = normalize_routes(failed_routes)

    preserved_evidence = value.get("preserved_evidence")
    if isinstance(preserved_evidence, list):
        retry_state.preserved_evidence = [
            json_safe_deep_copy(item)
            for item in preserved_evidence
            if isinstance(item, dict)
        ]

    preserved_retrieval_diagnostics = value.get("preserved_retrieval_diagnostics")
    if isinstance(preserved_retrieval_diagnostics, list):
        retry_state.preserved_retrieval_diagnostics = [
            diagnostic
            for item in preserved_retrieval_diagnostics
            if (diagnostic := parse_retrieval_diagnostic(item)) is not None
        ]

    return retry_state


def parse_llm_calls(value: Any) -> list[LLMCallMetadata]:
    if not isinstance(value, list):
        return []
    calls: list[LLMCallMetadata] = []
    for item in value:
        if isinstance(item, LLMCallMetadata):
            calls.append(item)
            continue
        if not isinstance(item, dict):
            continue

        stage = str(item.get("stage") or "").strip()
        if stage not in {"summarize", "planner", "synthesis"}:
            continue

        path = str(item.get("path") or "").strip()
        if path not in {
            "direct",
            "structured",
            "plain_fallback",
            "structured_compact_fallback",
            "plain_summary_attach_fallback",
        }:
            continue

        try:
            attempt = int(item.get("attempt", 0) or 0)
        except (TypeError, ValueError):
            attempt = 0

        response_metadata = item.get("response_metadata")
        usage_metadata = item.get("usage_metadata")
        calls.append(
            LLMCallMetadata(
                stage=stage,
                attempt=max(0, attempt),
                path=path,
                response_metadata=dict(response_metadata) if isinstance(response_metadata, dict) else {},
                usage_metadata=dict(usage_metadata) if isinstance(usage_metadata, dict) else {},
            )
        )
    return calls


def parse_token_usage(value: Any) -> TokenUsage | None:
    if isinstance(value, TokenUsage):
        return value
    if not isinstance(value, dict):
        return None
    try:
        return TokenUsage(
            prompt_tokens=int(value.get("prompt_tokens", 0) or 0),
            completion_tokens=int(value.get("completion_tokens", 0) or 0),
            total_tokens=int(value.get("total_tokens", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


def parse_debug_payload(value: Any) -> DebugPayload:
    if isinstance(value, DebugPayload):
        return value
    if not isinstance(value, dict):
        return DebugPayload()

    observed_evidence = (
        [
            dict(item)
            for item in value.get("observed_evidence", [])
            if isinstance(item, dict)
        ]
        if isinstance(value.get("observed_evidence"), list)
        else []
    )
    return DebugPayload(
        tool_calls=[str(item) for item in value.get("tool_calls", []) if str(item).strip()]
        if isinstance(value.get("tool_calls"), list)
        else [],
        tool_call_count=int(value.get("tool_call_count", 0) or 0),
        token_usage=parse_token_usage(value.get("token_usage")),
        model_name=str(value.get("model_name")) if value.get("model_name") else None,
        models_used=[str(item) for item in value.get("models_used", []) if str(item).strip()]
        if isinstance(value.get("models_used"), list)
        else [],
        llm_calls=parse_llm_calls(value.get("llm_calls")),
        errors=[str(item) for item in value.get("errors", []) if str(item).strip()]
        if isinstance(value.get("errors"), list)
        else [],
        planner_errors=[str(item) for item in value.get("planner_errors", []) if str(item).strip()]
        if isinstance(value.get("planner_errors"), list)
        else [],
        observed_evidence=observed_evidence,
        retry_context=parse_retry_state(value.get("retry_context")) if value.get("retry_context") else None,
        retrieval_diagnostics=parse_retrieval_diagnostics(value.get("retrieval_diagnostics")),
        planner_diagnostics=parse_planner_diagnostic(value.get("planner_diagnostics")),
        latency_breakdown=dict(value.get("latency_breakdown"))
        if isinstance(value.get("latency_breakdown"), dict)
        else None,
    )


def parse_debug_state(value: Any) -> DebugState:
    if isinstance(value, DebugState):
        return value
    if isinstance(value, DebugPayload):
        payload = value
    elif not isinstance(value, dict):
        return DebugState()
    else:
        payload = parse_debug_payload(value)
    return DebugState.model_validate(payload.model_dump(mode="json"))


def get_debug_state(state: dict[str, Any]) -> DebugState:
    return parse_debug_state(state.get("debug"))
