from __future__ import annotations

from typing import Any

from src.core.contracts.debug import ActionResults, DEBUG_SCHEMA_VERSION, DebugPayload, ErrorCode, LLMCallMetadata, ModelUsageStatus, RetryState, SaveTextActionResult, SlackActionResult, TokenUsage, json_safe_deep_copy
from src.core.contracts.graph_state import DebugState
from src.core.contracts.routes import normalize_routes
from src.core.contracts.boundary.planner import parse_planner_diagnostic
from src.core.contracts.boundary.retrieval import parse_retrieval_diagnostic, parse_retrieval_diagnostics


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
        "missing",
        "missing_route_coverage",
        "missing_sections",
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

    retry_scope = value.get("retry_scope")
    if retry_scope in {"refresh_routes", "reuse_evidence_resynthesize"}:
        retry_state.retry_scope = retry_scope

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
            "structured_hedge",
            "plain_fallback",
            "structured_compact_fallback",
            "plain_summary_attach_fallback",
            "korean_template_summary_fallback",
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


def parse_error_codes(value: Any) -> list[ErrorCode]:
    allowed = set(ErrorCode.__args__)  # type: ignore[attr-defined]
    if not isinstance(value, list):
        return []
    parsed: list[ErrorCode] = []
    for item in value:
        code = str(item or "").strip().upper()
        if code in allowed and code not in parsed:
            parsed.append(code)  # type: ignore[arg-type]
    return parsed


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


def _parse_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def parse_model_usage_status(value: Any, *, has_llm_usage: bool, has_debug_payload: bool = True) -> ModelUsageStatus:
    status = str(value or "").strip().lower()
    if status in {"llm_used", "deterministic", "missing_debug"}:
        return status  # type: ignore[return-value]
    if not has_debug_payload:
        return "missing_debug"
    return "llm_used" if has_llm_usage else "deterministic"


def parse_action_results(value: Any) -> ActionResults | None:
    if isinstance(value, ActionResults):
        return value
    if not isinstance(value, dict):
        return None

    payload: dict[str, Any] = {}
    slack_payload = value.get("slack_notify")
    if isinstance(slack_payload, SlackActionResult):
        payload["slack_notify"] = slack_payload
    elif isinstance(slack_payload, dict):
        payload["slack_notify"] = SlackActionResult(
            status=str(slack_payload.get("status") or "").strip(),
            channel_id=(str(slack_payload.get("channel_id")).strip() if slack_payload.get("channel_id") else None),
            target_type=(str(slack_payload.get("target_type")).strip() if slack_payload.get("target_type") else None),
            error=(str(slack_payload.get("error")).strip() if slack_payload.get("error") else None),
            reason=(str(slack_payload.get("reason")).strip() if slack_payload.get("reason") else None),
            error_code=(parse_error_codes([slack_payload.get("error_code")]) or [None])[0],
        )

    save_payload = value.get("save_text")
    if isinstance(save_payload, SaveTextActionResult):
        payload["save_text"] = save_payload
    elif isinstance(save_payload, dict):
        payload["save_text"] = SaveTextActionResult(
            status=str(save_payload.get("status") or "").strip(),
            file_path=(str(save_payload.get("file_path")).strip() if save_payload.get("file_path") else None),
            bytes=_parse_non_negative_int(save_payload.get("bytes", 0), default=0),
            error=(str(save_payload.get("error")).strip() if save_payload.get("error") else None),
            message=(str(save_payload.get("message")).strip() if save_payload.get("message") else None),
            error_code=(parse_error_codes([save_payload.get("error_code")]) or [None])[0],
        )

    return ActionResults(**payload) if payload else None


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
    raw_schema_version = value.get("schema_version", DEBUG_SCHEMA_VERSION)
    try:
        schema_version = int(raw_schema_version)
    except (TypeError, ValueError):
        schema_version = DEBUG_SCHEMA_VERSION

    observability_status = str(value.get("observability_status") or "ok").strip().lower()
    if observability_status not in {"ok", "degraded", "failed"}:
        observability_status = "ok"

    llm_calls = parse_llm_calls(value.get("llm_calls"))
    models_used = [str(item) for item in value.get("models_used", []) if str(item).strip()] if isinstance(value.get("models_used"), list) else []
    model_name = str(value.get("model_name")) if value.get("model_name") else None
    token_usage = parse_token_usage(value.get("token_usage"))
    has_llm_usage = bool(llm_calls or models_used or model_name or (token_usage is not None and token_usage.total_tokens > 0))

    return DebugPayload(
        schema_version=schema_version,
        observability_status=observability_status,  # type: ignore[arg-type]
        missing_required_debug_fields=[
            str(item)
            for item in value.get("missing_required_debug_fields", [])
            if str(item).strip()
        ]
        if isinstance(value.get("missing_required_debug_fields"), list)
        else [],
        tool_calls=[str(item) for item in value.get("tool_calls", []) if str(item).strip()]
        if isinstance(value.get("tool_calls"), list)
        else [],
        tool_call_count=int(value.get("tool_call_count", 0) or 0),
        token_usage=token_usage,
        model_name=model_name,
        models_used=models_used,
        model_usage_status=parse_model_usage_status(
            value.get("model_usage_status"),
            has_llm_usage=has_llm_usage,
        ),
        llm_calls=llm_calls,
        errors=[str(item) for item in value.get("errors", []) if str(item).strip()]
        if isinstance(value.get("errors"), list)
        else [],
        error_codes=parse_error_codes(value.get("error_codes")),
        validation_events=[
            str(item) for item in value.get("validation_events", []) if str(item).strip()
        ]
        if isinstance(value.get("validation_events"), list)
        else [],
        edge_decisions=[
            dict(item)
            for item in value.get("edge_decisions", [])
            if isinstance(item, dict)
        ]
        if isinstance(value.get("edge_decisions"), list)
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
        action_results=parse_action_results(value.get("action_results")),
    )


def parse_debug_state(value: Any) -> DebugState:
    if isinstance(value, DebugState):
        return value
    if isinstance(value, DebugPayload):
        return DebugState.model_validate(value.model_dump(mode="json"))
    if not isinstance(value, dict):
        return DebugState()

    payload = parse_debug_payload(value).model_dump(mode="json")
    payload["retrieval_errors"] = [
        str(item) for item in value.get("retrieval_errors", []) if str(item).strip()
    ] if isinstance(value.get("retrieval_errors"), list) else []
    payload["synthesis_errors"] = [
        str(item) for item in value.get("synthesis_errors", []) if str(item).strip()
    ] if isinstance(value.get("synthesis_errors"), list) else []
    payload["validation_errors"] = [
        str(item) for item in value.get("validation_errors", []) if str(item).strip()
    ] if isinstance(value.get("validation_errors"), list) else []
    payload["action_errors"] = [
        str(item) for item in value.get("action_errors", []) if str(item).strip()
    ] if isinstance(value.get("action_errors"), list) else []
    payload["latency_trace"] = list(value.get("latency_trace", [])) if isinstance(value.get("latency_trace"), list) else []
    return DebugState.model_validate(payload)


def get_debug_state(state: dict[str, Any]) -> DebugState:
    return parse_debug_state(state.get("debug"))
