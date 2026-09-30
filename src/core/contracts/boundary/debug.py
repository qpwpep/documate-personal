from __future__ import annotations

from typing import Any

from src.core.slack_contract import SlackDelivery
from src.core.contracts.debug import ActionResults, DEBUG_SCHEMA_VERSION, DebugDiagnostics, DebugPayload, ErrorCode, RetryState, SaveTextActionResult, json_safe_deep_copy, normalize_recorded_routes
from src.core.contracts.graph_state import DebugState
from src.core.contracts.usage import LLMCallRecord
from src.core.contracts.routing import validate_route_decisions
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
        "unresolved_references",
        "missing",
        "missing_route_coverage",
        "missing_content",
    }:
        retry_state.retry_reason = retry_reason

    retry_state.needs_retry = bool(value.get("needs_retry", retry_state.needs_retry))

    retrieval_feedback = value.get("retrieval_feedback")
    if retrieval_feedback is not None:
        retry_state.retrieval_feedback = str(retrieval_feedback).strip()

    hit_start_index = value.get("hit_start_index")
    if isinstance(hit_start_index, int) and hit_start_index >= 0:
        retry_state.hit_start_index = hit_start_index

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
        retry_state.failed_routes = normalize_recorded_routes(failed_routes)
    if isinstance(value.get("failed_requirement_ids"), list):
        retry_state.failed_requirement_ids = [str(item) for item in value["failed_requirement_ids"] if str(item).strip()]
    if isinstance(value.get("original_tasks"), list):
        retry_state.original_tasks = [json_safe_deep_copy(item) for item in value["original_tasks"] if isinstance(item, dict)]

    retry_scope = value.get("retry_scope")
    if retry_scope in {"refresh_routes", "reuse_hits_resynthesize"}:
        retry_state.retry_scope = retry_scope

    preserved_hits = value.get("preserved_hits")
    if isinstance(preserved_hits, list):
        retry_state.preserved_hits = [
            json_safe_deep_copy(item)
            for item in preserved_hits
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


def normalize_llm_call_observation(value: Any) -> tuple[list[LLMCallRecord] | None, list[str]]:
    """Validate canonical wire observations without interpreting provider metadata.

    One malformed entry makes the scope unknown: silently dropping it would make
    the remaining calls look like a fully observed turn. Raw diagnostics remain
    available in the caller's response envelope.
    """
    if value is None:
        return None, []
    if not isinstance(value, list):
        return None, ["debug.llm_calls must be a list or null"]
    calls: list[LLMCallRecord] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        try:
            payload = item.model_dump(mode="python") if isinstance(item, LLMCallRecord) else item
            calls.append(LLMCallRecord.model_validate(payload))
        except (TypeError, ValueError) as exc:
            errors.append(f"debug.llm_calls[{index}] invalid: {exc}")
    return (None if errors else calls), errors


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


def _parse_non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def parse_action_results(value: Any) -> ActionResults | None:
    if isinstance(value, ActionResults):
        return value
    if not isinstance(value, dict):
        return None

    payload: dict[str, Any] = {}
    slack_payload = value.get("slack_notify")
    if slack_payload is not None:
        payload["slack_notify"] = SlackDelivery.model_validate(
            slack_payload.model_dump() if isinstance(slack_payload, SlackDelivery) else slack_payload
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


def _parse_debug_diagnostics(value: Any) -> DebugDiagnostics:
    if isinstance(value, DebugDiagnostics):
        return value
    if not isinstance(value, dict):
        return DebugDiagnostics()

    observed_hits = (
        [
            dict(item)
            for item in value.get("observed_hits", [])
            if isinstance(item, dict)
        ]
        if isinstance(value.get("observed_hits"), list)
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

    llm_calls, usage_errors = normalize_llm_call_observation(value.get("llm_calls"))
    missing = [str(item) for item in value.get("missing_required_debug_fields", []) if str(item).strip()] if isinstance(value.get("missing_required_debug_fields"), list) else []
    if usage_errors:
        if "llm_calls" not in missing:
            missing.append("llm_calls")
        if observability_status != "failed":
            observability_status = "degraded"

    return DebugDiagnostics(
        schema_version=schema_version,
        observability_status=observability_status,  # type: ignore[arg-type]
        missing_required_debug_fields=missing,
        tool_calls=[str(item) for item in value.get("tool_calls", []) if str(item).strip()]
        if isinstance(value.get("tool_calls"), list)
        else [],
        tool_call_count=int(value.get("tool_call_count", 0) or 0),
        execution_evidence=value.get("execution_evidence"),
        llm_calls=llm_calls,
        errors=([str(item) for item in value.get("errors", []) if str(item).strip()]
                if isinstance(value.get("errors"), list) else []) + usage_errors,
        error_codes=parse_error_codes(value.get("error_codes")),
        validation_events=[
            str(item) for item in value.get("validation_events", []) if str(item).strip()
        ]
        if isinstance(value.get("validation_events"), list)
        else [],
        memory_compactions=[
            dict(item)
            for item in value.get("memory_compactions", [])
            if isinstance(item, dict)
        ]
        if isinstance(value.get("memory_compactions"), list)
        else [],
        planner_errors=[str(item) for item in value.get("planner_errors", []) if str(item).strip()]
        if isinstance(value.get("planner_errors"), list)
        else [],
        observed_hits=observed_hits,
        answer_provenance=value.get("answer_provenance"),
        retry_context=parse_retry_state(value.get("retry_context")) if value.get("retry_context") else None,
        retrieval_diagnostics=parse_retrieval_diagnostics(value.get("retrieval_diagnostics")),
        planner_diagnostics=parse_planner_diagnostic(value.get("planner_diagnostics")),
        latency_breakdown=dict(value.get("latency_breakdown"))
        if isinstance(value.get("latency_breakdown"), dict)
        else None,
        action_results=parse_action_results(value.get("action_results")),
    )


def parse_debug_payload(value: Any) -> DebugPayload:
    if isinstance(value, DebugPayload):
        return value
    if not isinstance(value, dict):
        raise ValueError("debug payload must be an object with route_decisions")
    decisions = validate_route_decisions(value.get("route_decisions"))
    return DebugPayload(
        **_parse_debug_diagnostics(value).model_dump(),
        route_decisions=decisions,
    )


def parse_debug_state(value: Any) -> DebugState:
    if isinstance(value, DebugState):
        return value
    if isinstance(value, DebugPayload):
        return DebugState.model_validate(value.model_dump(mode="json"))
    if not isinstance(value, dict):
        return DebugState()

    payload = _parse_debug_diagnostics(value).model_dump(mode="json")
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
