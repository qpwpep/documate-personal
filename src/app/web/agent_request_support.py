from __future__ import annotations

from src.core.contracts import SessionMetadata
from src.core.contracts.boundary.debug import parse_action_results, parse_error_codes, normalize_llm_call_observation, parse_retry_state
from src.core.contracts.boundary.planner import parse_planner_diagnostic
from src.core.contracts.boundary.retrieval import normalize_retrieval_diagnostic_observation
from src.core.contracts.debug import DEBUG_CRITICAL_FIELDS, DEBUG_REQUIRED_FIELDS, DEBUG_SCHEMA_VERSION
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import validate_route_decisions
from src.core.contracts.tool_execution import ToolExecutionEvidence
from src.core.latency import LatencyBreakdownModel
from src.app.web.schemas import AgentDebugInfo, AgentRequest
from src.core.evidence import SearchHit
from src.core.llm_errors import LLMDiagnostic


def normalize_debug_info(
    raw_debug: dict | None,
    latency_ms_server: int | None,
    *,
    answer_expected: bool = True,
) -> AgentDebugInfo:
    debug = raw_debug or {}
    present_keys = {str(key) for key in debug.keys()} if isinstance(debug, dict) else set()
    missing_required_debug_fields = [
        field for field in DEBUG_REQUIRED_FIELDS if field not in present_keys
        and (answer_expected or field != "answer_provenance")
    ]
    self_reported_missing_fields = [
        str(field_name)
        for field_name in (debug.get("missing_required_debug_fields") or [])
        if str(field_name).strip()
    ]
    for field_name in self_reported_missing_fields:
        if field_name == "answer_provenance" and not answer_expected and debug.get("answer_provenance") is None:
            continue
        if field_name not in missing_required_debug_fields:
            missing_required_debug_fields.append(field_name)
    critical_missing = [field for field in missing_required_debug_fields if field in DEBUG_CRITICAL_FIELDS]
    tool_calls = debug.get("tool_calls") or []
    raw_tool_call_count = debug.get("tool_call_count", len(tool_calls))
    tool_call_count = int(raw_tool_call_count) if raw_tool_call_count is not None else len(tool_calls)
    errors = debug.get("errors") or []
    error_codes = parse_error_codes(debug.get("error_codes") or [])
    execution_evidence = debug.get("execution_evidence")
    if execution_evidence is not None:
        try:
            execution_evidence = ToolExecutionEvidence.model_validate(execution_evidence)
        except (TypeError, ValueError):
            # Keep raw events: a malformed envelope must not erase a known start.
            if not isinstance(execution_evidence, dict):
                execution_evidence = None
            missing_required_debug_fields.append("execution_evidence")
            critical_missing.append("execution_evidence")
            if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
                error_codes.append("DEBUG_NORMALIZATION_FAILED")
    if isinstance(execution_evidence, ToolExecutionEvidence):
        started_tools = [event.tool_name for event in execution_evidence.events if event.phase == "started"]
        if tool_calls != started_tools or tool_call_count != len(started_tools):
            missing_required_debug_fields.append("execution_evidence")
            critical_missing.append("execution_evidence")
            if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
                error_codes.append("DEBUG_NORMALIZATION_FAILED")
    answer_provenance = None
    if debug.get("answer_provenance") is not None:
        try:
            answer_provenance = AnswerProvenance.model_validate(debug["answer_provenance"])
        except (TypeError, ValueError):
            if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
                error_codes.append("DEBUG_NORMALIZATION_FAILED")
    if answer_provenance is None and (answer_expected or debug.get("answer_provenance") is not None):
        if "answer_provenance" not in missing_required_debug_fields:
            missing_required_debug_fields.append("answer_provenance")
        if "answer_provenance" not in critical_missing:
            critical_missing.append("answer_provenance")
    validation_events_raw = debug.get("validation_events") or []
    try:
        route_decisions = validate_route_decisions(debug.get("route_decisions"))
    except ValueError as exc:
        route_decisions = []
        if "route_decisions" not in missing_required_debug_fields:
            missing_required_debug_fields.append("route_decisions")
        if "route_decisions" not in critical_missing:
            critical_missing.append("route_decisions")
        if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
            error_codes.append("DEBUG_NORMALIZATION_FAILED")
        errors = [*errors, f"route_decisions invalid: {exc}"]
    memory_compactions_raw = debug.get("memory_compactions")
    planner_errors_raw = debug.get("planner_errors") or []
    observed_hits_raw = debug.get("observed_hits") or []
    raw_llm_calls = debug.get("llm_calls")
    llm_diagnostics = []
    try:
        raw_llm_diagnostics = debug.get("llm_diagnostics", [])
        if not isinstance(raw_llm_diagnostics, list):
            raise ValueError("llm_diagnostics must be a list")
        llm_diagnostics = [LLMDiagnostic.model_validate(item) for item in raw_llm_diagnostics]
    except (TypeError, ValueError):
        missing_required_debug_fields.append("llm_diagnostics")
        critical_missing.append("llm_diagnostics")
        if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
            error_codes.append("DEBUG_NORMALIZATION_FAILED")
        errors = [*errors, "llm_diagnostics has an invalid structure"]

    observed_hits: list[SearchHit] = []
    if isinstance(observed_hits_raw, list):
        for item in observed_hits_raw:
            if not isinstance(item, dict):
                continue
            try:
                observed_hits.append(SearchHit.model_validate(item))
            except Exception:
                continue

    retry_context = parse_retry_state(debug.get("retry_context")) if debug.get("retry_context") else None
    retrieval_diagnostics, retrieval_issues = normalize_retrieval_diagnostic_observation(debug.get("retrieval_diagnostics"))
    if retrieval_issues:
        if "retrieval_diagnostics" not in missing_required_debug_fields:
            missing_required_debug_fields.append("retrieval_diagnostics")
        if "retrieval_diagnostics" not in critical_missing:
            critical_missing.append("retrieval_diagnostics")
        if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
            error_codes.append("DEBUG_NORMALIZATION_FAILED")
        errors = [*errors, *retrieval_issues]
    planner_diagnostics = parse_planner_diagnostic(debug.get("planner_diagnostics"))
    llm_calls, usage_errors = normalize_llm_call_observation(raw_llm_calls)
    if usage_errors:
        if "llm_calls" not in missing_required_debug_fields:
            missing_required_debug_fields.append("llm_calls")
        if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
            error_codes.append("DEBUG_NORMALIZATION_FAILED")
        errors = [*errors, *usage_errors]
    action_results = parse_action_results(debug.get("action_results"))

    latency_breakdown = None
    raw_latency_breakdown = debug.get("latency_breakdown")
    if isinstance(raw_latency_breakdown, dict):
        latency_payload = dict(raw_latency_breakdown)
        latency_payload["server_total_ms"] = latency_ms_server
        try:
            latency_breakdown = LatencyBreakdownModel.model_validate(latency_payload)
        except Exception:
            latency_breakdown = None
            if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
                error_codes.append("DEBUG_NORMALIZATION_FAILED")  # type: ignore[arg-type]

    schema_version_raw = debug.get("schema_version", DEBUG_SCHEMA_VERSION)
    try:
        schema_version = int(schema_version_raw)
    except (TypeError, ValueError):
        schema_version = DEBUG_SCHEMA_VERSION
    raw_observability_status = str(debug.get("observability_status") or "").strip().lower()
    if raw_observability_status not in {"ok", "degraded", "failed"}:
        raw_observability_status = "ok"

    return AgentDebugInfo(
        schema_version=schema_version,
        observability_status=(
            "failed"
            if critical_missing or raw_observability_status == "failed"
            else (
                "degraded"
                if raw_observability_status == "degraded" or missing_required_debug_fields
                else raw_observability_status
            )
        ),
        missing_required_debug_fields=missing_required_debug_fields,
        tool_calls=[str(name) for name in tool_calls if name],
        tool_call_count=tool_call_count,
        execution_evidence=execution_evidence,
        latency_ms_server=latency_ms_server,
        latency_breakdown=latency_breakdown,
        llm_calls=llm_calls,
        llm_diagnostics=llm_diagnostics,
        errors=[str(error) for error in errors if error],
        error_codes=error_codes,
        validation_events=[
            str(event) for event in validation_events_raw if str(event).strip()
        ]
        if isinstance(validation_events_raw, list)
        else [],
        route_decisions=route_decisions,
        memory_compactions=[
            dict(item)
            for item in memory_compactions_raw
            if isinstance(item, dict)
        ]
        if isinstance(memory_compactions_raw, list)
        else [],
        planner_errors=[str(error) for error in planner_errors_raw if error]
        if isinstance(planner_errors_raw, list)
        else [],
        observed_hits=observed_hits,
        answer_provenance=answer_provenance,
        retry_context=retry_context,
        retrieval_diagnostics=retrieval_diagnostics,
        planner_diagnostics=planner_diagnostics,
        action_results=action_results,
    )


def build_session_metadata_snapshot(request_data: AgentRequest) -> SessionMetadata:
    return SessionMetadata(slack_recipient=request_data.slack_recipient)
