from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.core.answer_schema import AnswerResponse, ActionReceipt, export_answer_text
from src.core.contracts.boundary.debug import parse_error_codes, normalize_llm_call_observation
from src.core.contracts.boundary.planner import parse_planner_diagnostic
from src.core.contracts.boundary.retrieval import normalize_retrieval_diagnostic_observation
from src.core.contracts.debug import DEBUG_CRITICAL_FIELDS, DEBUG_REQUIRED_FIELDS, DEBUG_SCHEMA_VERSION
from src.core.contracts.debug import PlannerDiagnostic, RetrievalDiagnostic
from src.core.contracts.debug import ModelUsageStatus, TokenUsage
from src.core.contracts.usage import LLMCallRecord
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import RoutingDecision, validate_route_decisions
from src.core.contracts.tool_execution import ToolExecutionEvidence
from src.core.evidence import SearchHit
from src.core.latency import LatencyBreakdownModel
from ..result_models import EvidenceAssessment
from ..tool_policy import executed_tool_names


_REQUEST_ID_PATTERN = re.compile(r"Request ID:\s*([^,\s]+)")


@dataclass(slots=True)
class ParsedResponseData:
    model_name: str | None = None
    models_used: list[str] = field(default_factory=list)
    model_usage_status: ModelUsageStatus = "missing_debug"
    token_usage: TokenUsage | None = None
    http_status: int = 0
    response_text: str = ""
    response: AnswerResponse | None = None
    debug: dict[str, Any] | None = None
    answer_provenance: AnswerProvenance | None = None
    evidence_assessment: EvidenceAssessment | None = None
    observed_hits: list[SearchHit] = field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    validator_reason: str | None = None
    validator_feedback: str | None = None
    response_trace: str | None = None
    request_id: str | None = None
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    tool_calls: list[str] = field(default_factory=list)
    execution_evidence: ToolExecutionEvidence | dict[str, Any] | None = None
    tool_call_count: int = 0
    llm_calls: list[LLMCallRecord] | None = None
    error_codes: list[str] = field(default_factory=list)
    validation_events: list[str] = field(default_factory=list)
    route_decisions: list[RoutingDecision] = field(default_factory=list)
    memory_compactions: list[dict[str, Any]] = field(default_factory=list)
    planner_errors: list[str] = field(default_factory=list)
    debug_errors: list[str] = field(default_factory=list)
    runtime_errors: list[str] = field(default_factory=list)
    response_errors: list[str] = field(default_factory=list)
    debug_schema_version: int | None = None
    debug_observability_status: str | None = None
    missing_required_debug_fields: list[str] = field(default_factory=list)
    synthesis_mode: str | None = None
    actions: list[ActionReceipt] = field(default_factory=list)


def _parse_string_list(
    raw_items: Any,
    *,
    label: str,
    response_errors: list[str],
    allow_none: bool = True,
) -> list[str]:
    parsed: list[str] = []
    if raw_items is None and allow_none:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append(f"{label} must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        if not isinstance(item, str) or not item.strip():
            response_errors.append(f"{label}[{index}] must be a non-empty string")
            continue
        parsed.append(item.strip())
    return parsed


def _parse_search_hits(
    raw_items: Any,
    *,
    label: str,
    response_errors: list[str],
) -> list[SearchHit]:
    parsed: list[SearchHit] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append(f"{label} must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            response_errors.append(f"{label}[{index}] must be an object")
            continue
        try:
            parsed.append(SearchHit.model_validate(item))
        except Exception as exc:
            response_errors.append(f"{label}[{index}] invalid: {exc}")
    return parsed


def _parse_retrieval_diagnostics(
    raw_items: Any,
    *,
    response_errors: list[str],
) -> list[RetrievalDiagnostic]:
    parsed, issues = normalize_retrieval_diagnostic_observation(raw_items)
    response_errors.extend(issues)
    return parsed


def _parse_planner_diagnostics(
    raw_item: Any,
    *,
    response_errors: list[str],
) -> PlannerDiagnostic | None:
    if raw_item is None:
        return None
    if not isinstance(raw_item, dict):
        response_errors.append("debug.planner_diagnostics must be an object")
        return None
    try:
        return parse_planner_diagnostic(raw_item)
    except (TypeError, ValueError, OverflowError) as exc:
        response_errors.append(f"debug.planner_diagnostics invalid: {exc}")
        return None


def _parse_latency_breakdown(
    raw_item: Any,
    *,
    response_errors: list[str],
) -> LatencyBreakdownModel | None:
    if raw_item is None:
        return None
    if not isinstance(raw_item, dict):
        response_errors.append("debug.latency_breakdown must be an object")
        return None
    try:
        return LatencyBreakdownModel.model_validate(raw_item)
    except Exception as exc:
        response_errors.append(f"debug.latency_breakdown invalid: {exc}")
        return None


def _parse_validator_metadata(
    raw_item: Any,
    *,
    response_errors: list[str],
) -> tuple[str | None, str | None]:
    if raw_item is None:
        return None, None
    if not isinstance(raw_item, dict):
        response_errors.append("debug.retry_context must be an object")
        return None, None
    reason = str(raw_item.get("retry_reason") or "").strip() or None
    feedback = str(raw_item.get("retrieval_feedback") or "").strip() or None
    return reason, feedback


def _extract_request_id(trace: str | None) -> str | None:
    if not trace:
        return None
    match = _REQUEST_ID_PATTERN.search(str(trace))
    if not match:
        return None
    request_id = str(match.group(1)).strip()
    return request_id or None


def parse_agent_response(
    body: dict[str, Any],
    *,
    http_status: int = 200,
    request_id: str | None = None,
    validated_response: AnswerResponse | None = None,
) -> ParsedResponseData:
    """Parse the complete data object from an SSE final_response event."""
    parsed = ParsedResponseData(http_status=http_status)
    if not isinstance(body, dict):
        parsed.response_errors.append("response body must be an object")
        return parsed

    trace_raw = body.get("trace")
    if trace_raw is None or isinstance(trace_raw, str):
        parsed.response_trace = trace_raw
    else:
        parsed.response_errors.append("trace must be a string")
    parsed.request_id = request_id or _extract_request_id(parsed.response_trace)

    response_raw = body.get("response")
    if not isinstance(response_raw, dict):
        parsed.response_errors.append("response payload must be an object")
    else:
        try:
            parsed.response = validated_response if validated_response is not None else AnswerResponse.model_validate(response_raw)
            parsed.response_text = export_answer_text(parsed.response)
            parsed.actions = list(parsed.response.actions)
            if not parsed.response_text.strip() and not parsed.actions:
                parsed.response_errors.append("response.content is empty")
        except Exception as exc:
            parsed.response_errors.append(f"response invalid: {exc}")

    debug_payload = body.get("debug")
    if isinstance(debug_payload, dict):
        parsed.debug = debug_payload
        raw_provenance = debug_payload.get("answer_provenance")
        if raw_provenance is None:
            parsed.response_errors.append("debug.answer_provenance is missing")
        else:
            try:
                parsed.answer_provenance = AnswerProvenance.model_validate(raw_provenance)
            except (TypeError, ValueError) as exc:
                error = f"debug.answer_provenance invalid: {exc}"
                parsed.response_errors.append(error)
                parsed.evidence_assessment = EvidenceAssessment(status="invalid", errors=[error])
        present_debug_keys = {str(key) for key in debug_payload.keys()}
        parsed.missing_required_debug_fields = [
            field for field in DEBUG_REQUIRED_FIELDS if field not in present_debug_keys
        ]
        schema_version_raw = debug_payload.get("schema_version")
        if schema_version_raw is None:
            parsed.response_errors.append("debug.schema_version is missing")
        else:
            try:
                parsed.debug_schema_version = int(schema_version_raw)
            except (TypeError, ValueError, OverflowError):
                parsed.response_errors.append("debug.schema_version must be an integer")
        if parsed.debug_schema_version is not None and parsed.debug_schema_version != DEBUG_SCHEMA_VERSION:
            parsed.response_errors.append(f"debug.schema_version must be {DEBUG_SCHEMA_VERSION}")
        observability_status_raw = debug_payload.get("observability_status")
        if observability_status_raw is None:
            parsed.response_errors.append("debug.observability_status is missing")
        else:
            normalized_observability_status = str(observability_status_raw).strip().lower()
            if normalized_observability_status in {"ok", "degraded", "failed"}:
                parsed.debug_observability_status = normalized_observability_status
            else:
                parsed.response_errors.append(
                    "debug.observability_status must be one of ok/degraded/failed"
                )
        if debug_payload.get("missing_required_debug_fields") is None:
            parsed.response_errors.append("debug.missing_required_debug_fields is missing")
        else:
            self_reported_missing_fields = _parse_string_list(
                debug_payload.get("missing_required_debug_fields"),
                label="debug.missing_required_debug_fields",
                response_errors=parsed.response_errors,
            )
            for field_name in self_reported_missing_fields:
                if field_name not in parsed.missing_required_debug_fields:
                    parsed.missing_required_debug_fields.append(field_name)
        critical_missing_debug_fields = [
            field
            for field in parsed.missing_required_debug_fields
            if field in DEBUG_CRITICAL_FIELDS
        ]
        if critical_missing_debug_fields:
            parsed.debug_observability_status = "failed"
            parsed.response_errors.append(
                "critical debug fields missing: " + ", ".join(critical_missing_debug_fields)
            )
        parsed.tool_calls = _parse_string_list(
            debug_payload.get("tool_calls"), label="debug.tool_calls",
            response_errors=parsed.response_errors, allow_none=False,
        )
        raw_execution = debug_payload.get("execution_evidence")
        if isinstance(raw_execution, dict):
            # Retain malformed envelopes: valid started events remain evidence
            # even when the complete journal fails structural validation.
            parsed.execution_evidence = raw_execution
        elif raw_execution is not None:
            parsed.response_errors.append("debug.execution_evidence must be an object")
        executed = executed_tool_names(parsed.execution_evidence)
        if executed is not None:
            parsed.tool_calls = executed
        try:
            parsed.tool_call_count = int(
                debug_payload.get("tool_call_count", len(parsed.tool_calls)) or len(parsed.tool_calls)
            )
        except (TypeError, ValueError, OverflowError):
            parsed.response_errors.append("debug.tool_call_count must be an integer")
            parsed.tool_call_count = len(parsed.tool_calls)
        latency_raw = debug_payload.get("latency_ms_server")
        if latency_raw is not None:
            try:
                parsed.latency_ms_server = int(latency_raw)
            except (TypeError, ValueError, OverflowError):
                parsed.response_errors.append("debug.latency_ms_server must be an integer")
        parsed.llm_calls, usage_errors = normalize_llm_call_observation(debug_payload.get("llm_calls"))
        parsed.response_errors.extend(usage_errors)
        parsed.error_codes = parse_error_codes(debug_payload.get("error_codes"))
        parsed.validation_events = _parse_string_list(
            debug_payload.get("validation_events"),
            label="debug.validation_events",
            response_errors=parsed.response_errors,
        )
        try:
            parsed.route_decisions = validate_route_decisions(debug_payload.get("route_decisions"))
        except ValueError as exc:
            parsed.response_errors.append(f"debug.route_decisions invalid: {exc}")
            parsed.debug_observability_status = "failed"
            if "route_decisions" not in parsed.missing_required_debug_fields:
                parsed.missing_required_debug_fields.append("route_decisions")
            if "DEBUG_NORMALIZATION_FAILED" not in parsed.error_codes:
                parsed.error_codes.append("DEBUG_NORMALIZATION_FAILED")
        memory_compactions_raw = debug_payload.get("memory_compactions")
        if not isinstance(memory_compactions_raw, list):
            parsed.response_errors.append("debug.memory_compactions must be a list")
        else:
            for index, item in enumerate(memory_compactions_raw):
                if not isinstance(item, dict):
                    parsed.response_errors.append(f"debug.memory_compactions[{index}] must be an object")
                    continue
                parsed.memory_compactions.append(dict(item))
        parsed.debug_errors = _parse_string_list(
            debug_payload.get("errors"),
            label="debug.errors",
            response_errors=parsed.response_errors,
        )
        parsed.planner_errors = _parse_string_list(
            debug_payload.get("planner_errors"),
            label="debug.planner_errors",
            response_errors=parsed.response_errors,
        )
        parsed.observed_hits = _parse_search_hits(
            debug_payload.get("observed_hits"),
            label="debug.observed_hits",
            response_errors=parsed.response_errors,
        )
        parsed.retrieval_diagnostics = _parse_retrieval_diagnostics(
            debug_payload.get("retrieval_diagnostics"),
            response_errors=parsed.response_errors,
        )
        parsed.planner_diagnostics = _parse_planner_diagnostics(
            debug_payload.get("planner_diagnostics"),
            response_errors=parsed.response_errors,
        )
        parsed.validator_reason, parsed.validator_feedback = _parse_validator_metadata(
            debug_payload.get("retry_context"),
            response_errors=parsed.response_errors,
        )
        parsed.latency_breakdown = _parse_latency_breakdown(
            debug_payload.get("latency_breakdown"),
            response_errors=parsed.response_errors,
        )
        if parsed.latency_breakdown is not None and parsed.latency_breakdown.synthesis_attempts:
            parsed.synthesis_mode = parsed.latency_breakdown.synthesis_attempts[0].mode
    else:
        parsed.response_errors.append("debug payload is missing (include_debug=true expected)")

    return parsed


__all__ = [
    "ParsedResponseData",
    "parse_agent_response",
]
