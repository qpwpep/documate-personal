from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import requests

from src.core.answer_schema.models import AnswerSection, ClaimItem
from src.core.contracts.boundary.debug import parse_action_results, parse_error_codes, parse_llm_calls, parse_model_usage_status, parse_token_usage
from src.core.contracts.boundary.planner import parse_planner_diagnostic
from src.core.contracts.boundary.retrieval import parse_retrieval_diagnostics
from src.core.contracts.debug import DEBUG_CRITICAL_FIELDS, DEBUG_REQUIRED_FIELDS, DEBUG_SCHEMA_VERSION
from src.core.contracts.debug import ActionResults, LLMCallMetadata, ModelUsageStatus, PlannerDiagnostic, RetrievalDiagnostic, TokenUsage
from src.core.evidence import EvidenceItem
from src.core.latency import LatencyBreakdownModel


_REQUEST_ID_PATTERN = re.compile(r"Request ID:\s*([^,\s]+)")


@dataclass(slots=True)
class ParsedResponseData:
    http_status: int = 0
    response_text: str = ""
    response_payload: dict[str, Any] | None = None
    response_evidence: list[EvidenceItem] = field(default_factory=list)
    observed_evidence: list[EvidenceItem] = field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = field(default_factory=list)
    planner_diagnostics: PlannerDiagnostic | None = None
    validator_reason: str | None = None
    validator_feedback: str | None = None
    response_trace: str | None = None
    request_id: str | None = None
    response_file_path: str | None = None
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    model_name: str | None = None
    models_used: list[str] = field(default_factory=list)
    model_usage_status: ModelUsageStatus = "missing_debug"
    tool_calls: list[str] = field(default_factory=list)
    tool_call_count: int = 0
    token_usage: TokenUsage | None = None
    llm_calls: list[LLMCallMetadata] = field(default_factory=list)
    error_codes: list[str] = field(default_factory=list)
    validation_events: list[str] = field(default_factory=list)
    edge_decisions: list[dict[str, Any]] = field(default_factory=list)
    planner_errors: list[str] = field(default_factory=list)
    debug_errors: list[str] = field(default_factory=list)
    runtime_errors: list[str] = field(default_factory=list)
    response_errors: list[str] = field(default_factory=list)
    debug_schema_version: int | None = None
    debug_observability_status: str | None = None
    missing_required_debug_fields: list[str] = field(default_factory=list)
    synthesis_mode: str | None = None
    action_results: ActionResults | None = None


def _build_error_message_from_response(response: requests.Response) -> str:
    body = response.text.strip()
    if len(body) > 300:
        body = body[:300] + " ..."
    return f"HTTP {response.status_code}: {body}"


def _parse_token_usage(raw_debug: dict[str, Any] | None) -> TokenUsage | None:
    if not raw_debug:
        return None
    return parse_token_usage(raw_debug.get("token_usage"))


def _parse_llm_calls(
    raw_items: Any,
    *,
    response_errors: list[str],
) -> list[LLMCallMetadata]:
    parsed: list[LLMCallMetadata] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append("debug.llm_calls must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            response_errors.append(f"debug.llm_calls[{index}] must be an object")
    return parse_llm_calls(raw_items)


def _parse_string_list(
    raw_items: Any,
    *,
    label: str,
    response_errors: list[str],
) -> list[str]:
    parsed: list[str] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append(f"{label} must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        text = str(item or "").strip()
        if not text:
            response_errors.append(f"{label}[{index}] must be a non-empty string")
            continue
        parsed.append(text)
    return parsed


def _parse_evidence_items(
    raw_items: Any,
    *,
    label: str,
    response_errors: list[str],
) -> list[EvidenceItem]:
    parsed: list[EvidenceItem] = []
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
            parsed.append(EvidenceItem.model_validate(item))
        except Exception as exc:
            response_errors.append(f"{label}[{index}] invalid: {exc}")
    return parsed


def _parse_retrieval_diagnostics(
    raw_items: Any,
    *,
    response_errors: list[str],
) -> list[RetrievalDiagnostic]:
    if raw_items is None:
        return []
    if not isinstance(raw_items, list):
        response_errors.append("debug.retrieval_diagnostics must be a list")
        return []
    return parse_retrieval_diagnostics(raw_items)


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
    return parse_planner_diagnostic(raw_item)


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


def _extract_request_id_from_response(response: Any, trace: str | None) -> str | None:
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        header_value = headers.get("x-request-id") or headers.get("X-Request-Id")
        if header_value:
            return str(header_value).strip()
    return _extract_request_id(trace)


def parse_claims_from_response_payload(response_payload: dict[str, Any] | None) -> list[ClaimItem]:
    if not isinstance(response_payload, dict):
        return []
    raw_claims = response_payload.get("claims")
    if not isinstance(raw_claims, list):
        return []
    claims: list[ClaimItem] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        try:
            claims.append(ClaimItem.model_validate(item))
        except Exception:
            continue
    return claims


def parse_sections_from_response_payload(response_payload: dict[str, Any] | None) -> list[AnswerSection]:
    if not isinstance(response_payload, dict):
        return []
    raw_sections = response_payload.get("sections")
    if not isinstance(raw_sections, list):
        return []
    sections: list[AnswerSection] = []
    for item in raw_sections:
        if not isinstance(item, dict):
            continue
        try:
            sections.append(AnswerSection.model_validate(item))
        except Exception:
            continue
    return sections


def parse_agent_response(response: requests.Response) -> ParsedResponseData:
    parsed = ParsedResponseData(http_status=response.status_code)
    if response.status_code != 200:
        parsed.runtime_errors.append(_build_error_message_from_response(response))
        return parsed

    try:
        body = response.json()
    except json.JSONDecodeError:
        parsed.response_errors.append("response is not valid JSON")
        body = {}

    if not isinstance(body, dict):
        parsed.response_errors.append("response body must be an object")
        return parsed

    parsed.response_trace = body.get("trace")
    parsed.request_id = _extract_request_id_from_response(response, parsed.response_trace)
    parsed.response_file_path = body.get("file_path")

    response_raw = body.get("response")
    if not isinstance(response_raw, dict):
        parsed.response_errors.append("response payload must be an object")
    else:
        parsed.response_payload = response_raw
        answer = response_raw.get("answer")
        if isinstance(answer, str):
            parsed.response_text = answer
        else:
            parsed.response_errors.append("response.answer must be a string")
        if not parsed.response_text.strip():
            parsed.response_errors.append("response.answer is empty")

        parsed.response_evidence = _parse_evidence_items(
            response_raw.get("evidence"),
            label="response.evidence",
            response_errors=parsed.response_errors,
        )

    debug_payload = body.get("debug")
    if isinstance(debug_payload, dict):
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
            except (TypeError, ValueError):
                parsed.response_errors.append("debug.schema_version must be an integer")
        if parsed.debug_schema_version is not None and parsed.debug_schema_version > DEBUG_SCHEMA_VERSION:
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
            parsed.response_errors.append(
                "critical debug fields missing: " + ", ".join(critical_missing_debug_fields)
            )
        parsed.tool_calls = [
            str(name)
            for name in (debug_payload.get("tool_calls") or [])
            if name
        ]
        try:
            parsed.tool_call_count = int(
                debug_payload.get("tool_call_count", len(parsed.tool_calls)) or len(parsed.tool_calls)
            )
        except (TypeError, ValueError):
            parsed.response_errors.append("debug.tool_call_count must be an integer")
            parsed.tool_call_count = len(parsed.tool_calls)
        latency_raw = debug_payload.get("latency_ms_server")
        if latency_raw is not None:
            try:
                parsed.latency_ms_server = int(latency_raw)
            except (TypeError, ValueError):
                parsed.response_errors.append("debug.latency_ms_server must be an integer")
        parsed.model_name = str(debug_payload.get("model_name")) if debug_payload.get("model_name") else None
        models_used_raw = debug_payload.get("models_used")
        if isinstance(models_used_raw, list):
            parsed.models_used = [str(name) for name in models_used_raw if name]
        elif parsed.model_name:
            parsed.models_used = [parsed.model_name]
        parsed.token_usage = _parse_token_usage(debug_payload)
        parsed.llm_calls = _parse_llm_calls(
            debug_payload.get("llm_calls"),
            response_errors=parsed.response_errors,
        )
        parsed.error_codes = parse_error_codes(debug_payload.get("error_codes"))
        parsed.validation_events = _parse_string_list(
            debug_payload.get("validation_events"),
            label="debug.validation_events",
            response_errors=parsed.response_errors,
        )
        edge_decisions_raw = debug_payload.get("edge_decisions")
        if edge_decisions_raw is None:
            parsed.edge_decisions = []
        elif not isinstance(edge_decisions_raw, list):
            parsed.response_errors.append("debug.edge_decisions must be a list")
        else:
            for index, item in enumerate(edge_decisions_raw):
                if not isinstance(item, dict):
                    parsed.response_errors.append(f"debug.edge_decisions[{index}] must be an object")
                    continue
                parsed.edge_decisions.append(dict(item))
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
        if not parsed.models_used and parsed.llm_calls:
            parsed.models_used = []
            for llm_call in parsed.llm_calls:
                response_metadata = llm_call.response_metadata
                model_name_candidate = response_metadata.get("model_name") or response_metadata.get("model")
                if model_name_candidate and model_name_candidate not in parsed.models_used:
                    parsed.models_used.append(str(model_name_candidate))
        parsed.model_usage_status = parse_model_usage_status(
            debug_payload.get("model_usage_status"),
            has_llm_usage=bool(
                parsed.llm_calls
                or parsed.models_used
                or parsed.model_name
                or (parsed.token_usage is not None and parsed.token_usage.total_tokens > 0)
            ),
        )
        parsed.observed_evidence = _parse_evidence_items(
            debug_payload.get("observed_evidence"),
            label="debug.observed_evidence",
            response_errors=parsed.response_errors,
        )
        raw_action_results = debug_payload.get("action_results")
        if raw_action_results is not None:
            parsed.action_results = parse_action_results(raw_action_results)
            if parsed.action_results is None:
                parsed.response_errors.append("debug.action_results is invalid")
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
    "parse_claims_from_response_payload",
    "parse_sections_from_response_payload",
]
