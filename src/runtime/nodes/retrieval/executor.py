from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from src.core.contracts import RetrievalDiagnostic
from src.core.contracts.routes import route_for_tool
from src.core.evidence import evidence_to_dicts, parse_evidence_payload
from src.core.latency import elapsed_ms, make_retrieval_route_latency_event
from src.core.planner_schema import RetrievalTask
from src.infra.tools._common import build_retrieval_payload
from src.infra.tools.docs_search import infer_docs_query_hint
from src.infra.tools.docs_search.serialization import filter_evidence_to_domains


@dataclass(slots=True)
class RetrievalTaskResult:
    index: int
    tool_name: str
    payload: Any
    evidence: list[dict[str, Any]]
    diagnostic: RetrievalDiagnostic
    errors: list[str]
    latency_trace: dict[str, Any]


def _non_negative_int(value: Any, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, default)


def normalize_retrieval_diagnostic(
    raw_payload: Any,
    *,
    tool_name: str,
    route: str,
    query: str,
    attempt: int,
    evidence_count: int,
) -> RetrievalDiagnostic:
    diagnostics: dict[str, Any] = {}
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("diagnostics"), dict):
        diagnostics = dict(raw_payload.get("diagnostics") or {})

    try:
        diagnostic_attempt = int(diagnostics.get("attempt") or attempt)
    except (TypeError, ValueError):
        diagnostic_attempt = attempt

    status = str(diagnostics.get("status") or ("success" if evidence_count > 0 else "no_result"))
    message = str(diagnostics.get("message") or "")
    warnings = diagnostics.get("warnings")
    if not isinstance(warnings, list):
        warnings = []

    return RetrievalDiagnostic(
        tool=str(diagnostics.get("tool") or tool_name),
        route=str(diagnostics.get("route") or route or route_for_tool(tool_name)),
        status=status,
        message=message,
        error_code=diagnostics.get("error_code"),
        query=str(diagnostics.get("query") or query),
        attempt=diagnostic_attempt,
        evidence_count=int(diagnostics.get("evidence_count", evidence_count) or evidence_count),
        metric=str(diagnostics.get("metric") or ""),
        score_direction=str(diagnostics.get("score_direction") or ""),  # type: ignore[arg-type]
        normalized_score=diagnostics.get("normalized_score"),
        raw_score=diagnostics.get("raw_score"),
        result_count=int(diagnostics.get("result_count", evidence_count) or evidence_count),
        provider_result_count=_non_negative_int(diagnostics.get("provider_result_count", 0), default=0),
        filtered_invalid_url_count=_non_negative_int(diagnostics.get("filtered_invalid_url_count", 0), default=0),
        filtered_path_prefix_count=_non_negative_int(diagnostics.get("filtered_path_prefix_count", 0), default=0),
        filtered_cross_domain_count=_non_negative_int(diagnostics.get("filtered_cross_domain_count", 0), default=0),
        filtered_http_error_count=_non_negative_int(diagnostics.get("filtered_http_error_count", 0), default=0),
        filtered_redirect_policy_count=_non_negative_int(diagnostics.get("filtered_redirect_policy_count", 0), default=0),
        filtered_url_request_failed_count=_non_negative_int(diagnostics.get("filtered_url_request_failed_count", 0), default=0),
        filtered_identifier_mismatch_count=_non_negative_int(diagnostics.get("filtered_identifier_mismatch_count", 0), default=0),
        validated_url_count=_non_negative_int(diagnostics.get("validated_url_count", 0), default=0),
        final_evidence_count=_non_negative_int(diagnostics.get("final_evidence_count", evidence_count), default=evidence_count),
        warnings=[str(item).strip() for item in warnings if str(item).strip()],
    )


def collect_retrieval_result(
    *,
    raw_payload: Any,
    tool_name: str,
    route: str,
    query: str,
    attempt: int,
    local_errors: list[str],
) -> tuple[list[dict[str, Any]], RetrievalDiagnostic]:
    parsed_items = parse_evidence_payload(raw_payload, context=f"tool:{tool_name}", errors=local_errors)
    payload_dicts = evidence_to_dicts(parsed_items)
    warnings: list[str] = []
    filtered_cross_domain_count = 0
    if route == "docs":
        hinted_domains = []
        if query_hint := infer_docs_query_hint(query):
            _library_name, hinted_domains, _fallback_queries = query_hint
        if hinted_domains:
            pre_filter_count = len(payload_dicts)
            filtered_payload_dicts = filter_evidence_to_domains(
                payload_dicts,
                allowed_domains=hinted_domains,
            )
            if len(filtered_payload_dicts) != len(payload_dicts):
                filtered_cross_domain_count = pre_filter_count - len(filtered_payload_dicts)
                warnings.append("cross_library_domain_filtered")
                payload_dicts = filtered_payload_dicts

    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("diagnostics"), dict):
        diagnostics = raw_payload["diagnostics"]
        diagnostics["warnings"] = sorted(set([*diagnostics.get("warnings", []), *warnings]))
        if route == "docs":
            diagnostics["filtered_cross_domain_count"] = _non_negative_int(
                diagnostics.get("filtered_cross_domain_count", 0),
                default=0,
            ) + filtered_cross_domain_count
            diagnostics["final_evidence_count"] = len(payload_dicts)
        if route == "docs" and filtered_cross_domain_count > 0:
            diagnostics["status"] = "success" if payload_dicts else "no_result"
            if not payload_dicts and not str(diagnostics.get("message") or "").strip():
                diagnostics["message"] = "no official documentation evidence found"
            diagnostics["normalized_score"] = max(
                (float(item["score"]) for item in payload_dicts if item.get("score") is not None),
                default=None,
            )
            diagnostics["raw_score"] = diagnostics.get("normalized_score")
            diagnostics["result_count"] = len(payload_dicts)
            diagnostics["evidence_count"] = len(payload_dicts)

    diagnostic = normalize_retrieval_diagnostic(
        raw_payload,
        tool_name=tool_name,
        route=route,
        query=query,
        attempt=attempt,
        evidence_count=len(payload_dicts),
    )
    if diagnostic.status in {"error", "unavailable"} and diagnostic.message:
        local_errors.append(f"{tool_name}: {diagnostic.message}")
    return payload_dicts, diagnostic


def execute_retrieval_task(
    *,
    index: int,
    task: RetrievalTask,
    tool_name: str,
    route: str,
    invoke_tool: Any,
    attempt: int,
) -> RetrievalTaskResult:
    local_errors: list[str] = []
    started = time.perf_counter()
    try:
        payload = invoke_tool(task)
    except Exception as exc:
        payload = {
            "evidence": [],
            "diagnostics": {
                "tool": tool_name,
                "route": route,
                "status": "error",
                "message": f"tool invocation failed ({exc})",
                "error_code": "RETRIEVAL_DOCS_FAILED" if route == "docs" else None,
                "query": task.query,
            },
        }

    latency_ms = elapsed_ms(started, time.perf_counter())
    payload_dicts, diagnostic = collect_retrieval_result(
        raw_payload=payload,
        tool_name=tool_name,
        route=route,
        query=task.query,
        attempt=attempt,
        local_errors=local_errors,
    )
    return RetrievalTaskResult(
        index=index,
        tool_name=tool_name,
        payload=payload,
        evidence=payload_dicts,
        diagnostic=diagnostic,
        errors=local_errors,
        latency_trace=make_retrieval_route_latency_event(
            route=route,
            tool=tool_name,
            attempt=attempt,
            latency_ms=latency_ms,
            status=diagnostic.status,
        ),
    )


def build_reused_retrieval_task_result(
    *,
    index: int,
    task: RetrievalTask,
    tool_name: str,
    route: str,
    attempt: int,
    preserved_evidence: list[dict[str, Any]],
    preserved_diagnostics: list[RetrievalDiagnostic],
) -> RetrievalTaskResult:
    route_evidence = [
        dict(item)
        for item in preserved_evidence
        if route_for_tool(str(item.get("tool") or "")) == route
    ]
    route_diagnostic = next(
        (
            item.model_copy(deep=True)
            for item in preserved_diagnostics
            if str(item.route or "").strip() == route
        ),
        RetrievalDiagnostic(
            tool=tool_name,
            route=route,
            status="success" if route_evidence else "no_result",
            message="reused previous successful retrieval result",
            query=task.query,
            attempt=attempt,
        ),
    )
    diagnostic = route_diagnostic.model_copy(
        update={
            "tool": str(route_diagnostic.tool or tool_name),
            "route": str(route_diagnostic.route or route),
            "query": str(route_diagnostic.query or task.query),
            "attempt": attempt,
        }
    )

    payload = build_retrieval_payload(
        tool=tool_name,
        route=route,  # type: ignore[arg-type]
        query=task.query,
        evidence=route_evidence,
        status="success" if route_evidence else "no_result",
        message="reused previous successful retrieval result" if route_evidence else "no reused evidence found",
        normalized_score=diagnostic.normalized_score,
        raw_score=diagnostic.raw_score,
        result_count=diagnostic.result_count,
        provider_result_count=diagnostic.provider_result_count,
        filtered_invalid_url_count=diagnostic.filtered_invalid_url_count,
        filtered_path_prefix_count=diagnostic.filtered_path_prefix_count,
        filtered_cross_domain_count=diagnostic.filtered_cross_domain_count,
        filtered_http_error_count=diagnostic.filtered_http_error_count,
        filtered_redirect_policy_count=diagnostic.filtered_redirect_policy_count,
        filtered_url_request_failed_count=diagnostic.filtered_url_request_failed_count,
        filtered_identifier_mismatch_count=diagnostic.filtered_identifier_mismatch_count,
        validated_url_count=diagnostic.validated_url_count,
        final_evidence_count=diagnostic.final_evidence_count,
        metric=diagnostic.metric or None,
        score_direction=diagnostic.score_direction or None,
        warnings=diagnostic.warnings,
    )
    return RetrievalTaskResult(
        index=index,
        tool_name=tool_name,
        payload=payload,
        evidence=route_evidence,
        diagnostic=diagnostic,
        errors=[],
        latency_trace=make_retrieval_route_latency_event(
            route=route,
            tool=tool_name,
            attempt=attempt,
            latency_ms=0,
            status=diagnostic.status,
        ),
    )
