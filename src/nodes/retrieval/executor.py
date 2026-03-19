from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any

from ...contracts import RetrievalDiagnostic
from ...contracts.routes import route_for_tool
from ...evidence import evidence_to_dicts, parse_evidence_payload
from ...latency import elapsed_ms, make_retrieval_route_latency_event
from ...planner_schema import RetrievalTask
from ...tools._common import build_retrieval_payload


@dataclass(slots=True)
class RetrievalTaskResult:
    index: int
    tool_name: str
    payload: Any
    evidence: list[dict[str, Any]]
    diagnostic: RetrievalDiagnostic
    errors: list[str]
    latency_trace: dict[str, Any]


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

    return RetrievalDiagnostic(
        tool=str(diagnostics.get("tool") or tool_name),
        route=str(diagnostics.get("route") or route or route_for_tool(tool_name)),
        status=status,
        message=message,
        query=str(diagnostics.get("query") or query),
        attempt=diagnostic_attempt,
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
