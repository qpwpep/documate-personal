from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Any

from ...contracts import RetrievalDiagnostic
from ...contracts.routes import route_for_tool
from ...evidence import evidence_to_dicts, parse_evidence_payload
from ...latency import elapsed_ms, make_retrieval_route_latency_event
from ...planner_schema import RetrievalTask
from ...tools.docs_search import filter_evidence_to_domains, infer_docs_query_hint
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


def _normalize_backend_score(value: Any) -> tuple[float | None, float | None, str | None]:
    try:
        if value is None or value == "":
            return None, None, None
        raw_score = float(value)
    except (TypeError, ValueError):
        return None, None, "non_numeric_score"
    if not math.isfinite(raw_score):
        return None, None, "non_finite_score"
    if 0.0 <= raw_score <= 1.0:
        return raw_score, raw_score, None
    if raw_score > 1.0:
        return raw_score / (1.0 + raw_score), raw_score, "normalized_score_gt_1"
    return 0.0, raw_score, "normalized_negative_score"


def _normalize_cell_id(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _normalize_payload_evidence(
    payload_dicts: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], float | None, float | None]:
    warnings: list[str] = []
    normalized_items: list[dict[str, Any]] = []
    normalized_scores: list[float] = []
    raw_scores: list[float] = []
    for item in payload_dicts:
        normalized = dict(item)
        score, raw_score, warning = _normalize_backend_score(normalized.get("score"))
        if score is not None:
            normalized["score"] = score
            normalized_scores.append(score)
        if raw_score is not None:
            normalized["raw_score"] = raw_score
            raw_scores.append(raw_score)
        cell_id = _normalize_cell_id(normalized.get("cell_id"))
        if cell_id is not None:
            normalized["cell_id"] = cell_id
        if warning:
            warnings.append(warning)
        normalized_items.append(normalized)
    relevance_score = max(normalized_scores) if normalized_scores else None
    raw_relevance_score = max(raw_scores) if raw_scores else None
    return normalized_items, sorted(set(warnings)), relevance_score, raw_relevance_score


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
        query=str(diagnostics.get("query") or query),
        attempt=diagnostic_attempt,
        evidence_count=int(diagnostics.get("evidence_count", evidence_count) or evidence_count),
        avg_score=diagnostics.get("avg_score"),
        max_score=diagnostics.get("max_score"),
        normalized_score=diagnostics.get("normalized_score", diagnostics.get("relevance_score")),
        relevance_score=diagnostics.get("relevance_score"),
        raw_relevance_score=diagnostics.get("raw_relevance_score"),
        result_count=int(diagnostics.get("result_count", evidence_count) or evidence_count),
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
    payload_dicts, warnings, relevance_score, raw_relevance_score = _normalize_payload_evidence(
        evidence_to_dicts(parsed_items)
    )
    domain_filtered = False
    if route == "docs":
        hinted_domains = []
        if query_hint := infer_docs_query_hint(query):
            _library_name, hinted_domains, _fallback_queries = query_hint
        if hinted_domains:
            filtered_payload_dicts = filter_evidence_to_domains(
                payload_dicts,
                allowed_domains=hinted_domains,
            )
            if len(filtered_payload_dicts) != len(payload_dicts):
                warnings.append("cross_library_domain_filtered")
                payload_dicts = filtered_payload_dicts
                domain_filtered = True
                relevance_score = max(
                    (float(item["score"]) for item in payload_dicts if item.get("score") is not None),
                    default=None,
                )
                raw_relevance_score = max(
                    (float(item.get("raw_score")) for item in payload_dicts if item.get("raw_score") is not None),
                    default=None,
                )
    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("diagnostics"), dict):
        diagnostics = raw_payload["diagnostics"]
        diagnostics["warnings"] = sorted(set([*diagnostics.get("warnings", []), *warnings]))
        if domain_filtered:
            filtered_scores = [float(item["score"]) for item in payload_dicts if item.get("score") is not None]
            diagnostics["avg_score"] = (
                (sum(filtered_scores) / len(filtered_scores)) if filtered_scores else None
            )
            diagnostics["max_score"] = max(filtered_scores) if filtered_scores else None
            diagnostics["status"] = "success" if payload_dicts else "no_result"
            if not payload_dicts and not str(diagnostics.get("message") or "").strip():
                diagnostics["message"] = "no official documentation evidence found"
            diagnostics["normalized_score"] = relevance_score
            diagnostics["relevance_score"] = relevance_score
            diagnostics["raw_relevance_score"] = raw_relevance_score
            diagnostics["result_count"] = len(payload_dicts)
            diagnostics["evidence_count"] = len(payload_dicts)
        else:
            diagnostics["relevance_score"] = diagnostics.get("relevance_score", relevance_score)
            diagnostics["raw_relevance_score"] = diagnostics.get("raw_relevance_score", raw_relevance_score)
            diagnostics["result_count"] = diagnostics.get("result_count", len(payload_dicts))
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
        relevance_score=diagnostic.relevance_score,
        raw_relevance_score=diagnostic.raw_relevance_score,
        result_count=diagnostic.result_count,
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
