from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any

from src.core.contracts import RetrievalDiagnostic
from src.core.contracts.routes import route_for_tool
from src.core.evidence import parse_search_hits
from src.core.latency import elapsed_ms, make_retrieval_route_latency_event
from src.core.planner_schema import RetrievalTask
from src.infra.tools.docs_search import infer_docs_query_hint
from src.infra.tools.docs_search.serialization import filter_hits_to_domains


@dataclass(slots=True)
class RetrievalTaskResult:
    index: int
    tool_name: str
    payload: Any
    hits: list[dict[str, Any]]
    diagnostic: RetrievalDiagnostic
    errors: list[str]
    latency_trace: dict[str, Any]


def retrieval_fingerprint(task: RetrievalTask) -> str:
    """Request identity within a turn, independent of a model-assigned task ID."""
    payload = {"route": task.route, "query": " ".join(task.query.split()).casefold(),
               "requirement": task.requirement.model_dump(mode="json"), "k": task.k}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


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
        requirement_id=str(diagnostics.get("requirement_id") or ""),
        answerability=diagnostics.get("answerability") if diagnostics.get("answerability") in {"covered", "partial", "missing", "unknown"} else "unknown",
        missing_requirements=[str(item) for item in diagnostics.get("missing_requirements", [])],
        candidate_count=_non_negative_int(diagnostics.get("candidate_count", evidence_count)),
        attempted_queries=[str(item) for item in diagnostics.get("attempted_queries", [])],
        request_fingerprint=str(diagnostics.get("request_fingerprint") or ""),
        reused=bool(diagnostics.get("reused", False)),
        attempt=diagnostic_attempt,
        evidence_count=evidence_count,
        metric=str(diagnostics.get("metric") or ""),
        score_direction=str(diagnostics.get("score_direction") or ""),  # type: ignore[arg-type]
        normalized_score=diagnostics.get("normalized_score"),
        raw_score=diagnostics.get("raw_score"),
        provider_ms=_non_negative_int(diagnostics.get("provider_ms", 0), default=0),
        url_validation_ms=_non_negative_int(diagnostics.get("url_validation_ms", 0), default=0),
        post_filter_ms=_non_negative_int(diagnostics.get("post_filter_ms", 0), default=0),
        include_raw_content_requested=bool(diagnostics.get("include_raw_content_requested", False)),
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
        final_evidence_count=evidence_count,
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
    task: RetrievalTask | None = None,
) -> tuple[list[dict[str, Any]], RetrievalDiagnostic]:
    try:
        parsed_items = parse_search_hits(raw_payload, errors=local_errors)
    except (TypeError, ValueError) as exc:
        local_errors.append(f"tool:{tool_name}: invalid search hits ({exc})")
        parsed_items = []
    warnings: list[str] = []
    filtered_cross_domain_count = 0
    if route == "docs" and not (task and task.requirement.specified):
        hinted_domains = []
        if query_hint := infer_docs_query_hint(query):
            _library_name, hinted_domains, _fallback_queries = query_hint
        if hinted_domains:
            pre_filter_count = len(parsed_items)
            filtered_items = parse_search_hits(filter_hits_to_domains(
                [hit.model_dump(mode="json") for hit in parsed_items],
                allowed_domains=hinted_domains,
            ))
            if len(filtered_items) != len(parsed_items):
                filtered_cross_domain_count = pre_filter_count - len(filtered_items)
                warnings.append("cross_library_domain_filtered")
                parsed_items = filtered_items

    if task is not None:
        parsed_items = [item.model_copy(update={"requirement_id": task.requirement_id}) for item in parsed_items]
    payload_dicts = [item.model_dump(mode="json") for item in parsed_items]

    if isinstance(raw_payload, dict) and isinstance(raw_payload.get("diagnostics"), dict):
        diagnostics = raw_payload["diagnostics"]
        if task is not None:
            diagnostics["requirement_id"] = task.requirement_id
            diagnostics["request_fingerprint"] = retrieval_fingerprint(task)
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
                (item.score.normalized for item in parsed_items if item.score.normalized is not None),
                default=None,
            )
            diagnostics["raw_score"] = max(
                (item.score.raw for item in parsed_items if item.score.raw is not None), default=None,
            )
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
            "hits": [],
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
        task=task,
    )
    payload = {"hits": payload_dicts, "diagnostics": diagnostic.model_dump(mode="json")}
    return RetrievalTaskResult(
        index=index,
        tool_name=tool_name,
        payload=payload,
        hits=payload_dicts,
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
    preserved_hits: list[dict[str, Any]],
    preserved_diagnostics: list[RetrievalDiagnostic],
) -> RetrievalTaskResult:
    route_hits = [hit for hit in parse_search_hits(preserved_hits)
                  if hit.requirement_id == task.requirement_id
                  or (not hit.requirement_id and hit.evidence.route == route)]
    prior = next((item for item in reversed(preserved_diagnostics) if item.requirement_id == task.requirement_id), None)
    if prior is None:
        prior = next((item for item in reversed(preserved_diagnostics) if not item.requirement_id and item.route == route), None)
    if prior is None:
        prior = RetrievalDiagnostic(tool=tool_name, route=route, query=task.query,
                                    status="success" if route_hits else "no_result")
    diagnostic = prior.model_copy(update={
        "tool": tool_name, "route": route, "attempt": attempt,
        "requirement_id": task.requirement_id,
        "evidence_count": len(route_hits), "result_count": len(route_hits),
        "final_evidence_count": len(route_hits), "reused": True,
        "provider_ms": 0, "url_validation_ms": 0, "post_filter_ms": 0,
    })
    payload = {"hits": [hit.model_dump(mode="json") for hit in route_hits],
               "diagnostics": diagnostic.model_dump(mode="json")}
    return RetrievalTaskResult(
        index=index, tool_name=tool_name, payload=payload, hits=payload["hits"],
        diagnostic=diagnostic, errors=[],
        latency_trace=make_retrieval_route_latency_event(
            route=route, tool=tool_name, attempt=attempt, latency_ms=0, status=diagnostic.status,
        ),
    )
