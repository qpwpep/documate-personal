from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any
from uuid import uuid4

from src.core.contracts.debug import RetrievalDiagnostic
from src.core.contracts.retrieval import RetrievalResult
from src.core.evidence import SearchHit, dedupe_search_hits
from src.core.latency import elapsed_ms, make_retrieval_route_latency_event
from src.core.planner_schema import RetrievalTask
from src.runtime.agent_runtime.tool_execution import record_nonexecution


@dataclass(slots=True)
class RetrievalTaskResult:
    index: int
    tool_name: str
    result: RetrievalResult
    errors: list[str]
    latency_trace: dict[str, Any]


def retrieval_fingerprint(task: RetrievalTask) -> str:
    """Request identity within a turn, independent of a model-assigned task ID."""
    payload = {"route": task.route, "query": task.query,
               "requirement": task.requirement.model_dump(mode="json"), "k": task.k}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def execute_retrieval_task(
    *, index: int, task: RetrievalTask, tool_name: str, route: str,
    invoke_tool: Callable[[RetrievalTask, str], RetrievalResult], attempt: int,
) -> RetrievalTaskResult:
    started = time.perf_counter()
    invocation_id = uuid4().hex
    result = invoke_tool(task, invocation_id)
    if not isinstance(result, RetrievalResult):
        raise TypeError(f"{tool_name} must return RetrievalResult")
    hits = [hit.model_copy(update={"requirement_id": task.requirement_id})
            for hit in dedupe_search_hits(result.hits)]
    diagnostic = result.diagnostics.model_copy(update={
        "tool": tool_name, "route": route, "attempt": attempt,
        "requirement_id": task.requirement_id, "request_fingerprint": retrieval_fingerprint(task),
        "k": task.k, "invocation_id": invocation_id,
        "evidence_count": len(hits), "result_count": len(hits), "final_evidence_count": len(hits),
        "warnings": sorted(set(result.diagnostics.warnings)),
    })
    errors = ([f"{tool_name}: {diagnostic.message}"]
              if diagnostic.status in {"error", "unavailable"} and diagnostic.message else [])
    return RetrievalTaskResult(
        index=index, tool_name=tool_name, result=RetrievalResult(hits=hits, diagnostics=diagnostic),
        errors=errors, latency_trace=make_retrieval_route_latency_event(
            route=route, tool=tool_name, attempt=attempt,
            latency_ms=elapsed_ms(started, time.perf_counter()), status=diagnostic.status,
        ),
    )


def build_reused_retrieval_task_result(
    *, index: int, task: RetrievalTask, tool_name: str, route: str, attempt: int,
    preserved_hits: list[SearchHit], preserved_diagnostics: list[RetrievalDiagnostic],
) -> RetrievalTaskResult:
    route_hits = [hit for hit in preserved_hits if hit.requirement_id == task.requirement_id]
    prior = next(item for item in reversed(preserved_diagnostics)
                 if item.requirement_id == task.requirement_id)
    record_nonexecution(tool_name, phase="reused", reason_code="retrieval_evidence_reused",
                        origin_invocation_id=prior.invocation_id)
    diagnostic = prior.model_copy(update={
        "tool": tool_name, "route": route, "attempt": attempt,
        "requirement_id": task.requirement_id, "evidence_count": len(route_hits),
        "result_count": len(route_hits), "final_evidence_count": len(route_hits),
        "reused": True, "provider_ms": 0, "post_filter_ms": 0,
        "k": task.k, "provider_requests": [],
    })
    return RetrievalTaskResult(
        index=index, tool_name=tool_name,
        result=RetrievalResult(hits=route_hits, diagnostics=diagnostic), errors=[],
        latency_trace=make_retrieval_route_latency_event(
            route=route, tool=tool_name, attempt=attempt, latency_ms=0, status=diagnostic.status,
        ),
    )
