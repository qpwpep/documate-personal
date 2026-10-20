from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from contextvars import copy_context
from dataclasses import dataclass, field
import json
import logging
import time
from typing import Any

from src.core.contracts import GraphState, RetrievalDiagnostic
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.latency import elapsed_ms, make_stage_latency_event
from src.infra.logging_utils import log_event
from src.core.message_utils import build_tool_message
from src.core.planner_schema import RetrievalTask
from src.core.evidence import SearchHit, dedupe_search_hits
from src.runtime.nodes.retry import current_retrieval_attempt
from src.runtime.nodes.retrieval.executor import RetrievalTaskResult, build_reused_retrieval_task_result, execute_retrieval_task, retrieval_fingerprint
from src.runtime.agent_runtime.tool_execution import invoke_tool


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalBatchPlan:
    attempt: int
    indexed_tasks: list[tuple[int, RetrievalTask, str, Any]] = field(default_factory=list)
    reused_results: list[RetrievalTaskResult] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalBatchResult:
    hit_updates: list[SearchHit] = field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = field(default_factory=list)
    tool_messages: list[Any] = field(default_factory=list)
    local_errors: list[str] = field(default_factory=list)
    latency_trace: list[dict[str, Any]] = field(default_factory=list)


def _requires_docs_refresh(
    task: RetrievalTask, retry_context: Any,
    prior_diagnostics: list[RetrievalDiagnostic] | None,
) -> bool:
    if not (
        task.route == "docs" and retry_context.attempt > 0
        and retry_context.retry_scope == "refresh_routes"
        and task.requirement_id in retry_context.failed_requirement_ids
    ):
        return False
    latest = next((diagnostic for diagnostic in reversed(prior_diagnostics or [])
                   if diagnostic.requirement_id == task.requirement_id), None)
    if latest is not None:
        # After partial success, resume failed requests without repeating the
        # completed ones, even when pre-validation reports no_evidence.
        if latest.error_code is not None:
            return False
        # The failed requirement came from semantic validation. Search tools
        # report candidates, never an independent meaning-based covered flag.
        return True
    return retry_context.retry_reason == "no_evidence"


def _build_route_handlers(
    *,
    tavily_search_tool: Any,
    upload_search_tool: Any,
    runtime: Any,
    retry_context: Any,
    prior_diagnostics: list[RetrievalDiagnostic] | None = None,
    prior_hits: list[SearchHit] | None = None,
) -> dict[str, tuple[str, Any]]:
    def docs_search(task: RetrievalTask, execution_id: str):
        latest = next((item for item in reversed(prior_diagnostics or [])
                       if item.requirement_id == task.requirement_id), None)
        # Each result carries the current completion set. A failed refresh can
        # remove a prior marker, so unioning historical results would revive it.
        completed = list(latest.completed_requests) if latest is not None else []
        return invoke_tool("tavily_search", tavily_search_tool, execution_id=execution_id, query=task.query, k=task.k,
                                  requirement=task.requirement,
                                  completed_requests=completed,
                                  previous_hits=[hit for hit in (prior_hits or []) if hit.requirement_id == task.requirement_id],
                                  preserved_evidence_ids=[hit.evidence.id for hit in retry_context.preserved_hits
                                                          if hit.requirement_id == task.requirement_id],
                                  refresh=_requires_docs_refresh(task, retry_context, prior_diagnostics))

    return {
        "docs": (
            "tavily_search",
            docs_search,
        ),
        "upload": (
            "upload_search",
            lambda task, execution_id: invoke_tool("upload_search", upload_search_tool, execution_id=execution_id,
                query=task.query,
                k=task.k,
                retriever=runtime.retriever,
                requirement=task.requirement,
            ),
        ),
    }


def _collect_retrieval_batch(
    *,
    planner_output: Any,
    retry_context: Any,
    route_handlers: dict[str, tuple[str, Any]],
    prior_diagnostics: list[RetrievalDiagnostic] | None = None,
    prior_hits: list[SearchHit] | None = None,
) -> RetrievalBatchPlan:
    attempt = current_retrieval_attempt(retry_context)
    preserved_hits = retry_context.preserved_hits
    preserved_diagnostics = list(retry_context.preserved_retrieval_diagnostics)
    batch_plan = RetrievalBatchPlan(attempt=attempt)

    failed_ids = set(retry_context.failed_requirement_ids)
    for index, task in enumerate(planner_output.tasks, start=1):
        tool_name, invoke_tool = route_handlers[task.route]
        preserve = bool(failed_ids) and task.requirement_id not in failed_ids
        latest_matching = next((d for d in reversed(prior_diagnostics or [])
                                if d.requirement_id == task.requirement_id
                                and d.request_fingerprint == retrieval_fingerprint(task)), None)
        duplicate = (latest_matching if latest_matching is not None
                     and latest_matching.status != "error" and latest_matching.error_code is None
                     and not _requires_docs_refresh(task, retry_context, prior_diagnostics) else None)
        if preserve or duplicate is not None:
            result = build_reused_retrieval_task_result(
                index=index, task=task, tool_name=tool_name, route=task.route,
                attempt=attempt, preserved_hits=preserved_hits if preserve else list(prior_hits or []),
                preserved_diagnostics=preserved_diagnostics if preserve else [duplicate],
            )
            if duplicate is not None and not preserve:
                result.result.diagnostics = result.result.diagnostics.model_copy(update={
                    "warnings": list(dict.fromkeys([*result.result.diagnostics.warnings, "duplicate_request_reused"])),
                })
            batch_plan.reused_results.append(result)
            continue
        batch_plan.indexed_tasks.append((index, task, tool_name, invoke_tool))
    return batch_plan

def _execute_retrieval_batch(
    batch_plan: RetrievalBatchPlan, *, prior_candidates: list[SearchHit] | None = None,
) -> RetrievalBatchResult:
    task_results: list[RetrievalTaskResult] = list(batch_plan.reused_results)

    if len(batch_plan.indexed_tasks) == 1:
        index, task, tool_name, invoke_tool = batch_plan.indexed_tasks[0]
        task_results.append(
            execute_retrieval_task(
                index=index,
                task=task,
                tool_name=tool_name,
                route=task.route,
                invoke_tool=invoke_tool,
                attempt=batch_plan.attempt,
            )
        )
    elif batch_plan.indexed_tasks:
        with ThreadPoolExecutor(max_workers=min(4, len(batch_plan.indexed_tasks))) as executor:
            futures = {
                executor.submit(
                    copy_context().run,
                    execute_retrieval_task,
                    index=index,
                    task=task,
                    tool_name=tool_name,
                    route=task.route,
                    invoke_tool=invoke_tool,
                    attempt=batch_plan.attempt,
                ): index
                for index, task, tool_name, invoke_tool in batch_plan.indexed_tasks
            }
            for future in as_completed(futures):
                task_results.append(future.result())

    task_results.sort(key=lambda item: int(item.index))

    result = RetrievalBatchResult()
    tool_call_counts: dict[str, int] = {}
    for item in task_results:
        # A refresh may return only replacement candidates. Preserve raw ranges
        # until pre-validation can compare their complete decision identity.
        # An old failed verdict cannot erase a candidate for a changed request.
        retained = [hit for hit in (prior_candidates or [])
                    if hit.requirement_id == item.result.diagnostics.requirement_id]
        if retained:
            merged = dedupe_search_hits([*retained, *item.result.hits])
            item.result = item.result.model_copy(update={
                "hits": merged,
                "diagnostics": item.result.diagnostics.model_copy(update={
                    "evidence_count": len(merged), "result_count": len(merged),
                    "final_evidence_count": len(merged),
                }),
            })
        result.hit_updates.extend(item.result.hits)
        result.retrieval_diagnostics.append(item.result.diagnostics)
        result.local_errors.extend(str(error) for error in item.errors if str(error).strip())
        result.latency_trace.append(item.latency_trace)

        tool_call_counts[item.tool_name] = tool_call_counts.get(item.tool_name, 0) + 1
        result.tool_messages.append(
            build_tool_message(item.tool_name, item.result.model_dump(mode="json"), tool_call_counts[item.tool_name])
        )
    return result


def _build_retrieval_updates(
    *,
    retrieval: Any,
    debug: Any,
    batch_result: RetrievalBatchResult,
) -> GraphState:
    updates: GraphState = {
        "retrieval": retrieval.model_copy(
            update={"hit_log": [*retrieval.hit_log, *batch_result.hit_updates], "prepared_evidence": None}
        ),
        "messages": batch_result.tool_messages,
    }
    if batch_result.local_errors or batch_result.retrieval_diagnostics or batch_result.latency_trace:
        retrieval_error_codes = [
            diagnostic.error_code
            for diagnostic in batch_result.retrieval_diagnostics
            if diagnostic.error_code
        ]
        updates["debug"] = debug.model_copy(
            update={
                "retrieval_errors": [*debug.retrieval_errors, *batch_result.local_errors],
                "retrieval_diagnostics": [*debug.retrieval_diagnostics, *batch_result.retrieval_diagnostics],
                "error_codes": [
                    *debug.error_codes,
                    *[
                        code
                        for code in retrieval_error_codes
                        if code not in debug.error_codes
                    ],
                ],
                "latency_trace": [*debug.latency_trace, *batch_result.latency_trace],
            }
        )
    return updates


def _emit_retrieval_progress_snapshot(
    *,
    progress_emitter: Any | None,
    batch_result: RetrievalBatchResult,
) -> None:
    if progress_emitter is None or not hasattr(progress_emitter, "emit_progress_snapshot"):
        return
    route_counts: dict[str, int] = {}
    for item in batch_result.retrieval_diagnostics:
        route = str(item.route or "").strip()
        if route:
            route_counts[route] = route_counts.get(route, 0) + int(item.evidence_count or 0)
    summary_parts = [
        f"{route} {count}건"
        for route, count in route_counts.items()
        if count > 0
    ]
    summary = "근거 요약: " + (", ".join(summary_parts) if summary_parts else "관련 근거 없음")
    progress_emitter.emit_progress_snapshot(
        stage="retrieval",
        summary=summary,
        evidence_count=sum(route_counts.values()),
        routes=route_counts,
        statuses=[str(item.status or "") for item in batch_result.retrieval_diagnostics],
    )


def make_retrieve_dispatch_node(
    tavily_search_tool: Any,
    upload_search_tool: Any,
    verbose: bool,
):
    def retrieve_dispatch(state: GraphState) -> GraphState:
        stage_started = time.perf_counter()
        runtime = get_runtime_state(state)
        planner = get_planner_state(state)
        retrieval = get_retrieval_state(state)
        debug = get_debug_state(state)
        planner_output = planner.output
        if not planner_output.use_retrieval:
            return {}

        retry_context = get_retry_state(state)
        prior_hits = dedupe_search_hits([*retry_context.preserved_hits, *retrieval.hit_log])

        route_handlers = _build_route_handlers(
            tavily_search_tool=tavily_search_tool,
            upload_search_tool=upload_search_tool,
            runtime=runtime,
            retry_context=retry_context,
            prior_diagnostics=debug.retrieval_diagnostics,
            prior_hits=prior_hits,
        )
        batch_plan = _collect_retrieval_batch(
            planner_output=planner_output,
            retry_context=retry_context,
            route_handlers=route_handlers,
            prior_diagnostics=debug.retrieval_diagnostics,
            prior_hits=prior_hits,
        )
        batch_result = _execute_retrieval_batch(
            batch_plan,
            prior_candidates=prior_hits,
        )
        _emit_retrieval_progress_snapshot(
            progress_emitter=getattr(runtime, "progress_emitter", None),
            batch_result=batch_result,
        )

        if verbose:
            routes = ",".join(task.route for task in planner_output.tasks)
            statuses = ",".join(item.status for item in batch_result.retrieval_diagnostics)
            log_event(
                logger,
                logging.INFO,
                "retrieve_dispatch",
                task_count=len(planner_output.tasks),
                routes=routes,
                evidence_count=len(batch_result.hit_updates),
                statuses=statuses,
            )
            for diagnostic in batch_result.retrieval_diagnostics:
                log_event(
                    logger,
                    logging.INFO,
                    "retrieval_diagnostic",
                    details=json.dumps({
                        "requirement_id": diagnostic.requirement_id,
                        "route": diagnostic.route,
                        "attempt": diagnostic.attempt,
                        "query": diagnostic.query,
                        "k": diagnostic.k,
                        "status": diagnostic.status,
                        "answerability": diagnostic.answerability,
                        "reused": diagnostic.reused,
                        "missing_requirements": diagnostic.missing_requirements,
                        "provider_requests": [request.model_dump(mode="json") for request in diagnostic.provider_requests],
                    }, ensure_ascii=False),
                )

        stage_status = None
        if batch_result.retrieval_diagnostics:
            statuses = {item.status for item in batch_result.retrieval_diagnostics}
            if statuses.issubset({"success", "no_result"}):
                stage_status = "success"
            elif statuses.issubset({"error", "unavailable"}):
                stage_status = "error"
            else:
                stage_status = "mixed"
        batch_result.latency_trace.append(
            make_stage_latency_event(
                stage="retrieval",
                attempt=batch_plan.attempt,
                latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                status=stage_status,
            )
        )

        return _build_retrieval_updates(
            retrieval=retrieval,
            debug=debug,
            batch_result=batch_result,
        )

    return retrieve_dispatch
