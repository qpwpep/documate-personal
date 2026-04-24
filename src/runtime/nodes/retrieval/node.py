from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
import logging
import time
from typing import Any

from src.core.contracts import GraphState, RetrievalDiagnostic
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.latency import elapsed_ms, make_stage_latency_event
from src.infra.logging_utils import log_event
from src.core.message_utils import build_tool_message
from src.core.planner_schema import RetrievalTask
from src.runtime.nodes.planner import sanitize_retrieval_query
from src.runtime.nodes.retry import current_retrieval_attempt
from src.runtime.nodes.retrieval.executor import RetrievalTaskResult, build_reused_retrieval_task_result, execute_retrieval_task


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievalBatchPlan:
    attempt: int
    indexed_tasks: list[tuple[int, RetrievalTask, str, Any]] = field(default_factory=list)
    reused_results: list[RetrievalTaskResult] = field(default_factory=list)
    planner_errors: list[str] = field(default_factory=list)
    local_errors: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RetrievalBatchResult:
    evidence_updates: list[dict[str, Any]] = field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = field(default_factory=list)
    tool_messages: list[Any] = field(default_factory=list)
    local_errors: list[str] = field(default_factory=list)
    latency_trace: list[dict[str, Any]] = field(default_factory=list)


def _build_route_handlers(
    *,
    tavily_search_tool: Any,
    upload_search_tool: Any,
    rag_search_tool: Any,
    runtime: Any,
) -> dict[str, tuple[str, Any]]:
    return {
        "docs": (
            "tavily_search",
            lambda task: tavily_search_tool.func(query=task.query),
        ),
        "upload": (
            "upload_search",
            lambda task: upload_search_tool.func(
                query=task.query,
                k=task.k,
                retriever=runtime.retriever,
            ),
        ),
        "local": (
            "rag_search",
            lambda task: rag_search_tool.func(query=task.query, k=task.k),
        ),
    }


def _collect_retrieval_batch(
    *,
    planner_output: Any,
    retry_context: Any,
    route_handlers: dict[str, tuple[str, Any]],
) -> RetrievalBatchPlan:
    attempt = current_retrieval_attempt(retry_context)
    failed_routes = {
        str(route).strip()
        for route in retry_context.failed_routes
        if str(route).strip()
    }
    retry_scope = str(getattr(retry_context, "retry_scope", "") or "").strip()
    preserved_evidence = [
        item
        for item in retry_context.preserved_evidence
        if isinstance(item, dict)
    ]
    preserved_diagnostics = list(retry_context.preserved_retrieval_diagnostics)
    batch_plan = RetrievalBatchPlan(attempt=attempt)

    if retry_scope == "reuse_evidence_resynthesize" and preserved_evidence:
        for index, task in enumerate(planner_output.tasks, start=1):
            handler = route_handlers.get(task.route)
            if handler is None:
                batch_plan.local_errors.append(f"planner: unsupported route ({task.route})")
                continue
            tool_name, _invoke_tool = handler
            sanitized_query = sanitize_retrieval_query(
                route=task.route,
                query=task.query,
                retry_context=retry_context,
            )
            sanitized_task = RetrievalTask(route=task.route, query=sanitized_query, k=task.k)
            batch_plan.reused_results.append(
                build_reused_retrieval_task_result(
                    index=index,
                    task=sanitized_task,
                    tool_name=tool_name,
                    route=sanitized_task.route,
                    attempt=attempt,
                    preserved_evidence=preserved_evidence,
                    preserved_diagnostics=preserved_diagnostics,
                )
            )
        return batch_plan

    for index, task in enumerate(planner_output.tasks, start=1):
        handler = route_handlers.get(task.route)
        if handler is None:
            batch_plan.local_errors.append(f"planner: unsupported route ({task.route})")
            continue
        tool_name, invoke_tool = handler
        sanitized_query = sanitize_retrieval_query(
            route=task.route,
            query=task.query,
            retry_context=retry_context,
        )
        sanitized_task = RetrievalTask(route=task.route, query=sanitized_query, k=task.k)
        if failed_routes and sanitized_task.route not in failed_routes:
            batch_plan.reused_results.append(
                build_reused_retrieval_task_result(
                    index=index,
                    task=sanitized_task,
                    tool_name=tool_name,
                    route=sanitized_task.route,
                    attempt=attempt,
                    preserved_evidence=preserved_evidence,
                    preserved_diagnostics=preserved_diagnostics,
                )
            )
            continue
        batch_plan.indexed_tasks.append((index, sanitized_task, tool_name, invoke_tool))
    return batch_plan


def _execute_retrieval_batch(batch_plan: RetrievalBatchPlan) -> RetrievalBatchResult:
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
        with ThreadPoolExecutor(max_workers=len(batch_plan.indexed_tasks)) as executor:
            futures = {
                executor.submit(
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

    result = RetrievalBatchResult(local_errors=list(batch_plan.local_errors))
    tool_call_counts: dict[str, int] = {}
    for item in task_results:
        result.evidence_updates.extend(item.evidence)
        result.retrieval_diagnostics.append(item.diagnostic)
        result.local_errors.extend(str(error) for error in item.errors if str(error).strip())
        result.latency_trace.append(item.latency_trace)

        tool_call_counts[item.tool_name] = tool_call_counts.get(item.tool_name, 0) + 1
        result.tool_messages.append(
            build_tool_message(item.tool_name, item.payload, tool_call_counts[item.tool_name])
        )
    return result


def _build_retrieval_updates(
    *,
    retrieval: Any,
    debug: Any,
    planner_errors: list[str],
    batch_result: RetrievalBatchResult,
) -> GraphState:
    updates: GraphState = {
        "retrieval": retrieval.model_copy(
            update={"evidence_log": [*retrieval.evidence_log, *batch_result.evidence_updates]}
        ),
        "messages": batch_result.tool_messages,
    }
    if planner_errors or batch_result.local_errors or batch_result.retrieval_diagnostics or batch_result.latency_trace:
        retrieval_error_codes = [
            diagnostic.error_code
            for diagnostic in batch_result.retrieval_diagnostics
            if diagnostic.error_code
        ]
        updates["debug"] = debug.model_copy(
            update={
                "planner_errors": [*debug.planner_errors, *planner_errors],
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
    rag_search_tool: Any,
    verbose: bool,
):
    def retrieve_dispatch(state: GraphState) -> GraphState:
        stage_started = time.perf_counter()
        runtime = get_runtime_state(state)
        planner = get_planner_state(state)
        retrieval = get_retrieval_state(state)
        debug = get_debug_state(state)
        planner_errors: list[str] = []
        planner_output = parse_planner_output(planner.output, planner_errors)
        if not planner_output.use_retrieval or not planner_output.tasks:
            if planner_errors:
                return {
                    "debug": debug.model_copy(
                        update={"planner_errors": [*debug.planner_errors, *planner_errors]}
                    )
                }
            return {}

        route_handlers = _build_route_handlers(
            tavily_search_tool=tavily_search_tool,
            upload_search_tool=upload_search_tool,
            rag_search_tool=rag_search_tool,
            runtime=runtime,
        )
        batch_plan = _collect_retrieval_batch(
            planner_output=planner_output,
            retry_context=get_retry_state(state),
            route_handlers=route_handlers,
        )
        batch_result = _execute_retrieval_batch(batch_plan)
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
                evidence_count=len(batch_result.evidence_updates),
                statuses=statuses,
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
            planner_errors=[*planner_errors, *batch_plan.planner_errors],
            batch_result=batch_result,
        )

    return retrieve_dispatch
