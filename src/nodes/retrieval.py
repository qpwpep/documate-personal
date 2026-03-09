from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from typing import Any

from ..evidence import evidence_to_dicts, parse_evidence_payload
from ..latency import (
    elapsed_ms,
    make_retrieval_route_latency_event,
    make_stage_latency_event,
)
from ..logging_utils import log_event
from ..planner_schema import RetrievalTask
from .retry import current_retrieval_attempt
from .state import (
    RetrievalDiagnostic,
    State,
    build_tool_message,
    coerce_planner_output,
    coerce_retry_context,
)


logger = logging.getLogger(__name__)


def route_for_tool(tool_name: str) -> str:
    return {
        "tavily_search": "docs",
        "upload_search": "upload",
        "rag_search": "local",
    }.get(tool_name, "unknown")


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

    return {
        "tool": str(diagnostics.get("tool") or tool_name),
        "route": str(diagnostics.get("route") or route or route_for_tool(tool_name)),
        "status": status,
        "message": message,
        "query": str(diagnostics.get("query") or query),
        "attempt": diagnostic_attempt,
    }


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
    if diagnostic["status"] in {"error", "unavailable"} and diagnostic["message"]:
        local_errors.append(f"{tool_name}: {diagnostic['message']}")
    return payload_dicts, diagnostic


def _execute_retrieval_task(
    *,
    index: int,
    task: RetrievalTask,
    tool_name: str,
    route: str,
    invoke_tool: Any,
    attempt: int,
) -> dict[str, Any]:
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
    return {
        "index": index,
        "tool_name": tool_name,
        "payload": payload,
        "evidence": payload_dicts,
        "diagnostic": diagnostic,
        "errors": local_errors,
        "latency_trace": make_retrieval_route_latency_event(
            route=route,
            tool=tool_name,
            attempt=attempt,
            latency_ms=latency_ms,
            status=str(diagnostic.get("status") or ""),
        ),
    }


def format_evidence_for_prompt(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No retrieved evidence."

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "unknown")
        source = str(item.get("url_or_path") or "unknown-source")
        source_id = str(item.get("source_id") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        chunk_id = item.get("chunk_id")
        cell_id = item.get("cell_id")
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        header = f"{index}. [{kind}] {title} - {source}" if title else f"{index}. [{kind}] {source}"
        lines.append(header)
        if source_id:
            lines.append(f"   source_id: {source_id}")
        if cell_id is not None or chunk_id is not None:
            location_parts: list[str] = []
            if cell_id is not None:
                location_parts.append(f"cell_id={cell_id}")
            if chunk_id is not None:
                location_parts.append(f"chunk_id={chunk_id}")
            if start_offset is not None and end_offset is not None:
                location_parts.append(f"offsets={start_offset}-{end_offset}")
            if location_parts:
                lines.append(f"   location: {', '.join(location_parts)}")
        if snippet:
            lines.append(f"   snippet: {snippet}")
    return "\n".join(lines)


def make_retrieve_dispatch_node(
    tavily_search_tool: Any,
    upload_search_tool: Any,
    rag_search_tool: Any,
    verbose: bool,
):
    def retrieve_dispatch(state: State) -> State:
        stage_started = time.perf_counter()
        planner_errors: list[str] = []
        planner_output = coerce_planner_output(state.get("planner_output"), planner_errors)
        if not planner_output.use_retrieval or not planner_output.tasks:
            if planner_errors:
                return {"retrieval_errors": planner_errors}
            return {}

        route_handlers: dict[str, tuple[str, Any]] = {
            "docs": (
                "tavily_search",
                lambda task: tavily_search_tool.func(query=task.query),
            ),
            "upload": (
                "upload_search",
                lambda task: upload_search_tool.func(
                    query=task.query,
                    k=task.k,
                    retriever=state.get("retriever"),
                ),
            ),
            "local": (
                "rag_search",
                lambda task: rag_search_tool.func(query=task.query, k=task.k),
            ),
        }

        local_errors: list[str] = list(planner_errors)
        evidence_updates: list[dict[str, Any]] = []
        retrieval_diagnostics: list[dict[str, Any]] = []
        tool_messages = []
        tool_call_counts: dict[str, int] = {}
        latency_trace: list[dict[str, Any]] = []
        retry_context = coerce_retry_context(state.get("retry_context"))
        attempt = current_retrieval_attempt(retry_context)
        indexed_tasks: list[tuple[int, RetrievalTask, str, Any]] = []
        for index, task in enumerate(planner_output.tasks, start=1):
            handler = route_handlers.get(task.route)
            if handler is None:
                local_errors.append(f"planner: unsupported route ({task.route})")
                continue
            tool_name, invoke_tool = handler
            indexed_tasks.append((index, task, tool_name, invoke_tool))

        task_results: list[dict[str, Any]] = []
        if len(indexed_tasks) == 1:
            index, task, tool_name, invoke_tool = indexed_tasks[0]
            task_results.append(
                _execute_retrieval_task(
                    index=index,
                    task=task,
                    tool_name=tool_name,
                    route=task.route,
                    invoke_tool=invoke_tool,
                    attempt=attempt,
                )
            )
        elif indexed_tasks:
            with ThreadPoolExecutor(max_workers=len(indexed_tasks)) as executor:
                futures = {
                    executor.submit(
                        _execute_retrieval_task,
                        index=index,
                        task=task,
                        tool_name=tool_name,
                        route=task.route,
                        invoke_tool=invoke_tool,
                        attempt=attempt,
                    ): index
                    for index, task, tool_name, invoke_tool in indexed_tasks
                }
                for future in as_completed(futures):
                    task_results.append(future.result())

        task_results.sort(key=lambda item: int(item.get("index", 0)))

        for result in task_results:
            tool_name = str(result.get("tool_name") or "")
            payload = result.get("payload")
            evidence_updates.extend(result.get("evidence") or [])
            diagnostic = result.get("diagnostic")
            if isinstance(diagnostic, dict):
                retrieval_diagnostics.append(diagnostic)
            local_errors.extend(str(error) for error in (result.get("errors") or []) if str(error).strip())
            latency_event = result.get("latency_trace")
            if isinstance(latency_event, dict):
                latency_trace.append(latency_event)

            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            tool_messages.append(
                build_tool_message(tool_name, payload, tool_call_counts[tool_name])
            )

        if verbose:
            routes = ",".join(task.route for task in planner_output.tasks)
            statuses = ",".join(str(item.get("status")) for item in retrieval_diagnostics)
            log_event(
                logger,
                logging.INFO,
                "retrieve_dispatch",
                task_count=len(planner_output.tasks),
                routes=routes,
                evidence_count=len(evidence_updates),
                statuses=statuses,
            )

        stage_status = None
        if retrieval_diagnostics:
            statuses = {str(item.get("status") or "") for item in retrieval_diagnostics}
            if statuses.issubset({"success", "no_result"}):
                stage_status = "success"
            elif statuses.issubset({"error", "unavailable"}):
                stage_status = "error"
            else:
                stage_status = "mixed"
        latency_trace.append(
            make_stage_latency_event(
                stage="retrieval",
                attempt=attempt,
                latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                status=stage_status,
            )
        )

        updates: State = {
            "retrieved_evidence": evidence_updates,
            "retrieval_diagnostics": retrieval_diagnostics,
            "messages": tool_messages,
            "latency_trace": latency_trace,
        }
        if local_errors:
            updates["retrieval_errors"] = local_errors
        return updates

    return retrieve_dispatch
