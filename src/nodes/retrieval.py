from __future__ import annotations

from typing import Any

from ..evidence import evidence_to_dicts, parse_evidence_payload
from ..planner_schema import RetrievalTask
from .retry import current_retrieval_attempt
from .state import (
    RetrievalDiagnostic,
    RetryContext,
    State,
    build_tool_message,
    coerce_planner_output,
    coerce_retry_context,
)


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


def tasks_for_route(state: State, route: str) -> list[RetrievalTask]:
    parse_errors: list[str] = []
    planner_output = coerce_planner_output(state.get("planner_output"), parse_errors)
    if not planner_output.use_retrieval:
        return []
    return [task for task in planner_output.tasks if task.route == route]


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


def run_retrieval_tasks(
    tasks: list[RetrievalTask],
    *,
    tool_name: str,
    route: str,
    context_label: str,
    invoke_tool: Any,
    verbose: bool,
    retry_context: RetryContext | None = None,
) -> State:
    if not tasks:
        return {}

    local_errors: list[str] = []
    evidence_updates: list[dict[str, Any]] = []
    retrieval_diagnostics: list[dict[str, Any]] = []
    tool_messages = []
    attempt = current_retrieval_attempt(retry_context or coerce_retry_context(None))

    for index, task in enumerate(tasks, start=1):
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

        payload_dicts, diagnostic = collect_retrieval_result(
            raw_payload=payload,
            tool_name=tool_name,
            route=route,
            query=task.query,
            attempt=attempt,
            local_errors=local_errors,
        )
        evidence_updates.extend(payload_dicts)
        retrieval_diagnostics.append(diagnostic)
        tool_messages.append(build_tool_message(tool_name, payload, index))

    if verbose:
        print(
            f"[{context_label}] tasks={len(tasks)} evidence={len(evidence_updates)} "
            f"statuses={','.join(str(item.get('status')) for item in retrieval_diagnostics)}"
        )

    updates: State = {
        "retrieved_evidence": evidence_updates,
        "retrieval_diagnostics": retrieval_diagnostics,
        "messages": tool_messages,
    }
    if local_errors:
        updates["retrieval_errors"] = local_errors
    return updates


def format_evidence_for_prompt(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No retrieved evidence."

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "unknown")
        source = str(item.get("url_or_path") or "unknown-source")
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        header = f"{index}. [{kind}] {title} - {source}" if title else f"{index}. [{kind}] {source}"
        lines.append(header)
        if snippet:
            lines.append(f"   snippet: {snippet}")
    return "\n".join(lines)


def make_retrieve_docs_node(tavily_search_tool: Any, verbose: bool):
    def retrieve_docs(state: State) -> State:
        tasks = tasks_for_route(state, "docs")
        return run_retrieval_tasks(
            tasks,
            tool_name="tavily_search",
            route="docs",
            context_label="retrieve_docs",
            invoke_tool=lambda task: tavily_search_tool.func(query=task.query),
            verbose=verbose,
            retry_context=coerce_retry_context(state.get("retry_context")),
        )

    return retrieve_docs


def make_retrieve_upload_node(upload_search_tool: Any, verbose: bool):
    def retrieve_upload(state: State) -> State:
        tasks = tasks_for_route(state, "upload")
        return run_retrieval_tasks(
            tasks,
            tool_name="upload_search",
            route="upload",
            context_label="retrieve_upload",
            invoke_tool=lambda task: upload_search_tool.func(
                query=task.query,
                k=task.k,
                retriever=state.get("retriever"),
            ),
            verbose=verbose,
            retry_context=coerce_retry_context(state.get("retry_context")),
        )

    return retrieve_upload


def make_retrieve_local_node(rag_search_tool: Any, verbose: bool):
    def retrieve_local(state: State) -> State:
        tasks = tasks_for_route(state, "local")
        return run_retrieval_tasks(
            tasks,
            tool_name="rag_search",
            route="local",
            context_label="retrieve_local",
            invoke_tool=lambda task: rag_search_tool.func(query=task.query, k=task.k),
            verbose=verbose,
            retry_context=coerce_retry_context(state.get("retry_context")),
        )

    return retrieve_local


def make_retrieve_dispatch_node(
    tavily_search_tool: Any,
    upload_search_tool: Any,
    rag_search_tool: Any,
    verbose: bool,
):
    def retrieve_dispatch(state: State) -> State:
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
        retry_context = coerce_retry_context(state.get("retry_context"))
        attempt = current_retrieval_attempt(retry_context)

        for task in planner_output.tasks:
            handler = route_handlers.get(task.route)
            if handler is None:
                local_errors.append(f"planner: unsupported route ({task.route})")
                continue

            tool_name, invoke_tool = handler
            try:
                payload = invoke_tool(task)
            except Exception as exc:
                payload = {
                    "evidence": [],
                    "diagnostics": {
                        "tool": tool_name,
                        "route": task.route,
                        "status": "error",
                        "message": f"tool invocation failed ({exc})",
                        "query": task.query,
                    },
                }

            payload_dicts, diagnostic = collect_retrieval_result(
                raw_payload=payload,
                tool_name=tool_name,
                route=task.route,
                query=task.query,
                attempt=attempt,
                local_errors=local_errors,
            )
            evidence_updates.extend(payload_dicts)
            retrieval_diagnostics.append(diagnostic)

            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            tool_messages.append(
                build_tool_message(tool_name, payload, tool_call_counts[tool_name])
            )

        if verbose:
            routes = ",".join(task.route for task in planner_output.tasks)
            print(
                f"[retrieve_dispatch] tasks={len(planner_output.tasks)} "
                f"routes={routes} evidence={len(evidence_updates)} "
                f"statuses={','.join(str(item.get('status')) for item in retrieval_diagnostics)}"
            )

        updates: State = {
            "retrieved_evidence": evidence_updates,
            "retrieval_diagnostics": retrieval_diagnostics,
            "messages": tool_messages,
        }
        if local_errors:
            updates["retrieval_errors"] = local_errors
        return updates

    return retrieve_dispatch
