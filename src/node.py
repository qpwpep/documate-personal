from __future__ import annotations

import json
from typing import Annotated, Any, List, Optional

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import add_messages
from typing_extensions import TypedDict

from .evidence import dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from .planner_schema import PlannerOutput, RetrievalTask
from .prompts import SYS_POLICY, needs_save, needs_slack


def _merge_string_lists(current: list[str] | None, update: list[str] | None) -> list[str]:
    merged = list(current or [])
    if update:
        merged.extend(update)
    return merged


def _merge_evidence_dict_lists(
    current: list[dict[str, Any]] | None,
    update: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    merged = list(current or [])
    if update:
        merged.extend(update)
    return merged


class State(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    user_input: str
    final_answer: Optional[str]
    retriever: Optional[Any]
    memory_summary: Optional[str]
    planner_output: PlannerOutput
    retrieved_evidence: Annotated[list[dict[str, Any]], _merge_evidence_dict_lists]
    retrieval_errors: Annotated[list[str], _merge_string_lists]
    validation_errors: Annotated[list[str], _merge_string_lists]
    action_errors: Annotated[list[str], _merge_string_lists]
    synthesis_attempt: int
    needs_retry: bool


def add_user_message(state: State) -> State:
    msgs = list(state.get("messages", []))
    msgs.append(HumanMessage(content=state["user_input"]))
    state["messages"] = msgs
    return state


def _keep_recent_messages(messages: List[BaseMessage], max_turns: int = 6) -> List[BaseMessage]:
    if not messages:
        return messages
    window_size = max_turns * 2 + 2
    return messages[-window_size:]


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


_SUMMARY_SYS = (
    "Summarize the older conversation in 4-5 lines.\n"
    "- Keep topic, conclusions, decisions, key code/version/URL.\n"
    "- Remove duplication.\n"
    "- If uncertain, state uncertainty explicitly.\n"
)


def make_summarize_node(llm_summarizer: Any, verbose: bool, max_turns: int = 6):
    def summarize_old_messages(state: State) -> State:
        msgs: List[BaseMessage] = state.get("messages", [])
        recent_window = _keep_recent_messages(msgs, max_turns=max_turns)
        if len(recent_window) == len(msgs):
            return state

        cutoff = len(msgs) - len(recent_window)
        old = msgs[:cutoff]
        recent = msgs[cutoff:]

        try:
            summary = llm_summarizer.invoke([SystemMessage(content=_SUMMARY_SYS)] + old).content.strip()
        except Exception as exc:
            if verbose:
                print(f"[summary] failed: {exc}")
            state["messages"] = recent
            return state

        prev = (state.get("memory_summary") or "").strip()
        merged = (prev + ("\n" if prev else "") + summary).strip()
        state["memory_summary"] = merged
        state["messages"] = recent
        if verbose:
            print(f"[summary] merged ({cutoff} msgs -> 4~5 lines)")
        return state

    return summarize_old_messages


_PLANNER_SYS = (
    "You are a retrieval planner. Return a structured plan only.\n"
    "Rules:\n"
    "- Choose retrieval routes from: docs, upload, local.\n"
    "- docs: official/latest docs on the web.\n"
    "- upload: currently uploaded-file retriever context.\n"
    "- local: local notebook/vector index examples.\n"
    "- If retrieval is unnecessary, set use_retrieval=false and tasks=[].\n"
    "- If retrieval is needed, set use_retrieval=true and include 1-3 tasks.\n"
    "- Each selected route must appear at most once.\n"
    "- Keep each task.query short and route-specific.\n"
    "- Do not include actions for save/slack; only retrieval planning."
)


def _coerce_planner_output(raw: Any, errors: list[str]) -> PlannerOutput:
    if isinstance(raw, PlannerOutput):
        return raw
    try:
        return PlannerOutput.model_validate(raw)
    except Exception as exc:
        errors.append(f"planner: output validation failed ({exc})")
        return PlannerOutput.fallback()


def _sanitize_planner_output(
    planner_output: PlannerOutput,
    *,
    has_retriever: bool,
    errors: list[str],
) -> PlannerOutput:
    tasks: list[RetrievalTask] = list(planner_output.tasks)
    if not has_retriever and any(task.route == "upload" for task in tasks):
        tasks = [task for task in tasks if task.route != "upload"]
        errors.append("planner: dropped upload route because retriever is unavailable")

    try:
        return PlannerOutput(
            use_retrieval=bool(planner_output.use_retrieval and tasks),
            tasks=tasks,
        )
    except Exception as exc:
        errors.append(f"planner: sanitized output validation failed ({exc})")
        return PlannerOutput.fallback()


def _build_planner_messages(state: State, max_turns: int = 6) -> list[BaseMessage]:
    model_msgs: list[BaseMessage] = [SystemMessage(content=_PLANNER_SYS)]
    model_msgs.append(
        SystemMessage(content=f"[Planner Context]\nretriever_available={bool(state.get('retriever'))}")
    )

    if state.get("memory_summary"):
        model_msgs.append(SystemMessage(content=f"[Conversation Summary]\n{state['memory_summary']}"))

    conversation = [m for m in state.get("messages", []) if not isinstance(m, ToolMessage)]
    conversation = _keep_recent_messages(conversation, max_turns=max_turns)
    model_msgs.extend(conversation)

    if not any(isinstance(m, HumanMessage) for m in model_msgs):
        model_msgs.append(HumanMessage(content=str(state.get("user_input", "")).strip()))
    return model_msgs


def make_planner_node(llm_planner: Any, verbose: bool, max_turns: int = 6):
    def planner(state: State) -> State:
        planner_errors: list[str] = []
        try:
            planner_raw = llm_planner.invoke(_build_planner_messages(state, max_turns=max_turns))
            planner_output = _coerce_planner_output(planner_raw, planner_errors)
        except Exception as exc:
            planner_errors.append(f"planner: structured output invocation failed ({exc})")
            planner_output = PlannerOutput.fallback()

        planner_output = _sanitize_planner_output(
            planner_output,
            has_retriever=bool(state.get("retriever")),
            errors=planner_errors,
        )
        if verbose:
            print(f"[planner] use_retrieval={planner_output.use_retrieval} tasks={len(planner_output.tasks)}")

        return {
            "planner_output": planner_output,
            "retrieval_errors": planner_errors,
            "synthesis_attempt": 0,
            "needs_retry": False,
        }

    return planner


def _tasks_for_route(state: State, route: str) -> list[RetrievalTask]:
    parse_errors: list[str] = []
    planner_output = _coerce_planner_output(state.get("planner_output"), parse_errors)
    if not planner_output.use_retrieval:
        return []
    return [task for task in planner_output.tasks if task.route == route]


def _build_tool_message(tool_name: str, payload: Any, index: int) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=tool_name,
        tool_call_id=f"{tool_name}-{index}",
    )


def _format_evidence_for_prompt(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No retrieved evidence."

    lines: list[str] = []
    for idx, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "unknown")
        source = str(item.get("url_or_path") or "unknown-source")
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        header = f"{idx}. [{kind}] {title} - {source}" if title else f"{idx}. [{kind}] {source}"
        lines.append(header)
        if snippet:
            lines.append(f"   snippet: {snippet}")
    return "\n".join(lines)


def make_retrieve_docs_node(tavily_search_tool: Any, verbose: bool):
    def retrieve_docs(state: State) -> State:
        tasks = _tasks_for_route(state, "docs")
        if not tasks:
            return {}

        local_errors: list[str] = []
        evidence_updates: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []

        for index, task in enumerate(tasks, start=1):
            try:
                payload = tavily_search_tool.func(query=task.query)
            except Exception as exc:
                local_errors.append(f"tavily_search: failed ({exc})")
                payload = []

            parsed_items = parse_evidence_payload(payload, context="tool:tavily_search", errors=local_errors)
            payload_dicts = evidence_to_dicts(parsed_items)
            evidence_updates.extend(payload_dicts)
            tool_messages.append(_build_tool_message("tavily_search", payload_dicts, index))

        if verbose:
            print(f"[retrieve_docs] tasks={len(tasks)} evidence={len(evidence_updates)}")

        updates: State = {
            "retrieved_evidence": evidence_updates,
            "messages": tool_messages,
        }
        if local_errors:
            updates["retrieval_errors"] = local_errors
        return updates

    return retrieve_docs


def make_retrieve_upload_node(upload_search_tool: Any, verbose: bool):
    def retrieve_upload(state: State) -> State:
        tasks = _tasks_for_route(state, "upload")
        if not tasks:
            return {}

        local_errors: list[str] = []
        evidence_updates: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []

        for index, task in enumerate(tasks, start=1):
            try:
                payload = upload_search_tool.func(
                    query=task.query,
                    k=task.k,
                    retriever=state.get("retriever"),
                )
            except Exception as exc:
                local_errors.append(f"upload_search: failed ({exc})")
                payload = []

            parsed_items = parse_evidence_payload(payload, context="tool:upload_search", errors=local_errors)
            payload_dicts = evidence_to_dicts(parsed_items)
            evidence_updates.extend(payload_dicts)
            tool_messages.append(_build_tool_message("upload_search", payload_dicts, index))

        if verbose:
            print(f"[retrieve_upload] tasks={len(tasks)} evidence={len(evidence_updates)}")

        updates: State = {
            "retrieved_evidence": evidence_updates,
            "messages": tool_messages,
        }
        if local_errors:
            updates["retrieval_errors"] = local_errors
        return updates

    return retrieve_upload


def make_retrieve_local_node(rag_search_tool: Any, verbose: bool):
    def retrieve_local(state: State) -> State:
        tasks = _tasks_for_route(state, "local")
        if not tasks:
            return {}

        local_errors: list[str] = []
        evidence_updates: list[dict[str, Any]] = []
        tool_messages: list[ToolMessage] = []

        for index, task in enumerate(tasks, start=1):
            try:
                payload = rag_search_tool.func(query=task.query, k=task.k)
            except Exception as exc:
                local_errors.append(f"rag_search: failed ({exc})")
                payload = []

            parsed_items = parse_evidence_payload(payload, context="tool:rag_search", errors=local_errors)
            payload_dicts = evidence_to_dicts(parsed_items)
            evidence_updates.extend(payload_dicts)
            tool_messages.append(_build_tool_message("rag_search", payload_dicts, index))

        if verbose:
            print(f"[retrieve_local] tasks={len(tasks)} evidence={len(evidence_updates)}")

        updates: State = {
            "retrieved_evidence": evidence_updates,
            "messages": tool_messages,
        }
        if local_errors:
            updates["retrieval_errors"] = local_errors
        return updates

    return retrieve_local


def make_synthesize_node(llm_synthesizer: Any, verbose: bool, max_turns: int = 6):
    def synthesize(state: State):
        attempt = int(state.get("synthesis_attempt", 0)) + 1
        model_msgs: List[BaseMessage] = [m for m in state.get("messages", []) if not isinstance(m, ToolMessage)]

        if not model_msgs or not isinstance(model_msgs[0], SystemMessage):
            model_msgs = [SystemMessage(content=SYS_POLICY)] + model_msgs

        if state.get("memory_summary"):
            model_msgs = [
                model_msgs[0],
                SystemMessage(content=f"[Conversation Summary]\n{state['memory_summary']}"),
            ] + model_msgs[1:]

        parse_errors: list[str] = []
        parsed_evidence = parse_evidence_payload(
            state.get("retrieved_evidence", []),
            context="retrieved_evidence",
            errors=parse_errors,
        )
        deduped_evidence = evidence_to_dicts(dedupe_evidence(parsed_evidence))

        model_msgs.append(
            SystemMessage(content=f"[Retrieved Evidence]\n{_format_evidence_for_prompt(deduped_evidence)}")
        )
        if attempt > 1:
            model_msgs.append(
                SystemMessage(
                    content=(
                        "Retry synthesis after evidence validation failed. "
                        "Use retrieved evidence when available and avoid unsupported claims."
                    )
                )
            )

        before = len(model_msgs)
        model_msgs = _keep_recent_messages(model_msgs, max_turns=max_turns)
        after = len(model_msgs)
        if verbose and before != after:
            print(f"[synthesize] trimmed model messages: {before} -> {after}")

        response_obj = llm_synthesizer.invoke(model_msgs)
        if isinstance(response_obj, AIMessage):
            response = response_obj
        else:
            response = AIMessage(content=_extract_text_content(getattr(response_obj, "content", response_obj)))
        final_answer = _extract_text_content(response.content)

        updates: State = {
            "messages": [response],
            "final_answer": final_answer,
            "synthesis_attempt": attempt,
            "needs_retry": False,
        }
        if parse_errors:
            updates["retrieval_errors"] = parse_errors
        return updates

    return synthesize


def make_validate_evidence_node(verbose: bool):
    def validate_evidence(state: State) -> State:
        local_errors: list[str] = []
        parse_errors: list[str] = []

        planner_output = _coerce_planner_output(state.get("planner_output"), local_errors)
        parsed_evidence = parse_evidence_payload(
            state.get("retrieved_evidence", []),
            context="retrieved_evidence",
            errors=parse_errors,
        )
        local_errors.extend(parse_errors)

        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
        has_valid_evidence = len(parsed_evidence) > 0

        needs_retry = False
        if retrieval_required and not has_valid_evidence:
            local_errors.append("validate_evidence: retrieval was requested but no valid evidence was collected")
            if int(state.get("synthesis_attempt", 0)) < 2:
                needs_retry = True

        if verbose:
            print(
                f"[validate_evidence] required={retrieval_required} "
                f"evidence={len(parsed_evidence)} retry={needs_retry}"
            )

        updates: State = {
            "needs_retry": needs_retry,
        }
        if local_errors:
            updates["validation_errors"] = local_errors
        return updates

    return validate_evidence


def _extract_slack_destinations(messages: list[AnyMessage]) -> dict[str, str | None]:
    destinations: dict[str, str | None] = {
        "channel_id": None,
        "user_id": None,
        "email": None,
    }
    for message in reversed(messages):
        if not isinstance(message, SystemMessage):
            continue
        content = _extract_text_content(message.content)
        if "[Slack Destinations]" not in content:
            continue

        for line in content.splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in destinations and value:
                destinations[key] = value
        break
    return destinations


def make_action_postprocess_node(save_text_tool: Any, slack_notify_tool: Any, verbose: bool):
    def action_postprocess(state: State) -> State:
        user_input = str(state.get("user_input", "") or "")
        final_answer = str(state.get("final_answer", "") or "")
        action_errors: list[str] = []
        tool_messages: list[ToolMessage] = []

        if (needs_save(user_input) or needs_slack(user_input)) and not final_answer.strip():
            action_errors.append("postprocess: final_answer is empty, skipping save/slack actions")

        if needs_save(user_input) and final_answer.strip():
            try:
                save_result = save_text_tool.func(content=final_answer, filename_prefix="response")
            except Exception as exc:
                save_result = {"status": "error", "error": str(exc)}
                action_errors.append(f"save_text: failed ({exc})")
            tool_messages.append(_build_tool_message("save_text", save_result, 1))

        if needs_slack(user_input) and final_answer.strip():
            destinations = _extract_slack_destinations(state.get("messages", []))
            try:
                slack_result = slack_notify_tool.func(
                    text=final_answer,
                    user_id=destinations.get("user_id"),
                    email=destinations.get("email"),
                    channel_id=destinations.get("channel_id"),
                    target="auto",
                )
            except Exception as exc:
                slack_result = {"status": "error", "error": str(exc)}
                action_errors.append(f"slack_notify: failed ({exc})")
            tool_messages.append(_build_tool_message("slack_notify", slack_result, 1))

        if verbose and tool_messages:
            tool_names = ", ".join(msg.name for msg in tool_messages if msg.name)
            print(f"[postprocess] tools={tool_names}")

        updates: State = {}
        if tool_messages:
            updates["messages"] = tool_messages
        if action_errors:
            updates["action_errors"] = action_errors
        return updates

    return action_postprocess
