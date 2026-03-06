from __future__ import annotations

import json
from typing import Annotated, Any, List, Literal, Optional

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

RetryReason = Literal["no_evidence", "low_score", "tool_error"]
LOW_SCORE_THRESHOLD = 0.5
DEFAULT_MAX_RETRIES = 1
RETRYABLE_REASONS: set[RetryReason] = {"no_evidence", "low_score"}

UNCERTAINTY_ANSWER_PREFIX = (
    "There is not enough reliable evidence to provide a confident final answer. "
    "Please narrow the question scope or broaden retrieval targets and try again."
)


class RetryContext(TypedDict, total=False):
    attempt: int
    max_retries: int
    retry_reason: RetryReason
    retrieval_feedback: str
    evidence_start_index: int
    retrieval_error_start_index: int
    score_avg: float | None


def _safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def _slice_from_index(items: list[Any], start_index: int) -> list[Any]:
    if start_index < 0:
        start_index = 0
    if start_index >= len(items):
        return []
    return items[start_index:]


def _coerce_retry_context(value: Any) -> RetryContext:
    context: RetryContext = {
        "attempt": 0,
        "max_retries": DEFAULT_MAX_RETRIES,
        "retrieval_feedback": "",
        "evidence_start_index": 0,
        "retrieval_error_start_index": 0,
        "score_avg": None,
    }
    if not isinstance(value, dict):
        return context

    attempt = value.get("attempt")
    if isinstance(attempt, int) and attempt >= 0:
        context["attempt"] = attempt

    max_retries = value.get("max_retries")
    if isinstance(max_retries, int) and max_retries >= 0:
        context["max_retries"] = max_retries

    retry_reason = value.get("retry_reason")
    if retry_reason in {"no_evidence", "low_score", "tool_error"}:
        context["retry_reason"] = retry_reason

    retrieval_feedback = value.get("retrieval_feedback")
    if retrieval_feedback is not None:
        context["retrieval_feedback"] = str(retrieval_feedback).strip()

    evidence_start_index = value.get("evidence_start_index")
    if isinstance(evidence_start_index, int) and evidence_start_index >= 0:
        context["evidence_start_index"] = evidence_start_index

    retrieval_error_start_index = value.get("retrieval_error_start_index")
    if isinstance(retrieval_error_start_index, int) and retrieval_error_start_index >= 0:
        context["retrieval_error_start_index"] = retrieval_error_start_index

    score_avg = value.get("score_avg")
    if isinstance(score_avg, (int, float)):
        context["score_avg"] = float(score_avg)
    elif score_avg is None:
        context["score_avg"] = None

    return context


def _format_retry_context_for_planner(state: State, retry_context: RetryContext) -> str | None:
    attempt = int(retry_context.get("attempt", 0))
    if attempt <= 0:
        return None

    max_retries = int(retry_context.get("max_retries", DEFAULT_MAX_RETRIES))
    retry_reason = str(retry_context.get("retry_reason") or "no_evidence")
    feedback = str(retry_context.get("retrieval_feedback") or "none")
    score_avg = retry_context.get("score_avg")
    score_text = f"{score_avg:.3f}" if isinstance(score_avg, (int, float)) else "n/a"

    planner_parse_errors: list[str] = []
    previous_output = _coerce_planner_output(state.get("planner_output"), planner_parse_errors)
    if previous_output.use_retrieval and previous_output.tasks:
        prev_tasks = ", ".join(
            f"{task.route}:{task.query}(k={task.k})" for task in previous_output.tasks
        )
    else:
        prev_tasks = "none"

    return (
        "[Retry Context]\n"
        f"attempt={attempt}/{max_retries}\n"
        f"reason={retry_reason}\n"
        f"retrieval_feedback={feedback}\n"
        f"previous_tasks={prev_tasks}\n"
        f"score_avg={score_text}\n"
        "Reformulate query scope and switch routes if needed."
    )


def _contains_tool_error(errors: list[str]) -> bool:
    if not errors:
        return False
    keywords = (
        "failed",
        "error",
        "unavailable",
        "invalid json",
        "payload must",
        "timeout",
    )
    for error in errors:
        lowered = str(error).lower()
        if any(keyword in lowered for keyword in keywords):
            return True
    return False


def _build_retrieval_feedback(
    reason: RetryReason,
    *,
    planner_output: PlannerOutput,
    retrieval_errors: list[str],
    score_avg: float | None,
) -> str:
    if reason == "tool_error":
        if any("upload" in str(error).lower() and "unavailable" in str(error).lower() for error in retrieval_errors):
            return "upload retriever unavailable; switch to docs/local routes."
        return "retrieval tool error detected; broaden query and simplify route strategy."
    if reason == "no_evidence":
        selected_routes = ", ".join(task.route for task in planner_output.tasks) if planner_output.tasks else "none"
        return f"query too narrow or domain mismatch on routes: {selected_routes}"
    if score_avg is not None:
        return f"low evidence confidence(avg_score={score_avg:.3f}); broaden query or switch route."
    return "low evidence confidence; broaden query or switch route."


def _build_uncertainty_answer(reason: RetryReason, feedback: str) -> str:
    return (
        f"{UNCERTAINTY_ANSWER_PREFIX}\n"
        f"- retry_reason: {reason}\n"
        f"- retrieval_feedback: {feedback}"
    )


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
    retry_context: RetryContext


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

    retry_context = _coerce_retry_context(state.get("retry_context"))
    retry_context_message = _format_retry_context_for_planner(state, retry_context)
    if retry_context_message:
        model_msgs.append(SystemMessage(content=retry_context_message))

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
        existing_retry_context = _coerce_retry_context(state.get("retry_context"))

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

        retry_context: RetryContext = dict(existing_retry_context)
        retry_context["max_retries"] = int(
            existing_retry_context.get("max_retries", DEFAULT_MAX_RETRIES)
        )
        retry_context["evidence_start_index"] = len(_safe_list(state.get("retrieved_evidence")))
        retry_context["retrieval_error_start_index"] = len(_safe_list(state.get("retrieval_errors")))
        # On the first attempt, clear stale retry metadata.
        if int(retry_context.get("attempt", 0)) <= 0:
            retry_context["retrieval_feedback"] = ""
            retry_context["score_avg"] = None
            retry_context.pop("retry_reason", None)

        return {
            "planner_output": planner_output,
            "retrieval_errors": planner_errors,
            "synthesis_attempt": int(state.get("synthesis_attempt", 0)),
            "needs_retry": False,
            "retry_context": retry_context,
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


def _run_retrieval_tasks(
    tasks: list[RetrievalTask],
    *,
    tool_name: str,
    context_label: str,
    invoke_tool: Any,
    verbose: bool,
) -> State:
    if not tasks:
        return {}

    local_errors: list[str] = []
    evidence_updates: list[dict[str, Any]] = []
    tool_messages: list[ToolMessage] = []

    for index, task in enumerate(tasks, start=1):
        try:
            payload = invoke_tool(task)
        except Exception as exc:
            local_errors.append(f"{tool_name}: failed ({exc})")
            payload = []

        parsed_items = parse_evidence_payload(payload, context=f"tool:{tool_name}", errors=local_errors)
        payload_dicts = evidence_to_dicts(parsed_items)
        evidence_updates.extend(payload_dicts)
        tool_messages.append(_build_tool_message(tool_name, payload_dicts, index))

    if verbose:
        print(f"[{context_label}] tasks={len(tasks)} evidence={len(evidence_updates)}")

    updates: State = {
        "retrieved_evidence": evidence_updates,
        "messages": tool_messages,
    }
    if local_errors:
        updates["retrieval_errors"] = local_errors
    return updates


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
        return _run_retrieval_tasks(
            tasks,
            tool_name="tavily_search",
            context_label="retrieve_docs",
            invoke_tool=lambda task: tavily_search_tool.func(query=task.query),
            verbose=verbose,
        )

    return retrieve_docs


def make_retrieve_upload_node(upload_search_tool: Any, verbose: bool):
    def retrieve_upload(state: State) -> State:
        tasks = _tasks_for_route(state, "upload")
        return _run_retrieval_tasks(
            tasks,
            tool_name="upload_search",
            context_label="retrieve_upload",
            invoke_tool=lambda task: upload_search_tool.func(
                query=task.query,
                k=task.k,
                retriever=state.get("retriever"),
            ),
            verbose=verbose,
        )

    return retrieve_upload


def make_retrieve_local_node(rag_search_tool: Any, verbose: bool):
    def retrieve_local(state: State) -> State:
        tasks = _tasks_for_route(state, "local")
        return _run_retrieval_tasks(
            tasks,
            tool_name="rag_search",
            context_label="retrieve_local",
            invoke_tool=lambda task: rag_search_tool.func(query=task.query, k=task.k),
            verbose=verbose,
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
        planner_output = _coerce_planner_output(state.get("planner_output"), planner_errors)
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
        tool_messages: list[ToolMessage] = []
        tool_call_counts: dict[str, int] = {}

        for task in planner_output.tasks:
            handler = route_handlers.get(task.route)
            if handler is None:
                local_errors.append(f"planner: unsupported route ({task.route})")
                continue

            tool_name, invoke_tool = handler
            try:
                payload = invoke_tool(task)
            except Exception as exc:
                local_errors.append(f"{tool_name}: failed ({exc})")
                payload = []

            parsed_items = parse_evidence_payload(payload, context=f"tool:{tool_name}", errors=local_errors)
            payload_dicts = evidence_to_dicts(parsed_items)
            evidence_updates.extend(payload_dicts)

            tool_call_counts[tool_name] = tool_call_counts.get(tool_name, 0) + 1
            tool_messages.append(
                _build_tool_message(tool_name, payload_dicts, tool_call_counts[tool_name])
            )

        if verbose:
            routes = ",".join(task.route for task in planner_output.tasks)
            print(
                f"[retrieve_dispatch] tasks={len(planner_output.tasks)} "
                f"routes={routes} evidence={len(evidence_updates)}"
            )

        updates: State = {
            "retrieved_evidence": evidence_updates,
            "messages": tool_messages,
        }
        if local_errors:
            updates["retrieval_errors"] = local_errors
        return updates

    return retrieve_dispatch


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

        retry_context = _coerce_retry_context(state.get("retry_context"))
        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        current_attempt_evidence_payload = _slice_from_index(
            _safe_list(state.get("retrieved_evidence")),
            evidence_start_index,
        )

        parse_errors: list[str] = []
        parsed_evidence = parse_evidence_payload(
            current_attempt_evidence_payload,
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
        retry_context = _coerce_retry_context(state.get("retry_context"))
        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        retrieval_error_start_index = int(retry_context.get("retrieval_error_start_index", 0))

        current_attempt_evidence_payload = _slice_from_index(
            _safe_list(state.get("retrieved_evidence")),
            evidence_start_index,
        )
        parsed_evidence = parse_evidence_payload(
            current_attempt_evidence_payload,
            context="retrieved_evidence",
            errors=parse_errors,
        )
        local_errors.extend(parse_errors)

        current_attempt_retrieval_errors = [
            str(error)
            for error in _slice_from_index(
                _safe_list(state.get("retrieval_errors")),
                retrieval_error_start_index,
            )
            if str(error).strip()
        ]

        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
        has_valid_evidence = len(parsed_evidence) > 0

        score_values = [
            float(item.score)
            for item in parsed_evidence
            if item.score is not None
        ]
        score_avg = (sum(score_values) / len(score_values)) if score_values else None
        low_score = bool(
            retrieval_required
            and score_avg is not None
            and score_avg < LOW_SCORE_THRESHOLD
        )
        tool_error = bool(
            retrieval_required
            and (
                _contains_tool_error(current_attempt_retrieval_errors)
                or _contains_tool_error(parse_errors)
            )
        )

        retry_reason: RetryReason | None = None
        if tool_error:
            retry_reason = "tool_error"
        elif retrieval_required and not has_valid_evidence:
            retry_reason = "no_evidence"
        elif low_score:
            retry_reason = "low_score"

        max_retries = int(retry_context.get("max_retries", DEFAULT_MAX_RETRIES))
        used_retries = int(retry_context.get("attempt", 0))
        needs_retry = False
        retrieval_feedback = ""
        if retry_reason is not None:
            retrieval_feedback = _build_retrieval_feedback(
                retry_reason,
                planner_output=planner_output,
                retrieval_errors=current_attempt_retrieval_errors + parse_errors,
                score_avg=score_avg,
            )
            local_errors.append(
                "validate_evidence: retry_reason="
                f"{retry_reason}, score_avg={score_avg}, feedback={retrieval_feedback}"
            )
            if retry_reason in RETRYABLE_REASONS and used_retries < max_retries:
                needs_retry = True
                used_retries += 1

        if verbose:
            print(
                f"[validate_evidence] required={retrieval_required} "
                f"evidence={len(parsed_evidence)} retry={needs_retry} reason={retry_reason}"
            )

        next_retry_context: RetryContext = dict(retry_context)
        next_retry_context["attempt"] = used_retries
        next_retry_context["max_retries"] = max_retries
        next_retry_context["score_avg"] = score_avg
        if retry_reason is not None:
            next_retry_context["retry_reason"] = retry_reason
            next_retry_context["retrieval_feedback"] = retrieval_feedback
        else:
            next_retry_context["retrieval_feedback"] = ""
            next_retry_context.pop("retry_reason", None)

        updates: State = {
            "needs_retry": needs_retry,
            "retry_context": next_retry_context,
        }
        if retry_reason is not None and not needs_retry:
            uncertainty_answer = _build_uncertainty_answer(retry_reason, retrieval_feedback)
            updates["messages"] = [AIMessage(content=uncertainty_answer)]
            updates["final_answer"] = uncertainty_answer
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


