import time
from typing import Any

from .latency import elapsed_ms, make_stage_latency_event
from .llm import build_llm_registry
from .make_graph import build_graph
from .nodes.actions import make_action_postprocess_node
from .nodes.planner import make_planner_node
from .nodes.retrieval import make_retrieve_dispatch_node
from .nodes.session import add_user_message, make_summarize_node
from .nodes.state import State, coerce_retry_context
from .nodes.synthesis import make_synthesize_node
from .nodes.validation import make_validate_evidence_node
from .settings import AppSettings, get_settings
from .tools import build_tool_registry


def _resolve_stage_attempt(stage: str, state: State, updates: State) -> int:
    if stage in {"planner"}:
        retry_context = coerce_retry_context(state.get("retry_context"))
        return int(retry_context.get("attempt", 0)) + 1
    if stage == "validation":
        return max(1, int(state.get("synthesis_attempt", 0) or 0))
    if stage == "action_postprocess":
        return max(1, int(state.get("synthesis_attempt", 0) or 1))
    return 1


def _resolve_stage_status(stage: str, updates: State) -> str | None:
    if stage == "planner":
        status = updates.get("planner_status")
        return str(status) if status else None
    if stage == "validation":
        return "retry" if bool(updates.get("needs_retry")) else "pass"
    return None


def _instrument_stage_node(stage: str, node: Any):
    def wrapped(state: State) -> State:
        started = time.perf_counter()
        updates = node(state)
        if not isinstance(updates, dict):
            return updates

        latency_trace = list(updates.get("latency_trace") or [])
        latency_trace.append(
            make_stage_latency_event(
                stage=stage,  # type: ignore[arg-type]
                attempt=_resolve_stage_attempt(stage, state, updates),
                latency_ms=elapsed_ms(started, time.perf_counter()),
                status=_resolve_stage_status(stage, updates),
            )
        )
        updates["latency_trace"] = latency_trace
        return updates

    return wrapped


def build_agent_graph(settings: AppSettings | None = None):
    app_settings = settings or get_settings()
    has_default_slack_destination = bool(
        app_settings.slack_default_user_id or app_settings.slack_default_dm_email
    )

    tool_registry = build_tool_registry(app_settings)
    llm_registry = build_llm_registry(app_settings)

    summarize_node = make_summarize_node(
        llm_summarizer=llm_registry.llm_summarizer,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    summarize_node = _instrument_stage_node("summarize", summarize_node)
    planner_node = make_planner_node(
        llm_planner=llm_registry.llm_planner,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    planner_node = _instrument_stage_node("planner", planner_node)
    retrieve_dispatch_node = make_retrieve_dispatch_node(
        tavily_search_tool=tool_registry.tavily_search_tool,
        upload_search_tool=tool_registry.upload_search_tool,
        rag_search_tool=tool_registry.rag_search_tool,
        verbose=llm_registry.verbose,
    )
    synthesize_node = make_synthesize_node(
        llm_synthesizer=llm_registry.llm_synthesizer,
        verbose=llm_registry.verbose,
        max_turns=6,
        has_default_slack_destination=has_default_slack_destination,
    )
    validate_evidence_node = make_validate_evidence_node(verbose=llm_registry.verbose)
    validate_evidence_node = _instrument_stage_node("validation", validate_evidence_node)
    action_postprocess_node = make_action_postprocess_node(
        save_text_tool=tool_registry.save_text_tool,
        slack_notify_tool=tool_registry.slack_notify_tool,
        verbose=llm_registry.verbose,
        has_default_slack_destination=has_default_slack_destination,
    )
    action_postprocess_node = _instrument_stage_node(
        "action_postprocess",
        action_postprocess_node,
    )

    graph_object = build_graph(
        state_type=State,
        add_user_node=add_user_message,
        summarize_node=summarize_node,
        planner_node=planner_node,
        retrieve_dispatch_node=retrieve_dispatch_node,
        synthesize_node=synthesize_node,
        validate_evidence_node=validate_evidence_node,
        action_postprocess_node=action_postprocess_node,
        summary_max_turns=6,
    )
    return graph_object
