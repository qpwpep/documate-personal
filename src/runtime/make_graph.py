from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.runtime.nodes.session import keep_recent_messages


def _record_edge_decision(
    state: dict[str, Any],
    *,
    source: str,
    decision: str,
    reason: str,
) -> None:
    debug = get_debug_state(state)
    decision_event = {
        "source": source,
        "decision": decision,
        "reason": reason,
    }
    state["debug"] = debug.model_copy(
        update={"edge_decisions": [*debug.edge_decisions, decision_event]}
    )


def _summary_router(state: dict[str, Any], summary_max_turns: int) -> str:
    messages = state.get("messages")
    if not isinstance(messages, list):
        messages = []
    recent_window = keep_recent_messages(messages, max_turns=summary_max_turns)
    if len(recent_window) < len(messages):
        _record_edge_decision(
            state,
            source="add_user_message",
            decision="summarize",
            reason="history_exceeds_summary_window",
        )
        return "summarize"
    _record_edge_decision(
        state,
        source="add_user_message",
        decision="planner",
        reason="history_within_summary_window",
    )
    return "planner"


def _planner_router(state: dict[str, Any]) -> str:
    planner = get_planner_state(state)
    if str(planner.guided_followup or "").strip():
        _record_edge_decision(
            state,
            source="planner",
            decision="pre_validate",
            reason="guided_followup_present",
        )
        return "pre_validate"
    planner_output = planner.output
    use_retrieval = bool(getattr(planner_output, "use_retrieval", False))
    tasks = getattr(planner_output, "tasks", []) or []
    if use_retrieval and tasks:
        _record_edge_decision(
            state,
            source="planner",
            decision="retrieve",
            reason=f"retrieval_required:{len(tasks)}_task(s)",
        )
        return "retrieve"
    _record_edge_decision(
        state,
        source="planner",
        decision="synthesize",
        reason="retrieval_not_required",
    )
    return "synthesize"


def _pre_synthesis_router(state: dict[str, Any]) -> str:
    if get_retry_state(state).needs_retry:
        _record_edge_decision(
            state,
            source="pre_synthesis_validation",
            decision="retry",
            reason=str(get_retry_state(state).retry_reason or "retry_requested"),
        )
        return "retry"
    response = get_response_state(state)
    if str(response.final_answer or "").strip() or str(response.payload.answer or "").strip():
        _record_edge_decision(
            state,
            source="pre_synthesis_validation",
            decision="postprocess",
            reason="terminal_response_available",
        )
        return "postprocess"
    _record_edge_decision(
        state,
        source="pre_synthesis_validation",
        decision="synthesize",
        reason="validation_passed",
    )
    return "synthesize"


def _post_synthesis_router(state: dict[str, Any]) -> str:
    if get_retry_state(state).needs_retry:
        _record_edge_decision(
            state,
            source="post_synthesis_validation",
            decision="retry",
            reason=str(get_retry_state(state).retry_reason or "retry_requested"),
        )
        return "retry"
    _record_edge_decision(
        state,
        source="post_synthesis_validation",
        decision="postprocess",
        reason=str(get_retry_state(state).retry_reason or "validation_passed"),
    )
    return "postprocess"


def build_graph(
    state_type: Any,
    add_user_node: Any,
    summarize_node: Any,
    planner_node: Any,
    retrieve_dispatch_node: Any,
    synthesize_node: Any,
    validate_evidence_node: Any | None = None,
    action_postprocess_node: Any | None = None,
    summary_max_turns: int = 6,
    *,
    pre_synthesis_validation_node: Any | None = None,
    post_synthesis_validation_node: Any | None = None,
):
    builder = StateGraph(state_type)
    pre_synthesis_validation_node = pre_synthesis_validation_node or (lambda _state: {})
    post_synthesis_validation_node = post_synthesis_validation_node or validate_evidence_node or (lambda _state: {})
    action_postprocess_node = action_postprocess_node or (lambda _state: {})

    builder.add_node("add_user_message", add_user_node)
    builder.set_entry_point("add_user_message")

    builder.add_node("summarize_old_messages", summarize_node)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve_dispatch", retrieve_dispatch_node)
    builder.add_node("pre_synthesis_validation", pre_synthesis_validation_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("post_synthesis_validation", post_synthesis_validation_node)
    builder.add_node("action_postprocess", action_postprocess_node)

    builder.add_conditional_edges(
        "add_user_message",
        lambda state: _summary_router(state, summary_max_turns),
        {
            "summarize": "summarize_old_messages",
            "planner": "planner",
        },
    )
    builder.add_edge("summarize_old_messages", "planner")
    builder.add_conditional_edges(
        "planner",
        _planner_router,
        {
            "pre_validate": "pre_synthesis_validation",
            "retrieve": "retrieve_dispatch",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("retrieve_dispatch", "pre_synthesis_validation")
    builder.add_conditional_edges(
        "pre_synthesis_validation",
        _pre_synthesis_router,
        {
            "retry": "planner",
            "synthesize": "synthesize",
            "postprocess": "action_postprocess",
        },
    )
    builder.add_edge("synthesize", "post_synthesis_validation")
    builder.add_conditional_edges(
        "post_synthesis_validation",
        _post_synthesis_router,
        {
            "retry": "planner",
            "postprocess": "action_postprocess",
        },
    )
    builder.add_edge("action_postprocess", END)

    return builder.compile()
