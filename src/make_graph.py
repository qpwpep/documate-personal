from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph


def _summary_router(state: dict[str, Any], summary_max_turns: int) -> str:
    messages = state.get("messages")
    if not isinstance(messages, list):
        messages = []
    window_size = summary_max_turns * 2 + 2
    if len(messages) > window_size:
        return "summarize"
    return "planner"


def _planner_router(state: dict[str, Any]) -> str:
    planner_output = state.get("planner_output")
    use_retrieval = bool(getattr(planner_output, "use_retrieval", False))
    tasks = getattr(planner_output, "tasks", []) or []
    if use_retrieval and tasks:
        return "retrieve"
    return "synthesize"


def _validate_router(state: dict[str, Any]) -> str:
    if bool(state.get("needs_retry")):
        return "retry"
    return "postprocess"


def build_graph(
    state_type: Any,
    add_user_node: Any,
    summarize_node: Any,
    planner_node: Any,
    retrieve_dispatch_node: Any,
    synthesize_node: Any,
    validate_evidence_node: Any,
    action_postprocess_node: Any,
    summary_max_turns: int = 6,
):
    builder = StateGraph(state_type)

    builder.add_node("add_user_message", add_user_node)
    builder.set_entry_point("add_user_message")

    builder.add_node("summarize_old_messages", summarize_node)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve_dispatch", retrieve_dispatch_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("validate_evidence", validate_evidence_node)
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
            "retrieve": "retrieve_dispatch",
            "synthesize": "synthesize",
        },
    )
    builder.add_edge("retrieve_dispatch", "synthesize")
    builder.add_edge("synthesize", "validate_evidence")
    builder.add_conditional_edges(
        "validate_evidence",
        _validate_router,
        {
            "retry": "planner",
            "postprocess": "action_postprocess",
        },
    )
    builder.add_edge("action_postprocess", END)

    return builder.compile()
