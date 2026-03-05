from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph


def _validate_router(state: dict[str, Any]) -> str:
    if bool(state.get("needs_retry")):
        return "retry"
    return "postprocess"


def build_graph(
    state_type: Any,
    add_user_node: Any,
    summarize_node: Any,
    planner_node: Any,
    retrieve_docs_node: Any,
    retrieve_upload_node: Any,
    retrieve_local_node: Any,
    synthesize_node: Any,
    validate_evidence_node: Any,
    action_postprocess_node: Any,
):
    builder = StateGraph(state_type)

    builder.add_node("add_user_message", add_user_node)
    builder.set_entry_point("add_user_message")

    builder.add_node("summarize_old_messages", summarize_node)
    builder.add_node("planner", planner_node)
    builder.add_node("retrieve_docs", retrieve_docs_node)
    builder.add_node("retrieve_upload", retrieve_upload_node)
    builder.add_node("retrieve_local", retrieve_local_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("validate_evidence", validate_evidence_node)
    builder.add_node("action_postprocess", action_postprocess_node)

    builder.add_edge("add_user_message", "summarize_old_messages")
    builder.add_edge("summarize_old_messages", "planner")

    builder.add_edge("planner", "retrieve_docs")
    builder.add_edge("planner", "retrieve_upload")
    builder.add_edge("planner", "retrieve_local")

    builder.add_edge(["retrieve_docs", "retrieve_upload", "retrieve_local"], "synthesize")
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
