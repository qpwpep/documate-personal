from typing import Any

from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_graph(
    state_type: Any,
    add_user_node: Any,
    summarize_node: Any,
    chatbot_node: Any,
    tool_list: list[Any],
):
    builder = StateGraph(state_type)

    builder.add_node("add_user_message", add_user_node)
    builder.set_entry_point("add_user_message")

    builder.add_node("summarize_old_messages", summarize_node)
    builder.add_node("chatbot", chatbot_node)

    builder.add_edge("add_user_message", "summarize_old_messages")
    builder.add_edge("summarize_old_messages", "chatbot")

    tool_node = ToolNode(tools=tool_list)
    builder.add_node("tools", tool_node)
    builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "chatbot")

    return builder.compile()
