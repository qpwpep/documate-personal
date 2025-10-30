from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from .node import State, chatbot
from .tools import tavilysearch, save_text_tool

def build_graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")

    tool_node = ToolNode(tools=[tavilysearch, save_text_tool])
    builder.add_node("tools", tool_node)
    builder.add_conditional_edges("chatbot", tools_condition, {"tools": "tools", END: END})
    builder.add_edge("tools", "chatbot")
    return builder.compile()