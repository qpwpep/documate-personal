from langgraph.graph import StateGraph, START
from .node import State, chatbot
from .tools import tavilysearch, save_text_tool
from .edge import wire_tool_edges

def build_graph():
    builder = StateGraph(State)
    builder.add_node("chatbot", chatbot)
    builder.add_edge(START, "chatbot")

    # Wire tools and conditional edges
    wire_tool_edges(builder, tavily_tool=tavilysearch)

    # Compile into a runnable app/graph
    return builder.compile()
