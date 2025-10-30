# edge.py
from .node import GraphState


def need_tools(state: GraphState):
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else "finalize"