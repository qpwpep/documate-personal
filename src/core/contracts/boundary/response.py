from __future__ import annotations

from src.core.contracts.graph_state import GraphState, ResponseState


def get_response_state(state: GraphState) -> ResponseState:
    return state["response"] if "response" in state else ResponseState()
