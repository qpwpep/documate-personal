from __future__ import annotations

from src.core.contracts.graph_state import GraphState, RuntimeState, SessionMetadata


def parse_session_metadata(value: object) -> SessionMetadata:
    if isinstance(value, SessionMetadata):
        return value
    if value is None:
        return SessionMetadata()
    return SessionMetadata.model_validate(value)


def get_runtime_state(state: GraphState) -> RuntimeState:
    return state["runtime"]
