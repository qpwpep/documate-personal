from __future__ import annotations

from typing import Any
from src.core.contracts.graph_state import ResponseState


def parse_response_state(value: Any) -> ResponseState:
    if isinstance(value, ResponseState):
        return value
    if value is None:
        return ResponseState()
    return ResponseState.model_validate(value)


def get_response_state(state: dict[str, Any]) -> ResponseState:
    return parse_response_state(state.get("response"))
