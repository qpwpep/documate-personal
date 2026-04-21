from __future__ import annotations

from typing import Any

from ..graph_state import RuntimeState, SessionMetadata, SlackDestination


def parse_slack_destination(value: Any) -> SlackDestination:
    if isinstance(value, SlackDestination):
        return value
    if not isinstance(value, dict):
        return SlackDestination()

    destination = SlackDestination()
    for key in ("channel_id", "user_id", "email"):
        raw_item = value.get(key)
        if raw_item is None:
            setattr(destination, key, None)
            continue
        text = str(raw_item).strip()
        setattr(destination, key, text or None)
    return destination


def parse_session_metadata(value: Any) -> SessionMetadata:
    if isinstance(value, SessionMetadata):
        return value
    if not isinstance(value, dict):
        return SessionMetadata()

    destination = parse_slack_destination(value.get("slack_destination"))
    if destination.has_destination():
        return SessionMetadata(slack_destination=destination)
    return SessionMetadata()


def parse_runtime_state(value: Any) -> RuntimeState:
    if isinstance(value, RuntimeState):
        return value
    if not isinstance(value, dict):
        return RuntimeState()
    return RuntimeState(
        user_input=str(value.get("user_input", "") or ""),
        retriever=value.get("retriever"),
        session_metadata=parse_session_metadata(value.get("session_metadata")),
        memory_summary=(
            str(value.get("memory_summary")).strip()
            if value.get("memory_summary") is not None
            else None
        ),
        progress_emitter=value.get("progress_emitter"),
    )


def get_runtime_state(state: dict[str, Any]) -> RuntimeState:
    return parse_runtime_state(state.get("runtime"))
