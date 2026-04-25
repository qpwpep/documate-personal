from __future__ import annotations

from typing import Any

from src.core.contracts.graph_state import RuntimeState, SessionMetadata, SlackDestination


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
    planner_mode = str(value.get("planner_mode") or "auto").strip()
    if planner_mode not in {"auto", "force_llm"}:
        planner_mode = "auto"
    raw_eval_faults = value.get("eval_faults")
    eval_faults = {
        str(key): str(item)
        for key, item in raw_eval_faults.items()
        if str(key).strip() and str(item).strip()
    } if isinstance(raw_eval_faults, dict) else {}
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
        planner_mode=planner_mode,  # type: ignore[arg-type]
        eval_faults=eval_faults,
    )


def get_runtime_state(state: dict[str, Any]) -> RuntimeState:
    return parse_runtime_state(state.get("runtime"))
