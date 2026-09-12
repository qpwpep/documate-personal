from __future__ import annotations

from typing import Any
from pydantic import ValidationError
from src.core.answer_schema import AnswerResponse

from src.core.contracts.graph_state import PendingAction, RuntimeState, SessionMetadata, SlackDestination
from src.core.request_contracts import RequestContract, UserTurnSnapshot


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


def parse_request_contract(value: Any) -> RequestContract | None:
    if value is None:
        return None
    try:
        return RequestContract.model_validate(value.model_dump() if isinstance(value, RequestContract) else value)
    except (ValidationError, TypeError, ValueError):
        return RequestContract.invalid()


def parse_runtime_state(value: Any) -> RuntimeState:
    if isinstance(value, RuntimeState):
        return value
    if not isinstance(value, dict):
        return RuntimeState()
    contract = parse_request_contract(value.get("request_contract"))
    pending = None
    if value.get("pending_action") is not None:
        try:
            raw_pending = value["pending_action"]
            pending = PendingAction.model_validate(raw_pending.model_dump() if isinstance(raw_pending, PendingAction) else raw_pending)
        except (ValidationError, TypeError, ValueError):
            contract = RequestContract.invalid()
    return RuntimeState(
        user_input=str(value.get("user_input", "") or ""),
        current_turn_id=str(value.get("current_turn_id", "") or ""),
        user_turns=tuple(UserTurnSnapshot.model_validate(turn) for turn in value.get("user_turns", ())),
        retriever=value.get("retriever"),
        upload_files=tuple(value.get("upload_files", ())),
        session_metadata=parse_session_metadata(value.get("session_metadata")),
        memory_summary=(
            str(value.get("memory_summary")).strip()
            if value.get("memory_summary") is not None
            else None
        ),
        progress_emitter=value.get("progress_emitter"),
        previous_response=(AnswerResponse.model_validate(value["previous_response"]) if value.get("previous_response") is not None else None),
        request_contract=contract,
        pending_action=pending,
    )


def get_runtime_state(state: dict[str, Any]) -> RuntimeState:
    return parse_runtime_state(state.get("runtime"))
