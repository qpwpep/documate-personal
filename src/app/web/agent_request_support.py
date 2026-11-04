from __future__ import annotations

from src.core.contracts import SessionMetadata
from src.core.contracts.debug import DebugPayload
from src.app.web.schemas import AgentRequest


def build_debug_info(
    debug: DebugPayload,
    latency_ms_server: int | None,
    *,
    answer_expected: bool = True,
) -> DebugPayload:
    """Add transport timing to completed, typed runtime observations."""
    if answer_expected and debug.answer_provenance is None:
        raise ValueError("An answer requires its recorded provenance")
    latency = debug.latency_breakdown
    return debug.model_copy(update={
        "latency_ms_server": latency_ms_server,
        "latency_breakdown": latency.model_copy(update={"server_total_ms": latency_ms_server})
        if latency is not None else None,
    })


def build_session_metadata_snapshot(request_data: AgentRequest) -> SessionMetadata:
    return SessionMetadata(slack_recipient=request_data.slack_recipient)
