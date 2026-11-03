from __future__ import annotations

from src.core.contracts import SessionMetadata
from src.core.contracts.debug import DEBUG_REQUIRED_FIELDS
from src.app.transport_contracts import AgentDebugInfo
from src.app.web.schemas import AgentRequest


def build_debug_info(
    raw_debug: dict[str, object],
    latency_ms_server: int | None,
    *,
    answer_expected: bool = True,
) -> AgentDebugInfo:
    """Check the runtime result at the HTTP boundary without repairing its fields."""
    missing = [field for field in DEBUG_REQUIRED_FIELDS if field not in raw_debug]
    if missing:
        raise ValueError(f"Runtime debug is missing required observations: {', '.join(missing)}")
    debug = AgentDebugInfo.model_validate(raw_debug)
    if answer_expected and debug.answer_provenance is None:
        raise ValueError("An answer requires its recorded provenance")
    debug.latency_ms_server = latency_ms_server
    if debug.latency_breakdown is not None:
        debug.latency_breakdown.server_total_ms = latency_ms_server
    return debug


def build_session_metadata_snapshot(request_data: AgentRequest) -> SessionMetadata:
    return SessionMetadata(slack_recipient=request_data.slack_recipient)
