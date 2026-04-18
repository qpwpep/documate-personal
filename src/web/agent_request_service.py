from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..logging_utils import log_event
from .agent_request_support import build_session_metadata_snapshot, normalize_debug_info
from .cleanup import RuntimeCleaner, validate_upload_file_path
from .schemas import AgentDebugInfo, AgentRequest, AgentResponse, AgentResponsePayload
from .session_store import InMemorySessionStore


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AgentRequestResult:
    response: AgentResponsePayload
    trace: str
    file_path: str | None = None
    debug: AgentDebugInfo | None = None

    def to_response(self) -> AgentResponse:
        return AgentResponse(
            response=self.response,
            trace=self.trace,
            file_path=self.file_path,
            debug=self.debug,
        )


class AgentRequestService:
    def __init__(
        self,
        *,
        runtime_cleaner: RuntimeCleaner,
        session_store: InMemorySessionStore,
    ) -> None:
        self._runtime_cleaner = runtime_cleaner
        self._session_store = session_store

    async def run(
        self,
        *,
        request_id: str,
        request_data: AgentRequest,
    ) -> AgentRequestResult:
        user_query = request_data.query
        session_id = request_data.session_id
        self._runtime_cleaner.run_once(force=False, current_session_id=session_id)
        upload_file_path = validate_upload_file_path(request_data.upload_file_path, session_id)
        session_metadata = build_session_metadata_snapshot(request_data)
        agent_manager = self._session_store.get_or_create(session_id)

        log_event(
            logger,
            logging.INFO,
            "agent_request",
            session_id=session_id[:8],
            request_id=request_id,
            agent_id=id(agent_manager),
            query=user_query[:60],
            upload_file_path=upload_file_path,
        )

        started = time.monotonic()
        agent_manager, agent_answer, session_lock_wait_ms = await asyncio.to_thread(
            self._session_store.run_session_request,
            session_id=session_id,
            session_metadata=session_metadata,
            user_input=user_query,
            upload_file_path=upload_file_path,
        )
        latency_ms_server = int((time.monotonic() - started) * 1000)

        file_path = agent_answer.get("filepath", "")
        response_payload = _build_response_payload(agent_answer)
        debug_info = normalize_debug_info(
            raw_debug=agent_answer.get("debug"),
            latency_ms_server=latency_ms_server,
        )

        log_event(
            logger,
            logging.INFO,
            "agent_response",
            session_id=session_id[:8],
            request_id=request_id,
            agent_id=id(agent_manager),
            latency_ms_server=latency_ms_server,
            session_lock_wait_ms=session_lock_wait_ms,
            session_lock_contended=session_lock_wait_ms > 0,
            file_path=file_path,
        )

        return AgentRequestResult(
            response=response_payload,
            trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
            file_path=file_path,
            debug=debug_info if request_data.include_debug else None,
        )


def _build_response_payload(agent_answer: dict[str, Any]) -> AgentResponsePayload:
    answer = str(agent_answer.get("message") or "")
    response_payload_raw = agent_answer.get("response_payload")
    fallback_payload = {
        "answer": answer,
        "claims": [],
        "evidence": [],
        "confidence": None,
    }
    payload_candidate = response_payload_raw if isinstance(response_payload_raw, dict) else fallback_payload
    try:
        return AgentResponsePayload.model_validate(payload_candidate)
    except Exception:
        return AgentResponsePayload.model_validate(fallback_payload)
