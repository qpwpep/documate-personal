from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Any

from src.infra.logging_utils import log_event
from src.runtime.progress import ProgressEmitter
from src.app.web.agent_request_support import build_session_metadata_snapshot, normalize_debug_info
from src.app.web.cleanup import RuntimeCleaner, validate_upload_file_paths
from src.app.web.schemas import AgentDebugInfo, AgentRequest, AgentResponse, AgentResponsePayload, AgentStreamEvent
from src.app.web.session_store import InMemorySessionStore


logger = logging.getLogger(__name__)
_STREAM_DONE = object()


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
        return await asyncio.to_thread(
            self._execute_request,
            request_id=request_id,
            request_data=request_data,
            progress_emitter=None,
        )

    def stream(
        self,
        *,
        request_id: str,
        request_data: AgentRequest,
    ) -> AsyncIterator[AgentStreamEvent]:
        event_queue: Queue[AgentStreamEvent | object] = Queue()

        def publish(event: str, data: dict[str, Any]) -> None:
            event_queue.put(AgentStreamEvent(event=event, data=dict(data)))

        progress_emitter = ProgressEmitter(
            publish=publish,
            request_id=request_id,
            session_id=request_data.session_id,
        )
        progress_emitter.emit_request_started()

        def worker() -> None:
            try:
                result = self._execute_request(
                    request_id=request_id,
                    request_data=request_data,
                    progress_emitter=progress_emitter,
                )
                progress_emitter.emit_final_response(
                    result.to_response().model_dump(mode="json")
                )
            except Exception as exc:
                progress_emitter.emit_error(message=str(exc), stage=None)
            finally:
                progress_emitter.emit_done()
                event_queue.put(_STREAM_DONE)

        Thread(
            target=worker,
            name=f"agent-stream-{request_id}",
            daemon=True,
        ).start()

        async def event_stream() -> AsyncIterator[AgentStreamEvent]:
            while True:
                item = await asyncio.to_thread(event_queue.get)
                if item is _STREAM_DONE:
                    break
                yield item

        return event_stream()

    def _execute_request(
        self,
        *,
        request_id: str,
        request_data: AgentRequest,
        progress_emitter: ProgressEmitter | None,
    ) -> AgentRequestResult:
        user_query = request_data.query
        session_id = request_data.session_id
        self._runtime_cleaner.run_once(force=False, current_session_id=session_id)
        upload_file_paths = validate_upload_file_paths(
            upload_file_path=request_data.upload_file_path,
            upload_file_paths=request_data.upload_file_paths,
            session_id=session_id,
        )
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
            upload_file_paths=upload_file_paths,
            planner_mode=request_data.planner_mode,
        )

        started = time.monotonic()
        run_request_kwargs = {
            "session_id": session_id,
            "session_metadata": session_metadata,
            "user_input": user_query,
            "upload_file_paths": upload_file_paths,
            "planner_mode": request_data.planner_mode,
            "eval_faults": request_data.eval_faults if request_data.include_debug else {},
        }
        if progress_emitter is not None:
            run_request_kwargs["progress_emitter"] = progress_emitter
        agent_manager, agent_answer, session_lock_wait_ms = self._session_store.run_session_request(
            **run_request_kwargs
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
