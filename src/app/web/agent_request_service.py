from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from queue import Queue
from threading import Thread
from typing import Any

from src.infra.logging_utils import log_event
from src.runtime.progress import ProgressEmitter
from src.app.web.agent_request_support import build_session_metadata_snapshot, normalize_debug_info
from src.app.web.cleanup import RuntimeCleaner
from src.app.web.schemas import AgentDebugInfo, AgentRequest, AgentResponse, AgentStreamEvent
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from src.core.contracts.outcome import TurnResult
from src.core.llm_errors import ExecutionProblem, LLMCallError, make_problem
from src.core.uploads import UploadManifest
from src.app.web.session_store import InMemorySessionStore


logger = logging.getLogger(__name__)
_STREAM_DONE = object()


def _query_log_fields(query: str) -> dict[str, Any]:
    normalized_query = str(query or "")
    encoded_query = normalized_query.encode("utf-8", errors="replace")
    return {
        "query_length": len(normalized_query),
        "query_utf8_bytes": len(encoded_query),
        "query_hash": hashlib.sha256(encoded_query).hexdigest(),
    }


class AgentRequestService:
    def __init__(
        self,
        *,
        runtime_cleaner: RuntimeCleaner,
        session_store: InMemorySessionStore,
    ) -> None:
        self._runtime_cleaner = runtime_cleaner
        self._session_store = session_store

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
                progress_emitter.emit_final_response(result.model_dump(mode="json"))
            except Exception as exc:
                # Infrastructure outside the runtime must obey the same terminal
                # contract. Only safe structured diagnostics cross this boundary.
                progress_emitter.emit_final_response(_failure_response(
                    request_id=request_id, error=exc, include_debug=request_data.include_debug,
                ).model_dump(mode="json"))
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
    ) -> AgentResponse:
        user_query = request_data.query
        session_id = request_data.session_id
        self._runtime_cleaner.run_once(force=False, current_session_id=session_id)
        session_metadata = build_session_metadata_snapshot(request_data)

        log_event(
            logger,
            logging.INFO,
            "agent_request",
            session_id=session_id[:8],
            request_id=request_id,
            **_query_log_fields(user_query),
        )

        started = time.monotonic()
        upload_manifest: UploadManifest | None = None
        try:
            with self._session_store.locked_session(session_id) as (entry, session_lock_wait_ms):
                agent_manager = entry.agent
                session = agent_manager._ensure_session()
                try:
                    # Check under the lock, before execution or reset. This is an
                    # attachment conflict, not a model or user-question failure.
                    try:
                        session.require_upload_context(request_data.uploads)
                    except ValueError:
                        return _failure_response(
                            request_id=request_id, upload_manifest=session.upload_manifest(),
                            problem=make_problem("upload_revision_conflict", "request"),
                        )
                    session.session_id = session_id
                    agent_manager.set_session_metadata(session_metadata)
                    agent_answer = agent_manager.run_agent_flow(user_query, progress_emitter=progress_emitter)
                    outcome = _build_response_payload(agent_answer, request_id=request_id)
                finally:
                    # Capture even failures while holding the lock; a concurrent
                    # upload mutation must not change this request's snapshot.
                    upload_manifest = session.upload_manifest()
        except Exception as exc:
            return _failure_response(request_id=request_id, upload_manifest=upload_manifest,
                                     error=exc, include_debug=request_data.include_debug)
        latency_ms_server = int((time.monotonic() - started) * 1000)

        try:
            debug_info = normalize_debug_info(
                raw_debug=agent_answer.get("debug"),
                latency_ms_server=latency_ms_server,
                answer_expected=outcome.response is not None,
            )
        except Exception as exc:
            # Observability failure must not replace a checked answer or discard
            # the manifest captured at its completion.
            log_event(logger, logging.ERROR, "debug_normalization_failure",
                      request_id=request_id, exception_type=type(exc).__name__)
            debug_info = AgentDebugInfo(
                schema_version=DEBUG_SCHEMA_VERSION, observability_status="failed", route_decisions=[],
                missing_required_debug_fields=["debug"], error_codes=["DEBUG_NORMALIZATION_FAILED"],
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
            outcome=outcome.status,
            problem_code=outcome.problem.code if outcome.problem else None,
        )

        return AgentResponse(
            **outcome.model_dump(mode="json"),
            trace=f"Session ID: {session_id}, Request ID: {request_id}, Agent ID: {id(agent_manager)}",
            upload_manifest=upload_manifest,
            debug=debug_info if request_data.include_debug else None,
        )


def _build_response_payload(agent_answer: dict[str, Any], *, request_id: str) -> TurnResult:
    return TurnResult.model_validate({
        **{key: value for key, value in agent_answer.items() if key in TurnResult.model_fields},
        "request_id": request_id,
    })


def _failure_response(
    *, request_id: str, upload_manifest: UploadManifest | None = None,
    problem: ExecutionProblem | None = None, error: Exception | None = None,
    include_debug: bool = False,
) -> AgentResponse:
    if isinstance(error, LLMCallError):
        problem = error.problem
    problem = problem or make_problem("internal_error", "request")
    log_event(
        logger, logging.WARNING if problem.code == "upload_revision_conflict" else logging.ERROR,
        "agent_request_failure", request_id=request_id, code=problem.code, stage=problem.stage,
        exception_type=type(error).__name__ if error else None,
        llm_diagnostic=error.diagnostic.model_dump(mode="json", exclude_none=True)
        if isinstance(error, LLMCallError) else None,
    )
    return AgentResponse(
        status="refused" if problem.code == "model_refusal" else "failed",
        request_id=request_id, response=None, problem=problem, message=problem.message,
        trace=f"Request ID: {request_id}", upload_manifest=upload_manifest,
        debug=AgentDebugInfo(
            schema_version=DEBUG_SCHEMA_VERSION, observability_status="failed", route_decisions=[],
            llm_diagnostics=[error.diagnostic] if isinstance(error, LLMCallError) else [],
        ) if include_debug else None,
    )
