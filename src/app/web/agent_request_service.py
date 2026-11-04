from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import AsyncIterator
from queue import Empty, Queue
from threading import Thread
from typing import Any

from src.infra.logging_utils import log_event
from src.runtime.progress import ProgressEmitter
from src.app.web.agent_request_support import build_session_metadata_snapshot, build_debug_info
from src.app.web.cleanup import RuntimeCleaner
from src.app.transport_contracts import AgentResponse, AgentStreamEvent
from src.app.web.schemas import AgentRequest
from src.core.contracts.debug import DebugPayload
from src.core.llm_errors import ExecutionProblem, LLMCallError, make_problem
from src.core.uploads import UploadManifest
from src.app.web.session_store import InMemorySessionStore


logger = logging.getLogger(__name__)


def _query_log_fields(query: str) -> dict[str, Any]:
    encoded_query = query.encode("utf-8", errors="replace")
    return {
        "query_length": len(query),
        "query_utf8_bytes": len(encoded_query),
        "query_hash": hashlib.sha256(encoded_query).hexdigest(),
    }


class AgentRequestService:
    def __init__(
        self,
        *,
        runtime_cleaner: RuntimeCleaner,
        session_store: InMemorySessionStore,
        heartbeat_interval_seconds: float = 1.5,
    ) -> None:
        self._runtime_cleaner = runtime_cleaner
        self._session_store = session_store
        self._heartbeat_interval_seconds = heartbeat_interval_seconds

    def stream(
        self,
        *,
        execution_id: str,
        request_data: AgentRequest,
    ) -> AsyncIterator[AgentStreamEvent]:
        event_queue: Queue[AgentStreamEvent] = Queue()
        started = time.monotonic()

        def publish(event: str, data: dict[str, Any]) -> None:
            event_queue.put(AgentStreamEvent(event=event, data=dict(data)))

        progress_emitter = ProgressEmitter(
            publish=publish,
            execution_id=execution_id,
            session_id=request_data.session_id,
        )
        progress_emitter.emit_request_started()

        def worker() -> None:
            try:
                result = self._execute_request(
                    execution_id=execution_id,
                    request_data=request_data,
                    progress_emitter=progress_emitter,
                )
                progress_emitter.emit_final_response(result.model_dump(mode="json"))
            except Exception as exc:
                # Infrastructure outside the runtime must obey the same terminal
                # contract. Only safe structured diagnostics cross this boundary.
                progress_emitter.emit_final_response(_failure_response(
                    execution_id=execution_id, error=exc, include_debug=request_data.include_debug,
                ).model_dump(mode="json"))
            finally:
                progress_emitter.emit_done()

        Thread(
            target=worker,
            name=f"agent-stream-{execution_id}",
            daemon=True,
        ).start()

        async def event_stream() -> AsyncIterator[AgentStreamEvent]:
            while True:
                try:
                    item = await asyncio.to_thread(event_queue.get, timeout=self._heartbeat_interval_seconds)
                except Empty:
                    # Keep the connection alive even before the graph starts.
                    # This does not extend model deadlines or retry execution.
                    yield AgentStreamEvent(event="heartbeat", data={
                        "execution_id": execution_id,
                        "elapsed_ms": round((time.monotonic() - started) * 1000),
                    })
                    continue
                yield item
                if item.event == "done":
                    return

        return event_stream()

    def _execute_request(
        self,
        *,
        execution_id: str,
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
            execution_id=execution_id,
            **_query_log_fields(user_query),
        )

        started = time.monotonic()
        upload_manifest: UploadManifest | None = None
        try:
            with self._session_store.locked_session(session_id) as (entry, session_lock_wait_ms):
                agent_manager = entry.agent
                session = agent_manager.session
                try:
                    # Check under the lock, before execution or reset. This is an
                    # attachment conflict, not a model or user-question failure.
                    try:
                        session.require_upload_context(request_data.uploads)
                    except ValueError:
                        return _failure_response(
                            execution_id=execution_id, upload_manifest=session.upload_manifest(),
                            problem=make_problem("upload_revision_conflict", "request"),
                        )
                    session.session_id = session_id
                    agent_manager.set_session_metadata(session_metadata)
                    agent_answer = agent_manager.run_agent_flow(user_query, progress_emitter=progress_emitter)
                    outcome = agent_answer.outcome
                finally:
                    # Capture even failures while holding the lock; a concurrent
                    # upload mutation must not change this request's snapshot.
                    upload_manifest = session.upload_manifest()
        except Exception as exc:
            return _failure_response(execution_id=execution_id, upload_manifest=upload_manifest,
                                     error=exc, include_debug=request_data.include_debug)
        latency_ms_server = int((time.monotonic() - started) * 1000)

        debug_info = None
        if request_data.include_debug:
            try:
                debug_info = build_debug_info(
                    debug=agent_answer.debug,
                    latency_ms_server=latency_ms_server,
                    answer_expected=outcome.response is not None,
                )
            except Exception as exc:
                # Observability failure must not replace a checked answer or discard
                # the manifest captured at its completion.
                log_event(logger, logging.ERROR, "debug_normalization_failure",
                          execution_id=execution_id, exception_type=type(exc).__name__)
                debug_info = DebugPayload(
                    observability_status="failed", route_decisions=[],
                    missing_required_debug_fields=["debug"], error_codes=["DEBUG_NORMALIZATION_FAILED"],
                    latency_ms_server=latency_ms_server,
                )

        log_event(
            logger,
            logging.INFO,
            "agent_response",
            session_id=session_id[:8],
            execution_id=execution_id,
            agent_id=id(agent_manager),
            latency_ms_server=latency_ms_server,
            session_lock_wait_ms=session_lock_wait_ms,
            session_lock_contended=session_lock_wait_ms > 0,
            outcome=outcome.status,
            problem_code=outcome.problem.code if outcome.problem else None,
        )

        return AgentResponse(
            status=outcome.status, response=outcome.response, answer_id=outcome.answer_id,
            message=outcome.message, problem=outcome.problem, execution_id=execution_id,
            missing_slots=outcome.missing_slots,
            trace=f"Session ID: {session_id}, Execution ID: {execution_id}, Agent ID: {id(agent_manager)}",
            upload_manifest=upload_manifest,
            debug=debug_info,
        )


def _failure_response(
    *, execution_id: str, upload_manifest: UploadManifest | None = None,
    problem: ExecutionProblem | None = None, error: Exception | None = None,
    include_debug: bool = False,
) -> AgentResponse:
    if isinstance(error, LLMCallError):
        problem = error.problem
    problem = problem or make_problem("internal_error", "request")
    log_event(
        logger, logging.WARNING if problem.code == "upload_revision_conflict" else logging.ERROR,
        "agent_request_failure", execution_id=execution_id, code=problem.code, stage=problem.stage,
        exception_type=type(error).__name__ if error else None,
        llm_diagnostic=error.diagnostic.model_dump(mode="json", exclude_none=True)
        if isinstance(error, LLMCallError) else None,
    )
    return AgentResponse(
        status="refused" if problem.code == "model_refusal" else "failed",
        execution_id=execution_id, response=None, problem=problem, message=problem.message,
        trace=f"Execution ID: {execution_id}", upload_manifest=upload_manifest,
        debug=DebugPayload(
            observability_status="failed", route_decisions=[],
            llm_diagnostics=[error.diagnostic] if isinstance(error, LLMCallError) else [],
        ) if include_debug else None,
    )
