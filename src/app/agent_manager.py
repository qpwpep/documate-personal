from __future__ import annotations

import logging
import time
from typing import Any

from src.runtime.agent_runtime import DebugCollector, ExecutionRunner, GraphInvocationError, ResponseAssembler, SessionContext
from src.core.answer_schema import build_empty_response_payload
from src.core.contracts import SessionMetadata
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from src.core.contracts.boundary.runtime import parse_session_metadata
from src.runtime.graph_builder import StageExecutionError, build_agent_graph
from src.core.latency import build_latency_breakdown, elapsed_ms, make_stage_latency_event
from src.infra.logging_utils import log_event
from src.runtime.progress import ProgressEmitter
from src.infra.settings import AppSettings, get_settings
from src.infra.tools.local_rag import build_temp_retriever


logger = logging.getLogger(__name__)


class AgentFlowManager:
    """Facade over session state, graph execution, debug collection, and response assembly."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.graph = build_agent_graph(self.settings)
        self._session = SessionContext()
        self._runner = ExecutionRunner(
            settings=self.settings,
            graph=self.graph,
            session=self._session,
            build_temp_retriever_fn=build_temp_retriever,
        )
        self._debug_collector = DebugCollector()
        self._response_assembler = ResponseAssembler()

    def _ensure_session(self) -> SessionContext:
        if not hasattr(self, "_session"):
            self._session = SessionContext()
        return self._session

    def _ensure_components(self) -> None:
        session = self._ensure_session()
        if not hasattr(self, "_debug_collector"):
            self._debug_collector = DebugCollector()
        if not hasattr(self, "_response_assembler"):
            self._response_assembler = ResponseAssembler()
        if not hasattr(self, "_runner"):
            self._runner = ExecutionRunner(
                settings=getattr(self, "settings", None),
                graph=self.graph,
                session=session,
                build_temp_retriever_fn=build_temp_retriever,
            )
        else:
            self._runner.graph = self.graph
            self._runner.session = session
            self._runner.settings = getattr(self, "settings", None)

    @property
    def messages(self) -> list[Any]:
        return self._ensure_session().messages

    @messages.setter
    def messages(self, value: list[Any]) -> None:
        self._ensure_session().messages = list(value or [])

    @property
    def session_metadata(self) -> SessionMetadata:
        return self._ensure_session().session_metadata

    @session_metadata.setter
    def session_metadata(self, value: SessionMetadata | dict[str, Any] | None) -> None:
        self._ensure_session().session_metadata = parse_session_metadata(value)

    @property
    def upload_retriever_handle(self):
        return self._ensure_session().upload_retriever_handle

    @upload_retriever_handle.setter
    def upload_retriever_handle(self, value) -> None:
        self._ensure_session().upload_retriever_handle = value

    @property
    def upload_file_path(self) -> str | None:
        return self._ensure_session().upload_file_path

    @upload_file_path.setter
    def upload_file_path(self, value: str | None) -> None:
        self._ensure_session().upload_file_path = value
        self._ensure_session().upload_file_paths = (value,) if value else ()

    @property
    def upload_file_paths(self) -> tuple[str, ...]:
        return self._ensure_session().upload_file_paths

    @upload_file_paths.setter
    def upload_file_paths(self, value: list[str] | tuple[str, ...] | None) -> None:
        paths = tuple(str(path) for path in (value or []) if str(path).strip())
        session = self._ensure_session()
        session.upload_file_paths = paths
        session.upload_file_path = paths[0] if paths else None

    def set_session_metadata(self, session_metadata: SessionMetadata | None) -> None:
        self._ensure_session().set_session_metadata(session_metadata)

    def close(self) -> None:
        self._ensure_session().close()

    @staticmethod
    def _extract_observed_evidence(current_turn_messages: list[Any], *, errors: list[str]) -> list[dict[str, Any]]:
        return DebugCollector._extract_observed_evidence(current_turn_messages, errors=errors)

    @staticmethod
    def _exit_payload(message: str) -> dict[str, Any]:
        return {
            "message": message,
            "filepath": "",
            "response": None,
            "response_payload": build_empty_response_payload(answer=message).model_dump(mode="json"),
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model_name": None,
                "models_used": [],
                "llm_calls": [],
                "errors": [],
                "validation_events": [],
                "edge_decisions": [],
                "planner_errors": [],
                "observed_evidence": [],
                "retry_context": None,
                "retrieval_diagnostics": [],
                "planner_diagnostics": None,
                "latency_breakdown": None,
            },
        }

    @staticmethod
    def _error_payload(
        *,
        message: str,
        graph_total_ms: int | None,
        flow_started: float,
        upload_retriever_build_ms: int | None,
        stage_error: StageExecutionError | None,
        error_code: str | None = None,
    ) -> dict[str, Any]:
        raw_trace: list[dict[str, Any]] = []
        if stage_error is not None:
            raw_trace.append(
                make_stage_latency_event(
                    stage=stage_error.stage,  # type: ignore[arg-type]
                    attempt=1,
                    latency_ms=stage_error.latency_ms,
                    status="error",
                )
            )
        latency_breakdown = build_latency_breakdown(
            raw_trace=raw_trace,
            graph_total_ms=graph_total_ms,
            server_total_ms=elapsed_ms(flow_started, time.perf_counter()),
            upload_retriever_build_ms=upload_retriever_build_ms,
        )
        return {
            "message": message,
            "filepath": "",
            "response": None,
            "response_payload": build_empty_response_payload(answer=message).model_dump(mode="json"),
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "observability_status": "failed",
                "missing_required_debug_fields": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "model_name": None,
                "models_used": [],
                "llm_calls": [],
                "errors": [message],
                "error_codes": [error_code] if error_code else [],
                "validation_events": [],
                "edge_decisions": [],
                "planner_errors": [],
                "observed_evidence": [],
                "retry_context": None,
                "retrieval_diagnostics": [],
                "planner_diagnostics": None,
                "latency_breakdown": latency_breakdown.model_dump(mode="json"),
            },
        }

    def run_agent_flow(
        self,
        user_input: str,
        upload_file_path: str | None = None,
        *,
        upload_file_paths: list[str] | None = None,
        planner_mode: str = "auto",
        eval_faults: dict[str, str] | None = None,
        progress_emitter: ProgressEmitter | None = None,
    ) -> dict[str, Any]:
        self._ensure_components()

        if user_input.lower() in {"exit", "종료", "quit", "q"}:
            self.close()
            return self._exit_payload("Chat session has been reset. Start again.")

        flow_started = time.perf_counter()
        upload_retriever_build_ms: int | None = None
        try:
            state, upload_retriever_build_ms = self._runner.prepare_graph_state(
                user_input,
                upload_file_path,
                upload_file_paths=upload_file_paths,
                planner_mode=planner_mode,
                eval_faults=eval_faults,
                progress_emitter=progress_emitter,
            )
            response, graph_total_ms = self._runner.invoke_graph(state)
            updated_messages = response["messages"]
            self.messages = updated_messages
            debug_info = self._debug_collector.build(
                response=response,
                updated_messages=updated_messages,
                graph_total_ms=graph_total_ms,
                upload_retriever_build_ms=upload_retriever_build_ms,
            )
            return self._response_assembler.assemble(
                response=response,
                updated_messages=updated_messages,
                debug_info=debug_info,
            )

        except Exception as exc:
            self._ensure_session().cleanup_upload_retriever()
            self.upload_file_path = None
            self.upload_file_paths = ()
            graph_total_ms = None
            stage_error = None
            root_exc = exc
            if isinstance(exc, GraphInvocationError):
                graph_total_ms = exc.graph_total_ms
                root_exc = exc.cause
            if isinstance(root_exc, StageExecutionError):
                stage_error = root_exc
                root_exc = root_exc.cause
            error_code = None
            if "UPLOAD_RETRIEVER_BUILD_FAILED" in str(root_exc):
                error_code = "UPLOAD_RETRIEVER_BUILD_FAILED"
            if progress_emitter is not None and stage_error is None:
                progress_emitter.emit_error(message=str(root_exc), stage=None)
            log_event(logger, logging.ERROR, "agent_execution_error", error=root_exc)
            return self._error_payload(
                message=str(root_exc),
                graph_total_ms=graph_total_ms,
                flow_started=flow_started,
                upload_retriever_build_ms=upload_retriever_build_ms,
                stage_error=stage_error,
                error_code=error_code,
            )
