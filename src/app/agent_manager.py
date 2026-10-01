from __future__ import annotations

import logging
import time
from typing import Any

from src.core.conversation_memory import (
    ConversationMemoryPolicy,
    build_durable_conversation_memory,
    validate_query_text,
)
from src.runtime.agent_runtime import DebugCollector, ExecutionRunner, GraphInvocationError, ResponseAssembler, SessionContext
from src.runtime.agent_runtime.tool_execution import capture_tool_execution
from src.runtime.agent_runtime.llm_usage import capture_llm_usage
from src.core.answer_schema import AnswerResponse, finalize_answer, text_document, export_answer_text
from src.core.request_contracts import required_contract_turn_ids
from src.core.contracts import RuntimeState, SessionMetadata
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import validate_route_decisions
from src.core.contracts.boundary.runtime import parse_runtime_state, parse_session_metadata
from src.core.contracts.boundary.response import get_response_state
from src.runtime.graph_builder import StageExecutionError, build_agent_graph
from src.core.latency import build_latency_breakdown, elapsed_ms, make_stage_latency_event
from src.infra.logging_utils import log_event
from src.runtime.progress import ProgressEmitter
from src.infra.settings import AppSettings, get_settings


logger = logging.getLogger(__name__)


def is_session_reset_command(user_input: str) -> bool:
    """Identify explicit reset commands before graph execution."""
    return user_input.lower() in {"exit", "종료", "quit", "q"}


class AgentFlowManager:
    """Facade over session state, graph execution, debug collection, and response assembly."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.graph = build_agent_graph(self.settings)
        self._session = SessionContext()
        self._runner = ExecutionRunner(
            graph=self.graph,
            session=self._session,
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
                graph=self.graph,
                session=session,
            )
        else:
            self._runner.graph = self.graph
            self._runner.session = session

    @property
    def messages(self) -> list[Any]:
        return self._ensure_session().messages

    @messages.setter
    def messages(self, value: list[Any]) -> None:
        self._ensure_session().messages = list(value or [])

    @property
    def memory_summary(self) -> str | None:
        return self._ensure_session().memory_summary

    @memory_summary.setter
    def memory_summary(self, value: str | None) -> None:
        self._ensure_session().memory_summary = value

    @property
    def session_metadata(self) -> SessionMetadata:
        return self._ensure_session().session_metadata

    @session_metadata.setter
    def session_metadata(self, value: SessionMetadata | dict[str, Any] | None) -> None:
        self._ensure_session().session_metadata = parse_session_metadata(value)

    @property
    def upload_retriever_handle(self):
        return self._ensure_session().upload_retriever_handle

    def set_session_metadata(self, session_metadata: SessionMetadata | None) -> None:
        self._ensure_session().set_session_metadata(session_metadata)

    def close(self) -> None:
        self._ensure_session().close()

    def _conversation_memory_policy(self) -> ConversationMemoryPolicy:
        settings = getattr(self, "settings", None)
        if isinstance(settings, AppSettings):
            return settings.conversation_memory_policy()
        return ConversationMemoryPolicy()

    @staticmethod
    def _resolve_response_memory_summary(
        response: dict[str, Any],
        *,
        fallback: str | None,
    ) -> str | None:
        if "runtime" not in response:
            return fallback
        raw_runtime = response.get("runtime")
        if isinstance(raw_runtime, RuntimeState):
            return raw_runtime.memory_summary
        if isinstance(raw_runtime, dict) and "memory_summary" in raw_runtime:
            return parse_runtime_state(raw_runtime).memory_summary
        return fallback

    @staticmethod
    def _exit_payload(message: str) -> dict[str, Any]:
        result = finalize_answer(text_document(message), [])
        return {
            "response": result.model_dump(mode="json"),
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "llm_calls": [],
                "errors": [],
                "validation_events": [],
                "route_decisions": [],
                "memory_compactions": [],
                "planner_errors": [],
                "observed_hits": [],
                "answer_provenance": AnswerProvenance(
                    body_kind="acknowledge", response_hash=result.content_hash, evidence_packet=[],
                ).model_dump(mode="json"),
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
        stage_error: StageExecutionError | None,
        graph_state: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        debug = graph_state.get("debug") if graph_state is not None else None
        raw_trace = (
            debug.get("latency_trace", []) if isinstance(debug, dict)
            else getattr(debug, "latency_trace", [])
        )
        raw_trace = list(raw_trace) if isinstance(raw_trace, list) else []
        memory_compactions = (
            debug.get("memory_compactions", []) if isinstance(debug, dict)
            else getattr(debug, "memory_compactions", [])
        )
        route_decisions: list[dict[str, Any]] = []
        missing_debug_fields: list[str] = []
        errors = [message]
        error_codes: list[str] = []
        if graph_state is not None:
            try:
                route_decisions = [
                    decision.model_dump(mode="json")
                    for decision in validate_route_decisions(graph_state.get("route_decisions"))
                ]
            except ValueError as exc:
                missing_debug_fields.append("route_decisions")
                errors.append(f"Invalid committed route_decisions: {exc}")
                error_codes.append("DEBUG_NORMALIZATION_FAILED")
        if stage_error is not None:
            raw_trace.append(
                make_stage_latency_event(
                    stage=stage_error.stage,  # type: ignore[arg-type]
                    attempt=stage_error.attempt,
                    latency_ms=stage_error.latency_ms,
                    status="error",
                )
            )
        latency_breakdown = build_latency_breakdown(
            raw_trace=raw_trace,
            graph_total_ms=graph_total_ms,
            server_total_ms=elapsed_ms(flow_started, time.perf_counter()),
        )
        result = finalize_answer(text_document(message), [])
        return {
            "response": result.model_dump(mode="json"),
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "observability_status": "failed",
                "missing_required_debug_fields": missing_debug_fields,
                "tool_calls": [],
                "tool_call_count": 0,
                "llm_calls": [],
                "errors": errors,
                "error_codes": error_codes,
                "validation_events": [],
                "route_decisions": route_decisions,
                "memory_compactions": memory_compactions,
                "planner_errors": [],
                "observed_hits": [],
                "answer_provenance": AnswerProvenance(
                    body_kind="unresolved", response_hash=result.content_hash, evidence_packet=[],
                ).model_dump(mode="json"),
                "retry_context": None,
                "retrieval_diagnostics": [],
                "planner_diagnostics": None,
                "latency_breakdown": latency_breakdown.model_dump(mode="json"),
            },
        }

    def run_agent_flow(
        self,
        user_input: str,
        *,
        progress_emitter: ProgressEmitter | None = None,
    ) -> dict[str, Any]:
        with (capture_tool_execution(getattr(progress_emitter, "request_id", None)) as recorder,
              capture_llm_usage() as usage_recorder):
            result = self._run_agent_flow(user_input, progress_emitter=progress_emitter)
            evidence = recorder.snapshot()
            debug = result["debug"]
            debug["llm_calls"] = [call.model_dump(mode="json") for call in usage_recorder.snapshot()]
            debug["execution_evidence"] = evidence.model_dump(mode="json")
            debug["tool_calls"] = [event.tool_name for event in evidence.events if event.phase == "started"]
            debug["tool_call_count"] = len(debug["tool_calls"])
            return result

    def _run_agent_flow(
        self,
        user_input: str,
        *,
        progress_emitter: ProgressEmitter | None = None,
    ) -> dict[str, Any]:
        self._ensure_components()

        flow_started = time.perf_counter()
        try:
            validate_query_text(user_input)
        except ValueError as exc:
            return self._error_payload(
                message=str(exc),
                graph_total_ms=None,
                flow_started=flow_started,
                stage_error=None,
            )

        if is_session_reset_command(user_input):
            self.close()
            return self._exit_payload("Chat session has been reset. Start again.")

        response: dict[str, Any] | None = None
        graph_total_ms: int | None = None
        try:
            previous_memory = self._ensure_session().snapshot_conversation_memory()
            state = self._runner.prepare_graph_state(
                user_input,
                progress_emitter=progress_emitter,
            )
            response, graph_total_ms = self._runner.invoke_graph(state)
            updated_messages = list(response["messages"])
            candidate_summary = self._resolve_response_memory_summary(
                response,
                fallback=previous_memory.memory_summary,
            )
            debug_info = self._debug_collector.build(
                response=response,
                updated_messages=updated_messages,
                graph_total_ms=graph_total_ms,
            )
            assembled_response = self._response_assembler.assemble(
                response=response,
                debug_info=debug_info,
            )
            final_runtime = parse_runtime_state(response.get("runtime", state.get("runtime")))
            final_response_state = get_response_state(response)
            durable_memory = build_durable_conversation_memory(
                updated_messages,
                memory_summary=candidate_summary,
                policy=self._conversation_memory_policy(),
                canonical_assistant_text=export_answer_text(AnswerResponse.model_validate(assembled_response["response"])),
            )
            self._ensure_session().commit_conversation_memory(
                messages=durable_memory.messages,
                memory_summary=durable_memory.memory_summary,
                user_turns=final_runtime.user_turns,
                preserve_turn_ids=required_contract_turn_ids(final_runtime.pending_action.contract)
                if final_runtime.pending_action is not None else (),
            )
            # Only completed answers become the referent of "the previous answer".
            # A destination question or failure must not replace a pending document.
            self._ensure_session().commit_response_state(
                response=AnswerResponse.model_validate(assembled_response["response"]),
                response_kind=final_response_state.kind,
                pending_action=final_runtime.pending_action,
            )
            return assembled_response

        except Exception as exc:
            # Committed attachments outlive an individual answer/LLM failure.
            stage_error = None
            root_exc = exc
            if isinstance(exc, GraphInvocationError):
                graph_total_ms = exc.graph_total_ms
                root_exc = exc.cause
                response = exc.last_state
            if isinstance(root_exc, StageExecutionError):
                stage_error = root_exc
                root_exc = root_exc.cause
            if progress_emitter is not None and stage_error is None:
                progress_emitter.emit_error(message=str(root_exc), stage=None)
            log_event(logger, logging.ERROR, "agent_execution_error", error=root_exc)
            return self._error_payload(
                message=str(root_exc),
                graph_total_ms=graph_total_ms,
                flow_started=flow_started,
                stage_error=stage_error,
                graph_state=response,
            )
