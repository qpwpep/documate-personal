from __future__ import annotations

import logging
import time
from typing import Any

from src.core.conversation_memory import (
    build_durable_conversation_memory,
)
from src.runtime.agent_runtime import AgentGraph, DebugCollector, ExecutionRunner, GraphInvocationError, ResponseAssembler, SessionContext
from src.runtime.agent_runtime.tool_execution import capture_tool_execution
from src.runtime.agent_runtime.llm_usage import capture_llm_usage
from src.core.answer_schema import finalize_answer, text_document
from src.core.answer_schema.rendering import render_answer_text
from src.core.request_contracts import required_contract_turn_ids
from src.core.contracts import GraphState, SessionMetadata
from src.core.contracts.outcome import TurnResult
from src.core.llm_errors import LLMCallError, ExecutionProblem, LLMDiagnostic, make_problem
from src.infra.llm_boundary import capture_llm_diagnostics, check_request_deadline, request_deadline
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import validate_route_decisions
from src.core.contracts.boundary.runtime import get_runtime_state
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

    def __init__(self, settings: AppSettings | None = None, *, graph: AgentGraph | None = None) -> None:
        self.settings = settings or get_settings()
        self._session = SessionContext()
        self._runner = ExecutionRunner(
            graph=graph if graph is not None else build_agent_graph(self.settings),
            session=self._session,
        )
        self._debug_collector = DebugCollector()
        self._response_assembler = ResponseAssembler()

    @property
    def session(self) -> SessionContext:
        return self._session

    @property
    def messages(self) -> list[Any]:
        return self.session.messages

    @property
    def memory_summary(self) -> str | None:
        return self.session.memory_summary

    @memory_summary.setter
    def memory_summary(self, value: str | None) -> None:
        self.session.memory_summary = value

    def set_session_metadata(self, session_metadata: SessionMetadata | None) -> None:
        self.session.set_session_metadata(session_metadata)

    def close(self) -> None:
        self.session.close()

    @staticmethod
    def _early_result(outcome: TurnResult) -> dict[str, Any]:
        result = outcome.response
        return {
            **outcome.model_dump(mode="json"),
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "observability_status": "ok",
                "missing_required_debug_fields": [],
                "tool_calls": [],
                "tool_call_count": 0,
                "llm_calls": [],
                "errors": [],
                "error_codes": [],
                "validation_events": [],
                "route_decisions": [],
                "memory_compactions": [],
                "planner_errors": [],
                "observed_hits": [],
                "answer_provenance": AnswerProvenance(
                    body_kind="acknowledge", response_hash=result.content_hash, evidence_packet=[],
                ).model_dump(mode="json") if result is not None else None,
                "retry_context": None,
                "retrieval_diagnostics": [],
                "planner_diagnostics": None,
                "latency_breakdown": None,
            },
        }

    @staticmethod
    def _error_payload(
        *,
        graph_total_ms: int | None,
        flow_started: float,
        stage_error: StageExecutionError | None,
        graph_state: GraphState | None = None,
        problem: ExecutionProblem | None = None,
        diagnostic: LLMDiagnostic | None = None,
    ) -> dict[str, Any]:
        debug = graph_state.get("debug") if graph_state is not None else None
        raw_trace = (
            debug.get("latency_trace", []) if isinstance(debug, dict)
            else getattr(debug, "latency_trace", [])
        )
        latency_errors: list[str] = []
        if isinstance(raw_trace, list):
            raw_trace = list(raw_trace)
        elif stage_error is not None:
            latency_errors.append("latency_trace must be a list")
            raw_trace = []
        memory_compactions = (
            debug.get("memory_compactions", []) if isinstance(debug, dict)
            else getattr(debug, "memory_compactions", [])
        )
        route_decisions: list[dict[str, Any]] = []
        missing_debug_fields: list[str] = []
        problem = problem or make_problem("internal_error", stage_error.stage if stage_error else "request")
        errors = [problem.code]
        error_codes: list[str] = [problem.code]
        if graph_state is not None:
            try:
                route_decisions = [
                    decision.model_dump(mode="json")
                    for decision in validate_route_decisions(graph_state.get("route_decisions"))
                ]
            except ValueError as exc:
                missing_debug_fields.append("route_decisions")
                errors.append("Invalid committed route_decisions")
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
            errors=latency_errors,
            graph_total_ms=graph_total_ms,
            server_total_ms=elapsed_ms(flow_started, time.perf_counter()),
        )
        if latency_errors:
            errors.extend(latency_errors)
            missing_debug_fields.append("latency_breakdown")
            if "DEBUG_NORMALIZATION_FAILED" not in error_codes:
                error_codes.append("DEBUG_NORMALIZATION_FAILED")
        return {
            **TurnResult(status="refused" if problem.code == "model_refusal" else "failed",
                         message=problem.message, problem=problem).model_dump(mode="json"),
            "debug": {
                "schema_version": DEBUG_SCHEMA_VERSION,
                "observability_status": "failed",
                "missing_required_debug_fields": missing_debug_fields,
                "tool_calls": [],
                "tool_call_count": 0,
                "llm_calls": [],
                "llm_diagnostics": [diagnostic.model_dump(mode="json")] if diagnostic else [],
                "errors": errors,
                "error_codes": error_codes,
                "validation_events": [],
                "route_decisions": route_decisions,
                "memory_compactions": memory_compactions,
                "planner_errors": [],
                "observed_hits": [],
                "answer_provenance": None,
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
        with (capture_tool_execution(progress_emitter.request_id if progress_emitter is not None else None) as recorder,
              capture_llm_usage() as usage_recorder, capture_llm_diagnostics() as diagnostics,
              request_deadline(self.settings.request_timeout_seconds)):
            result = self._run_agent_flow(user_input, progress_emitter=progress_emitter)
            evidence = recorder.snapshot()
            debug = result["debug"]
            debug["llm_calls"] = [call.model_dump(mode="json") for call in usage_recorder.snapshot()]
            debug["llm_diagnostics"] = [entry.model_dump(mode="json") for entry in diagnostics] or debug.get("llm_diagnostics", [])
            result["request_id"] = progress_emitter.request_id if progress_emitter is not None else ""
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
        flow_started = time.perf_counter()
        if is_session_reset_command(user_input):
            self.close()
            return self._early_result(TurnResult(
                response=finalize_answer(text_document("Chat session has been reset. Start again."), []),
            ))

        response: GraphState | None = None
        graph_total_ms: int | None = None
        try:
            check_request_deadline("request")
            state = self._runner.prepare_graph_state(
                user_input,
                progress_emitter=progress_emitter,
            )
            response, graph_total_ms = self._runner.invoke_graph(state)
            updated_messages = list(response["messages"])
            final_runtime = get_runtime_state(response)
            candidate_summary = final_runtime.memory_summary
            debug_info = self._debug_collector.build(
                response=response,
                graph_total_ms=graph_total_ms,
            )
            assembled_response = self._response_assembler.assemble(
                response=response,
                debug_info=debug_info,
            )
            check_request_deadline("response_assembly")
            final_response_state = get_response_state(response)
            if assembled_response["status"] in {"failed", "refused"}:
                return assembled_response
            canonical_text = (assembled_response["message"] if assembled_response["status"] == "needs_input"
                              else render_answer_text(final_response_state.result))
            if (assembled_response["status"] == "partial"
                    and assembled_response["problem"]["code"] == "save_failed"):
                canonical_text = "\n\n".join([canonical_text, assembled_response["message"]])
            elif assembled_response["status"] == "partial":
                # Retain what the user actually saw, including its limits. The
                # answer reference below remains the original checked document;
                # recording it is not a promotion of the failed request.
                limitations = list(dict.fromkeys([
                    assembled_response["message"],
                    *(issue.message for issue in final_response_state.result.issues),
                ]))
                canonical_text = "\n\n".join([
                    canonical_text, "부분 답변이며 요청 전체는 완료되지 않았습니다.",
                    *(message for message in limitations if message.strip()),
                ])
            durable_memory = build_durable_conversation_memory(
                updated_messages,
                memory_summary=candidate_summary,
                policy=self.settings.conversation_memory_policy(),
                canonical_assistant_text=canonical_text,
            )
            check_request_deadline("memory_commit")
            self.session.commit_conversation_memory(
                messages=durable_memory.messages,
                memory_summary=durable_memory.memory_summary,
                user_turns=final_runtime.user_turns,
                preserve_turn_ids=required_contract_turn_ids(final_runtime.pending_action.contract)
                if final_runtime.pending_action is not None else (),
            )
            # A failed copy/transform cannot replace the selected source answer
            # with its fallback, regardless of how the result was to be delivered.
            # Other displayed partial answers remain reusable source answers.
            contract = final_runtime.request_contract
            answer_reuse_failed = bool(
                contract is not None and contract.body.kind in {"copy_answer", "transform_answer"}
                and final_response_state.kind == "failure"
            )
            self.session.commit_response_state(
                response=final_response_state.result,
                response_kind=final_response_state.kind,
                displayed_partial=assembled_response["status"] == "partial" and not answer_reuse_failed,
                pending_action=final_runtime.pending_action,
            )
            self.session.answer_export_operations.update(final_runtime.answer_export_operations)
            if (not answer_reuse_failed and assembled_response["response"] is not None
                    and final_response_state.result.content.blocks):
                assembled_response["answer_id"] = self.session.register_answer(final_response_state.result)
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
            problem = root_exc.problem if isinstance(root_exc, LLMCallError) else make_problem(
                "internal_error", stage_error.stage if stage_error else "request")
            diagnostic = root_exc.diagnostic if isinstance(root_exc, LLMCallError) else None
            log_event(logger, logging.ERROR, "agent_execution_error", code=problem.code,
                      stage=problem.stage, exception_type=type(root_exc).__name__)
            result = self._error_payload(
                problem=problem,
                diagnostic=diagnostic,
                graph_total_ms=graph_total_ms,
                flow_started=flow_started,
                stage_error=stage_error,
                graph_state=response,
            )
            if problem.code == "request_timeout" and response is not None:
                self._retain_timed_out_delivery(result, response, problem)
            return result

    def _retain_timed_out_delivery(
        self, result: dict[str, Any], state: GraphState, problem: ExecutionProblem,
    ) -> None:
        response = get_response_state(state)
        if not any(receipt.invocation_id and receipt.status != "skipped" for receipt in response.result.actions):
            return
        runtime = get_runtime_state(state)
        # A completed or uncertain external effect must remain visible and must
        # not become a fresh delivery on a follow-up. This is delivery bookkeeping,
        # not a successful conversation/answer commit for the timed-out request.
        self.session.pending_action = (
            runtime.pending_action.model_copy(deep=True) if runtime.pending_action is not None else None
        )
        self.session.answer_export_operations.update(runtime.answer_export_operations)
        result.update(TurnResult(
            status="partial", response=response.result, problem=problem, message=problem.message,
        ).model_dump(mode="json"))
        result["debug"]["answer_provenance"] = AnswerProvenance(
            request_id=response.request_id, contract_revision=response.contract_revision,
            save_operation_binding_sha256=response.save_operation_binding_sha256,
            body_kind=response.body_kind, response_hash=response.result.content_hash,
            source=response.evidence_source, evidence_packet=response.evidence_packet,
        ).model_dump(mode="json") if response.body_kind is not None else None
