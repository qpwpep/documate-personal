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
from src.runtime.agent_runtime.validation_diagnostics import capture_validation_diagnostics
from src.core.answer_schema import finalize_answer, text_document
from src.core.answer_schema.rendering import render_answer_text
from src.core.request_contracts import required_contract_turn_ids
from src.core.contracts import GraphState, SessionMetadata
from src.core.contracts.outcome import TurnResult
from src.core.contracts.execution_result import AgentExecutionResult
from src.core.llm_errors import LLMCallError, ExecutionProblem, LLMDiagnostic, make_problem
from src.infra.llm_boundary import capture_llm_diagnostics, check_request_deadline, request_deadline
from src.core.contracts.debug import DebugPayload, ErrorCode
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import RoutingDecision, validate_route_decisions
from src.core.contracts.boundary.debug import get_debug_state
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
    def _early_result(outcome: TurnResult) -> AgentExecutionResult:
        result = outcome.response
        return AgentExecutionResult(
            outcome=outcome,
            debug=DebugPayload(
                llm_calls=[],
                answer_provenance=AnswerProvenance(
                    body_kind="acknowledge", response_hash=result.content_hash, evidence_packet=[],
                ) if result is not None else None,
            ),
        )

    @staticmethod
    def _error_payload(
        *,
        graph_total_ms: int | None,
        flow_started: float,
        stage_error: StageExecutionError | None,
        graph_state: GraphState | None = None,
        problem: ExecutionProblem | None = None,
        diagnostic: LLMDiagnostic | None = None,
    ) -> AgentExecutionResult:
        debug = get_debug_state(graph_state or {})
        raw_trace = list(debug.latency_trace)
        latency_errors: list[str] = []
        route_decisions: list[RoutingDecision] = []
        missing_debug_fields: list[str] = []
        problem = problem or make_problem("internal_error", stage_error.stage if stage_error else "request")
        errors = [problem.code]
        error_codes: list[ErrorCode] = [problem.code]
        if graph_state is not None:
            try:
                route_decisions = validate_route_decisions(graph_state.get("route_decisions"))
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
        return AgentExecutionResult(
            outcome=TurnResult(status="refused" if problem.code == "model_refusal" else "failed",
                               message=problem.message, problem=problem),
            debug=DebugPayload(
                observability_status="failed", missing_required_debug_fields=missing_debug_fields,
                llm_calls=[], llm_diagnostics=[diagnostic] if diagnostic else [],
                errors=errors, error_codes=error_codes, route_decisions=route_decisions,
                memory_compactions=list(debug.memory_compactions), latency_breakdown=latency_breakdown,
            ),
        )

    def run_agent_flow(
        self,
        user_input: str,
        *,
        progress_emitter: ProgressEmitter | None = None,
    ) -> AgentExecutionResult:
        with (capture_tool_execution(progress_emitter.execution_id if progress_emitter is not None else None) as recorder,
              capture_llm_usage() as usage_recorder, capture_llm_diagnostics() as diagnostics,
              capture_validation_diagnostics() as validation_recorder,
              request_deadline(self.settings.request_timeout_seconds)):
            result = self._run_agent_flow(user_input, progress_emitter=progress_emitter)
            evidence = recorder.snapshot()
            debug = result.debug
            debug.llm_calls = usage_recorder.snapshot()
            try:
                debug.validation_diagnostics = validation_recorder.snapshot()
            except (TypeError, ValueError) as exc:
                # Diagnostics cannot replace an already determined product outcome.
                debug.observability_status = "failed"
                if "validation_diagnostics" not in debug.missing_required_debug_fields:
                    debug.missing_required_debug_fields.append("validation_diagnostics")
                if "DEBUG_NORMALIZATION_FAILED" not in debug.error_codes:
                    debug.error_codes.append("DEBUG_NORMALIZATION_FAILED")
                log_event(logger, logging.ERROR, "validation_diagnostics_failure", exception_type=type(exc).__name__)
            debug.llm_diagnostics = list(diagnostics) or debug.llm_diagnostics
            result.outcome.execution_id = progress_emitter.execution_id if progress_emitter is not None else ""
            debug.execution_evidence = evidence
            debug.tool_calls = [event.tool_name for event in evidence.events if event.phase == "started"]
            debug.tool_call_count = len(debug.tool_calls)
            return result

    def _run_agent_flow(
        self,
        user_input: str,
        *,
        progress_emitter: ProgressEmitter | None = None,
    ) -> AgentExecutionResult:
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
            try:
                debug_info = self._debug_collector.build(response=response, graph_total_ms=graph_total_ms)
            except (TypeError, ValueError) as exc:
                # An observation failure must not discard an already checked answer.
                log_event(logger, logging.ERROR, "debug_normalization_failure", exception_type=type(exc).__name__)
                debug_info = DebugPayload(
                    observability_status="failed", missing_required_debug_fields=["debug"],
                    error_codes=["DEBUG_NORMALIZATION_FAILED"],
                )
            outcome = self._response_assembler.assemble(response=response)
            if outcome.response is None:
                debug_info.answer_provenance = None
            assembled_response = AgentExecutionResult(outcome=outcome, debug=debug_info)
            check_request_deadline("response_assembly")
            final_response_state = get_response_state(response)
            if outcome.status in {"failed", "refused"}:
                return assembled_response
            canonical_text = (outcome.message if outcome.status == "needs_input"
                              else render_answer_text(final_response_state.result))
            if outcome.status == "partial" and outcome.problem.code == "save_failed":
                canonical_text = "\n\n".join([canonical_text, outcome.message])
            elif outcome.status == "partial":
                # Retain what the user actually saw, including its limits. The
                # answer reference below remains the original checked document;
                # recording it is not a promotion of the failed request.
                limitations = list(dict.fromkeys([
                    outcome.message,
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
                preserve_turn_ids=required_contract_turn_ids(final_runtime.pending_request.contract)
                if final_runtime.pending_request is not None else (),
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
                displayed_partial=outcome.status == "partial" and not answer_reuse_failed,
                pending_request=final_runtime.pending_request,
                export_operations=final_runtime.answer_export_operations,
            )
            if (not answer_reuse_failed and outcome.response is not None
                    and final_response_state.result.content.blocks):
                outcome.answer_id = self.session.register_answer(final_response_state.result)
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
                return self._retain_timed_out_delivery(result, response, problem)
            return result

    def _retain_timed_out_delivery(
        self, result: AgentExecutionResult, state: GraphState, problem: ExecutionProblem,
    ) -> AgentExecutionResult:
        response = get_response_state(state)
        if not any(receipt.invocation_id and receipt.status != "skipped" for receipt in response.result.actions):
            return result
        runtime = get_runtime_state(state)
        # A completed or uncertain external effect must remain visible and must
        # not become a fresh delivery on a follow-up. This is delivery bookkeeping,
        # not a successful conversation/answer commit for the timed-out request.
        self.session.commit_delivery_state(
            pending_request=runtime.pending_request, export_operations=runtime.answer_export_operations,
        )
        result.debug.answer_provenance = AnswerProvenance(
            request_id=response.request_id, contract_revision=response.contract_revision,
            save_operation_binding_sha256=response.save_operation_binding_sha256,
            body_kind=response.body_kind, response_hash=response.result.content_hash,
            source=response.evidence_source, evidence_packet=response.evidence_packet,
        ) if response.body_kind is not None else None
        return AgentExecutionResult(
            outcome=TurnResult(status="partial", response=response.result, problem=problem, message=problem.message),
            debug=result.debug,
        )
