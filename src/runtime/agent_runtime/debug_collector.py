from __future__ import annotations

from typing import Any, get_args

from src.core.answer_schema import ActionReceipt
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION, ErrorCode, RetryState
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import validate_route_decisions
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.evidence import dedupe_search_hits
from src.core.latency import build_latency_breakdown
from src.runtime.agent_runtime.tool_execution import current_execution_evidence
from src.runtime.agent_runtime.llm_usage import current_llm_calls


class DebugCollector:
    @staticmethod
    def _extract_action_results(receipts: list[ActionReceipt]) -> dict[str, Any] | None:
        action_results: dict[str, Any] = {}
        for receipt in receipts:
            if receipt.kind == "save_text":
                action_results["save_text"] = {
                    "status": receipt.status,
                    "file_path": receipt.file_path,
                    "bytes": receipt.artifact.byte_count if receipt.artifact is not None else 0,
                    "error": receipt.error,
                    "message": receipt.message,
                    "error_code": receipt.error_code,
                }
            elif receipt.slack is not None:
                action_results["slack_notify"] = receipt.slack.model_dump(mode="json")
        return action_results or None

    @staticmethod
    def _serialize_retry_context(retry: RetryState) -> dict[str, Any] | None:
        if not retry.needs_retry and retry.attempt <= 0 and retry.retry_reason is None and not retry.retrieval_feedback:
            if (
                retry.hit_start_index == 0
                and retry.retrieval_error_start_index == 0
                and retry.retrieval_diagnostic_start_index == 0
                and retry.score_avg is None
                and not retry.failed_routes
                and retry.content_repair_attempt == 0
                and not retry.terminal_failed_requirement_ids
                and retry.terminal_retrieval_reason is None
            ):
                return None
        return retry.model_dump(mode="json", exclude={"preserved_hits", "preserved_retrieval_diagnostics"})

    @staticmethod
    def _collect_error_codes(
        *,
        state_error_codes: list[str],
        retrieval_diagnostics: list[dict[str, Any]],
        action_results: dict[str, Any] | None,
        debug_errors: list[str],
    ) -> list[str]:
        codes: list[str] = []
        allowed = get_args(ErrorCode)

        def add(code: str | None) -> None:
            if code is not None and code in allowed and code not in codes:
                codes.append(code)

        for code in state_error_codes:
            add(code)
        for diagnostic in retrieval_diagnostics:
            add(diagnostic.get("error_code"))
        for result in (action_results or {}).values():
            add(result.get("error_code"))
        for error in debug_errors:
            lowered = error.lower()
            if "local_rag_failed" in lowered or "local similarity search failed" in lowered:
                add("LOCAL_RAG_FAILED")
        return codes

    def build(
        self,
        *,
        response: dict[str, Any],
        graph_total_ms: int,
    ) -> dict[str, Any]:
        tool_calls: list[str] = []
        state_debug = get_debug_state(response)
        state_response = get_response_state(response)
        state_retry = get_retry_state(response)
        state_planner = get_planner_state(response)
        debug_errors = [
            *state_debug.retrieval_errors,
            *state_debug.synthesis_errors,
            *state_debug.action_errors,
        ]
        missing_required_debug_fields = []
        try:
            route_decisions = validate_route_decisions(response.get("route_decisions"))
        except ValueError as exc:
            route_decisions = []
            missing_required_debug_fields.append("route_decisions")
            debug_errors.append(f"route_decisions invalid: {exc}")
        calls = current_llm_calls()
        llm_calls = [item.model_dump(mode="json") for item in calls] if calls is not None else None
        planner_errors = list(state_debug.planner_errors)

        execution_evidence = current_execution_evidence()
        if execution_evidence is not None:
            tool_calls = [event.tool_name for event in execution_evidence.events if event.phase == "started"]

        observed_hits = [
            hit.model_dump(mode="json") for hit in dedupe_search_hits(get_retrieval_state(response).hit_log)
        ]
        retry_context = self._serialize_retry_context(state_retry)
        retrieval_diagnostics = [item.model_dump(mode="json") for item in state_debug.retrieval_diagnostics]
        planner_diagnostics = state_planner.diagnostics.model_dump(mode="json")
        action_results = self._extract_action_results(state_response.result.actions)
        latency_errors: list[str] = []
        latency_breakdown = build_latency_breakdown(
            raw_trace=state_debug.latency_trace,
            errors=latency_errors,
            graph_total_ms=graph_total_ms,
        )
        if latency_errors:
            debug_errors.extend(latency_errors)
            missing_required_debug_fields.append("latency_breakdown")
        error_codes = self._collect_error_codes(
            state_error_codes=list(state_debug.error_codes),
            retrieval_diagnostics=retrieval_diagnostics,
            action_results=action_results,
            debug_errors=debug_errors,
        )
        if state_response.problem is not None and state_response.problem.code not in error_codes:
            error_codes.append(state_response.problem.code)
        if missing_required_debug_fields and "DEBUG_NORMALIZATION_FAILED" not in error_codes:
            error_codes.append("DEBUG_NORMALIZATION_FAILED")
        answer_provenance = None
        if state_response.body_kind is not None:
            answer_provenance = AnswerProvenance(
                request_id=state_response.request_id,
                contract_revision=state_response.contract_revision or None,
                save_operation_binding_sha256=state_response.save_operation_binding_sha256,
                body_kind=state_response.body_kind,
                response_hash=state_response.result.content_hash,
                source=state_response.evidence_source,
                evidence_packet=state_response.evidence_packet,
            ).model_dump(mode="json")

        return {
            "schema_version": DEBUG_SCHEMA_VERSION,
            "observability_status": "failed" if missing_required_debug_fields else state_debug.observability_status,
            "missing_required_debug_fields": missing_required_debug_fields,
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "execution_evidence": execution_evidence.model_dump(mode="json") if execution_evidence is not None else None,
            "llm_calls": llm_calls,
            "llm_diagnostics": [item.model_dump(mode="json") for item in state_debug.llm_diagnostics],
            "errors": debug_errors,
            "error_codes": error_codes,
            "validation_events": list(state_debug.validation_events or []),
            "route_decisions": [item.model_dump(mode="json") for item in route_decisions],
            "memory_compactions": list(state_debug.memory_compactions),
            "planner_errors": planner_errors,
            "observed_hits": observed_hits,
            "answer_provenance": answer_provenance,
            "retry_context": retry_context,
            "retrieval_diagnostics": retrieval_diagnostics,
            "planner_diagnostics": planner_diagnostics,
            "latency_breakdown": latency_breakdown.model_dump(mode="json"),
            "action_results": action_results,
        }
