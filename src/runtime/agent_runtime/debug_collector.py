from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, ToolMessage

from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.routing import validate_route_decisions
from src.core.contracts.boundary.debug import get_debug_state, parse_retry_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_diagnostic
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import parse_retrieval_diagnostics
from src.core.evidence import dedupe_search_hits, parse_search_hits
from src.core.latency import build_latency_breakdown
from src.runtime.agent_runtime.tool_execution import current_execution_evidence
from src.runtime.agent_runtime.llm_usage import current_llm_calls


class DebugCollector:
    @staticmethod
    def _parse_tool_payload(message: ToolMessage) -> dict[str, Any]:
        content = getattr(message, "content", None)
        if isinstance(content, str):
            raw_text = content
        elif isinstance(content, list):
            raw_text = "\n".join(str(item) for item in content)
        else:
            raw_text = str(content or "")
        try:
            parsed = json.loads(raw_text)
        except (TypeError, ValueError, json.JSONDecodeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @classmethod
    def _extract_action_results(cls, current_turn_messages: list[Any]) -> dict[str, Any] | None:
        action_results: dict[str, Any] = {}
        for message in current_turn_messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_name = str(getattr(message, "name", "") or "").strip()
            payload = cls._parse_tool_payload(message)
            if tool_name == "save_text":
                try:
                    saved_bytes = max(0, int(payload.get("bytes", 0) or 0))
                except (TypeError, ValueError):
                    saved_bytes = 0
                action_results["save_text"] = {
                    "status": str(payload.get("status") or "").strip(),
                    "file_path": str(payload.get("file_path") or "").strip() or None,
                    "bytes": saved_bytes,
                    "error": str(payload.get("error") or "").strip() or None,
                    "message": str(payload.get("message") or "").strip() or None,
                    "error_code": str(payload.get("error_code") or "").strip().upper() or None,
                }
        return action_results or None

    @staticmethod
    def _extract_observed_hits(
        current_turn_messages: list[Any],
        *,
        errors: list[str],
    ) -> list[dict[str, Any]]:
        collected = []
        # Only this turn's retrieval tool results count as observed sources.
        evidence_tools = {"tavily_search", "upload_search"}

        for message in current_turn_messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_name = str(getattr(message, "name", "") or "").strip()
            if tool_name not in evidence_tools:
                continue

            parsed_items = parse_search_hits(
                getattr(message, "content", None),
                errors=errors,
            )
            collected.extend(parsed_items)

        return [hit.model_dump(mode="json") for hit in dedupe_search_hits(collected)]

    @staticmethod
    def _normalize_retry_context(raw_retry_context: Any) -> dict[str, Any] | None:
        retry = parse_retry_state(raw_retry_context)
        payload = retry.model_dump(mode="json")
        payload.pop("preserved_hits", None)
        payload.pop("preserved_retrieval_diagnostics", None)
        if not retry.needs_retry and retry.attempt <= 0 and retry.retry_reason is None and not retry.retrieval_feedback:
            if (
                retry.hit_start_index == 0
                and retry.retrieval_error_start_index == 0
                and retry.retrieval_diagnostic_start_index == 0
                and retry.score_avg is None
                and not retry.failed_routes
            ):
                return None
        return payload

    @staticmethod
    def _normalize_retrieval_diagnostics(raw_diagnostics: Any) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in parse_retrieval_diagnostics(raw_diagnostics)]

    @staticmethod
    def _normalize_planner_diagnostics(raw_planner_diagnostics: Any) -> dict[str, Any] | None:
        diagnostics = parse_planner_diagnostic(raw_planner_diagnostics)
        return diagnostics.model_dump(mode="json") if diagnostics is not None else None

    @staticmethod
    def _collect_error_codes(
        *,
        state_error_codes: list[str],
        retrieval_diagnostics: list[dict[str, Any]],
        action_results: dict[str, Any] | None,
        planner_errors: list[str],
        debug_errors: list[str],
    ) -> list[str]:
        codes: list[str] = []

        def add(code: Any) -> None:
            normalized = str(code or "").strip().upper()
            if normalized and normalized not in codes:
                codes.append(normalized)

        for code in state_error_codes:
            add(code)
        for diagnostic in retrieval_diagnostics:
            add(diagnostic.get("error_code"))
        for result in (action_results or {}).values():
            if isinstance(result, dict):
                add(result.get("error_code"))
        for error in planner_errors:
            lowered = str(error or "").lower()
            if "output validation failed" in lowered or "schema" in lowered:
                add("PLANNER_SCHEMA_INVALID")
            if "timeout" in lowered or "timed out" in lowered:
                add("PLANNER_TIMEOUT")
        for error in debug_errors:
            lowered = str(error or "").lower()
            if "structured output was empty" in lowered:
                add("LLM_STRUCTURED_EMPTY")
            if "timed out" in lowered or "timeout" in lowered:
                add("SYNTHESIS_TIMEOUT")
            if "local_rag_failed" in lowered or "local similarity search failed" in lowered:
                add("LOCAL_RAG_FAILED")
            if "upload_retriever_build_failed" in lowered:
                add("UPLOAD_RETRIEVER_BUILD_FAILED")
        return codes

    def build(
        self,
        *,
        response: dict[str, Any],
        updated_messages: list[Any],
        graph_total_ms: int,
        upload_retriever_build_ms: int | None,
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
        current_turn_start_index = -1
        for index in range(len(updated_messages) - 1, -1, -1):
            if isinstance(updated_messages[index], HumanMessage):
                current_turn_start_index = index
                break

        current_turn_messages = (
            updated_messages[current_turn_start_index + 1 :]
            if current_turn_start_index >= 0
            else updated_messages
        )

        execution_evidence = current_execution_evidence()
        if execution_evidence is not None:
            tool_calls = [event.tool_name for event in execution_evidence.events if event.phase == "started"]

        observed_hits = self._extract_observed_hits(
            current_turn_messages,
            errors=debug_errors,
        )
        retry_context = self._normalize_retry_context(state_retry.model_dump(mode="json"))
        retrieval_diagnostics = self._normalize_retrieval_diagnostics(
            [item.model_dump(mode="json") for item in state_debug.retrieval_diagnostics]
        )
        planner_diagnostics = self._normalize_planner_diagnostics(
            state_planner.diagnostics.model_dump(mode="json")
        )
        action_results = self._extract_action_results(current_turn_messages) or {}
        for receipt in state_response.result.actions:
            if receipt.kind == "slack_notify" and receipt.slack is not None:
                action_results["slack_notify"] = receipt.slack.model_dump(mode="json")
        action_results = action_results or None
        error_codes = self._collect_error_codes(
            state_error_codes=list(state_debug.error_codes),
            retrieval_diagnostics=retrieval_diagnostics,
            action_results=action_results,
            planner_errors=planner_errors,
            debug_errors=debug_errors,
        )
        if missing_required_debug_fields and "DEBUG_NORMALIZATION_FAILED" not in error_codes:
            error_codes.append("DEBUG_NORMALIZATION_FAILED")
        latency_breakdown = build_latency_breakdown(
            raw_trace=[item for item in state_debug.latency_trace],
            graph_total_ms=graph_total_ms,
            upload_retriever_build_ms=upload_retriever_build_ms,
        )
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
