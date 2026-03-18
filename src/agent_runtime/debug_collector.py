from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..contracts.debug import (
    coerce_llm_calls,
    coerce_planner_diagnostic,
    coerce_retrieval_diagnostics,
    coerce_retry_state,
)
from ..contracts.graph_state import debug_state, planner_state, response_state, retry_state
from ..evidence import dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from ..latency import build_latency_breakdown


class DebugCollector:
    @staticmethod
    def _extract_token_usage_from_llm_call(llm_call: dict[str, Any]) -> dict[str, int]:
        usage_metadata = llm_call.get("usage_metadata")
        response_metadata = llm_call.get("response_metadata")
        usage_candidates = []
        if isinstance(usage_metadata, dict):
            usage_candidates.append(usage_metadata)
        if isinstance(response_metadata, dict):
            token_usage = response_metadata.get("token_usage")
            if isinstance(token_usage, dict):
                usage_candidates.append(token_usage)

        for usage in usage_candidates:
            prompt_tokens = usage.get("prompt_tokens", usage.get("input_tokens", 0))
            completion_tokens = usage.get("completion_tokens", usage.get("output_tokens", 0))
            total_tokens = usage.get("total_tokens", 0)
            try:
                prompt_tokens = int(prompt_tokens or 0)
                completion_tokens = int(completion_tokens or 0)
                total_tokens = int(total_tokens or 0)
            except (TypeError, ValueError):
                continue
            if total_tokens <= 0:
                total_tokens = prompt_tokens + completion_tokens
            if prompt_tokens >= 0 and completion_tokens >= 0 and total_tokens >= 0:
                return {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }

        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    @staticmethod
    def _extract_model_name_from_llm_call(llm_call: dict[str, Any]) -> str | None:
        response_metadata = llm_call.get("response_metadata")
        if not isinstance(response_metadata, dict):
            return None
        model_name = response_metadata.get("model_name") or response_metadata.get("model")
        return str(model_name) if model_name else None

    @staticmethod
    def _build_fallback_llm_call_from_ai_message(message: AIMessage, *, attempt: int) -> dict[str, Any] | None:
        response_metadata = getattr(message, "response_metadata", None)
        usage_metadata = getattr(message, "usage_metadata", None)
        has_response_metadata = isinstance(response_metadata, dict) and bool(response_metadata)
        has_usage_metadata = isinstance(usage_metadata, dict) and bool(usage_metadata)
        if not has_response_metadata and not has_usage_metadata:
            return None
        return {
            "stage": "synthesis",
            "attempt": max(0, int(attempt)),
            "path": "direct",
            "response_metadata": dict(response_metadata) if has_response_metadata else {},
            "usage_metadata": dict(usage_metadata) if has_usage_metadata else {},
        }

    @classmethod
    def _summarize_llm_calls(
        cls,
        llm_calls: list[dict[str, Any]],
    ) -> tuple[dict[str, int], str | None, list[str]]:
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        models_used: list[str] = []
        final_synthesis_model: str | None = None

        for call in llm_calls:
            usage = cls._extract_token_usage_from_llm_call(call)
            total_prompt_tokens += usage["prompt_tokens"]
            total_completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]
            model_name = cls._extract_model_name_from_llm_call(call)
            if model_name and model_name not in models_used:
                models_used.append(model_name)
            if call.get("stage") == "synthesis" and model_name:
                final_synthesis_model = model_name

        if total_tokens <= 0:
            total_tokens = total_prompt_tokens + total_completion_tokens

        return (
            {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
            final_synthesis_model,
            models_used,
        )

    @staticmethod
    def _extract_tool_names_from_ai_message(message: AIMessage) -> list[str]:
        tool_names: list[str] = []
        tool_calls = getattr(message, "tool_calls", None)
        if not isinstance(tool_calls, list):
            return tool_names

        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                continue
            name = tool_call.get("name")
            if name:
                tool_names.append(str(name))
        return tool_names

    @staticmethod
    def _extract_observed_evidence(
        current_turn_messages: list[Any],
        *,
        errors: list[str],
    ) -> list[dict[str, Any]]:
        collected = []
        evidence_tools = {"tavily_search", "rag_search", "upload_search"}

        for message in current_turn_messages:
            if not isinstance(message, ToolMessage):
                continue

            tool_name = str(getattr(message, "name", "") or "").strip()
            if tool_name not in evidence_tools:
                continue

            parsed_items = parse_evidence_payload(
                getattr(message, "content", None),
                context=f"tool:{tool_name}",
                errors=errors,
            )
            collected.extend(parsed_items)

        return evidence_to_dicts(dedupe_evidence(collected))

    @staticmethod
    def _normalize_retry_context(raw_retry_context: Any) -> dict[str, Any] | None:
        retry = coerce_retry_state(raw_retry_context)
        payload = retry.model_dump(mode="json")
        payload.pop("preserved_evidence", None)
        payload.pop("preserved_retrieval_diagnostics", None)
        if not retry.needs_retry and retry.attempt <= 0 and retry.retry_reason is None and not retry.retrieval_feedback:
            if (
                retry.evidence_start_index == 0
                and retry.retrieval_error_start_index == 0
                and retry.retrieval_diagnostic_start_index == 0
                and retry.score_avg is None
                and not retry.failed_routes
            ):
                return None
        return payload

    @staticmethod
    def _normalize_retrieval_diagnostics(raw_diagnostics: Any) -> list[dict[str, Any]]:
        return [item.model_dump(mode="json") for item in coerce_retrieval_diagnostics(raw_diagnostics)]

    @staticmethod
    def _normalize_planner_diagnostics(raw_planner_diagnostics: Any) -> dict[str, Any] | None:
        diagnostics = coerce_planner_diagnostic(raw_planner_diagnostics)
        return diagnostics.model_dump(mode="json") if diagnostics is not None else None

    def build(
        self,
        *,
        response: dict[str, Any],
        updated_messages: list[Any],
        graph_total_ms: int,
        upload_retriever_build_ms: int | None,
    ) -> dict[str, Any]:
        tool_calls: list[str] = []
        state_debug = debug_state(response)
        state_response = response_state(response)
        state_retry = retry_state(response)
        state_planner = planner_state(response)
        debug_errors = [
            *state_debug.retrieval_errors,
            *state_debug.synthesis_errors,
            *state_debug.validation_errors,
            *state_debug.action_errors,
        ]
        llm_calls = [item.model_dump(mode="json") for item in state_debug.llm_calls]
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

        if not llm_calls:
            fallback_llm_calls = [
                item
                for item in (
                    self._build_fallback_llm_call_from_ai_message(
                        message,
                        attempt=int(state_response.synthesis_attempt or 0),
                    )
                    for message in current_turn_messages
                    if isinstance(message, AIMessage)
                )
                if item is not None
            ]
            if fallback_llm_calls:
                llm_calls = fallback_llm_calls

        token_usage, model_name, models_used = self._summarize_llm_calls(llm_calls)

        for message in current_turn_messages:
            if isinstance(message, AIMessage):
                tool_calls.extend(self._extract_tool_names_from_ai_message(message))
            elif isinstance(message, ToolMessage) and getattr(message, "name", ""):
                tool_calls.append(str(message.name))

        observed_evidence = self._extract_observed_evidence(
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
        latency_breakdown = build_latency_breakdown(
            raw_trace=[item for item in state_debug.latency_trace],
            graph_total_ms=graph_total_ms,
            upload_retriever_build_ms=upload_retriever_build_ms,
        )

        return {
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "token_usage": token_usage,
            "model_name": model_name,
            "models_used": models_used,
            "llm_calls": llm_calls,
            "errors": debug_errors,
            "planner_errors": planner_errors,
            "observed_evidence": observed_evidence,
            "retry_context": retry_context,
            "retrieval_diagnostics": retrieval_diagnostics,
            "planner_diagnostics": planner_diagnostics,
            "latency_breakdown": latency_breakdown.model_dump(mode="json"),
        }
