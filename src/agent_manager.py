import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .answer_schema import build_empty_response_payload
from .evidence import dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from .graph_builder import build_agent_graph
from .latency import build_latency_breakdown, elapsed_ms
from .logging_utils import log_event
from .nodes.state import SessionMetadata, coerce_session_metadata
from .settings import AppSettings, get_settings
from .tools.local_rag import UploadedRetrieverHandle, build_temp_retriever


logger = logging.getLogger(__name__)


class AgentFlowManager:
    """Manages per-session LangGraph execution and message state."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.graph = build_agent_graph(self.settings)
        self.messages: List[Any] = []
        self.session_metadata: SessionMetadata = coerce_session_metadata(None)
        self.upload_retriever_handle: UploadedRetrieverHandle | None = None
        self.upload_file_path: Optional[str] = None

    def set_session_metadata(self, session_metadata: SessionMetadata | None) -> None:
        self.session_metadata = coerce_session_metadata(session_metadata)

    def _snapshot_session_metadata(self) -> SessionMetadata:
        return coerce_session_metadata(getattr(self, "session_metadata", None))

    def _cleanup_upload_retriever(self) -> None:
        handle = getattr(self, "upload_retriever_handle", None)
        if handle is None:
            return

        try:
            handle.cleanup()
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "upload_retriever_cleanup_failed",
                collection=handle.collection_name,
                error=exc,
            )
        finally:
            self.upload_retriever_handle = None

    def close(self) -> None:
        self._cleanup_upload_retriever()
        self.upload_file_path = None
        self.messages = []
        self.session_metadata = coerce_session_metadata(None)

    @staticmethod
    def _extract_token_usage_from_ai_message(message: AIMessage) -> Dict[str, int]:
        usage_candidates = []
        usage_metadata = getattr(message, "usage_metadata", None)
        if isinstance(usage_metadata, dict):
            usage_candidates.append(usage_metadata)

        response_metadata = getattr(message, "response_metadata", None)
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

        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }

    @staticmethod
    def _extract_model_name_from_ai_message(message: AIMessage) -> str | None:
        response_metadata = getattr(message, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            return None
        model_name = response_metadata.get("model_name") or response_metadata.get("model")
        return str(model_name) if model_name else None

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
    def _extract_text_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        parts.append(str(text))
            return "\n".join(parts)
        return str(content)

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
    def _normalize_retry_context(raw_retry_context: Any) -> Dict[str, Any] | None:
        if not isinstance(raw_retry_context, dict):
            return None

        normalized: Dict[str, Any] = {}

        attempt = raw_retry_context.get("attempt")
        if isinstance(attempt, int) and attempt >= 0:
            normalized["attempt"] = attempt

        max_retries = raw_retry_context.get("max_retries")
        if isinstance(max_retries, int) and max_retries >= 0:
            normalized["max_retries"] = max_retries

        retry_reason = raw_retry_context.get("retry_reason")
        if retry_reason in {
            "no_evidence",
            "low_score",
            "tool_error",
            "blocked_missing_upload",
            "unsupported_claims",
        }:
            normalized["retry_reason"] = retry_reason

        retrieval_feedback = raw_retry_context.get("retrieval_feedback")
        if retrieval_feedback is not None:
            normalized["retrieval_feedback"] = str(retrieval_feedback)

        evidence_start_index = raw_retry_context.get("evidence_start_index")
        if isinstance(evidence_start_index, int) and evidence_start_index >= 0:
            normalized["evidence_start_index"] = evidence_start_index

        retrieval_error_start_index = raw_retry_context.get("retrieval_error_start_index")
        if isinstance(retrieval_error_start_index, int) and retrieval_error_start_index >= 0:
            normalized["retrieval_error_start_index"] = retrieval_error_start_index

        retrieval_diagnostic_start_index = raw_retry_context.get("retrieval_diagnostic_start_index")
        if (
            isinstance(retrieval_diagnostic_start_index, int)
            and retrieval_diagnostic_start_index >= 0
        ):
            normalized["retrieval_diagnostic_start_index"] = retrieval_diagnostic_start_index

        score_avg = raw_retry_context.get("score_avg")
        if isinstance(score_avg, (int, float)):
            normalized["score_avg"] = float(score_avg)
        elif score_avg is None and "score_avg" in raw_retry_context:
            normalized["score_avg"] = None

        return normalized or None

    @staticmethod
    def _normalize_retrieval_diagnostics(raw_diagnostics: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_diagnostics, list):
            return []

        normalized: list[dict[str, Any]] = []
        for item in raw_diagnostics:
            if not isinstance(item, dict):
                continue
            try:
                attempt = int(item.get("attempt", 0) or 0)
            except (TypeError, ValueError):
                attempt = 0
            normalized.append(
                {
                    "tool": str(item.get("tool") or "").strip(),
                    "route": str(item.get("route") or "").strip(),
                    "status": str(item.get("status") or "").strip(),
                    "message": str(item.get("message") or ""),
                    "query": str(item.get("query") or ""),
                    "attempt": attempt,
                }
            )
        return normalized

    @staticmethod
    def _normalize_planner_diagnostics(raw_planner_diagnostics: Any) -> Dict[str, Any] | None:
        if not isinstance(raw_planner_diagnostics, dict):
            return None

        fallback_routes_raw = raw_planner_diagnostics.get("fallback_routes")
        fallback_routes = (
            [str(route) for route in fallback_routes_raw if route]
            if isinstance(fallback_routes_raw, list)
            else []
        )
        required_routes_raw = raw_planner_diagnostics.get("required_routes")
        required_routes = (
            [str(route) for route in required_routes_raw if route]
            if isinstance(required_routes_raw, list)
            else []
        )
        status = raw_planner_diagnostics.get("status")
        reason = raw_planner_diagnostics.get("reason")
        intent_required = bool(raw_planner_diagnostics.get("intent_required", False))
        override_applied = bool(raw_planner_diagnostics.get("override_applied", False))
        override_reason = raw_planner_diagnostics.get("override_reason")
        if override_reason not in {
            "missing_required_retrieval",
            "missing_required_routes",
            "upload_retriever_missing",
        }:
            override_reason = None

        if (
            not status
            and reason is None
            and not fallback_routes
            and not required_routes
            and not intent_required
            and not override_applied
            and override_reason is None
        ):
            return None
        return {
            "status": str(status) if status is not None else "",
            "reason": (str(reason) if reason is not None else None),
            "fallback_routes": fallback_routes,
            "intent_required": intent_required,
            "required_routes": required_routes,
            "override_applied": override_applied,
            "override_reason": override_reason,
        }

    def _prepare_graph_state(
        self,
        user_input: str,
        upload_file_path: Optional[str],
    ) -> tuple[dict[str, Any], int | None]:
        state = {
            "user_input": user_input,
            "messages": self.messages,
            "session_metadata": self._snapshot_session_metadata(),
        }
        upload_retriever_build_ms: int | None = None

        if upload_file_path is not None:
            if (
                self.upload_file_path != upload_file_path
                or getattr(self, "upload_retriever_handle", None) is None
            ):
                self._cleanup_upload_retriever()
                build_started = time.perf_counter()
                new_handle = build_temp_retriever(
                    upload_file_path,
                    api_key=self.settings.openai_api_key,
                )
                upload_retriever_build_ms = elapsed_ms(build_started, time.perf_counter())
                self.upload_retriever_handle = new_handle
                self.upload_file_path = upload_file_path

            handle = getattr(self, "upload_retriever_handle", None)
            if handle is not None:
                state["retriever"] = handle.retriever
        else:
            self._cleanup_upload_retriever()
            self.upload_file_path = None

        return state, upload_retriever_build_ms

    def _invoke_graph(self, state: dict[str, Any]) -> tuple[dict[str, Any], int]:
        graph_started = time.perf_counter()
        response = self.graph.invoke(state)
        graph_total_ms = elapsed_ms(graph_started, time.perf_counter())
        return response, graph_total_ms

    def _extract_debug_info(
        self,
        response: dict[str, Any],
        updated_messages: list[Any],
        graph_total_ms: int,
        upload_retriever_build_ms: int | None,
    ) -> dict[str, Any]:
        tool_calls: list[str] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        model_name: str | None = None
        debug_errors: list[str] = []

        for state_error_key in (
            "retrieval_errors",
            "synthesis_errors",
            "validation_errors",
            "action_errors",
        ):
            raw_errors = response.get(state_error_key)
            if not isinstance(raw_errors, list):
                continue
            for error in raw_errors:
                text = str(error).strip()
                if text:
                    debug_errors.append(text)

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

        for message in current_turn_messages:
            if isinstance(message, AIMessage):
                tool_calls.extend(self._extract_tool_names_from_ai_message(message))
                usage = self._extract_token_usage_from_ai_message(message)
                total_prompt_tokens += usage["prompt_tokens"]
                total_completion_tokens += usage["completion_tokens"]
                total_tokens += usage["total_tokens"]
                if model_name is None:
                    model_name = self._extract_model_name_from_ai_message(message)
            elif isinstance(message, ToolMessage) and getattr(message, "name", ""):
                tool_calls.append(str(message.name))

        if total_tokens <= 0:
            total_tokens = total_prompt_tokens + total_completion_tokens

        observed_evidence = self._extract_observed_evidence(
            current_turn_messages,
            errors=debug_errors,
        )
        retry_context = self._normalize_retry_context(response.get("retry_context"))
        retrieval_diagnostics = self._normalize_retrieval_diagnostics(
            response.get("retrieval_diagnostics")
        )
        planner_diagnostics = self._normalize_planner_diagnostics(
            response.get("planner_diagnostics")
        )
        latency_breakdown = build_latency_breakdown(
            raw_trace=response.get("latency_trace"),
            graph_total_ms=graph_total_ms,
            upload_retriever_build_ms=upload_retriever_build_ms,
        )

        return {
            "tool_calls": tool_calls,
            "tool_call_count": len(tool_calls),
            "token_usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
            },
            "model_name": model_name,
            "errors": debug_errors,
            "observed_evidence": observed_evidence,
            "retry_context": retry_context,
            "retrieval_diagnostics": retrieval_diagnostics,
            "planner_diagnostics": planner_diagnostics,
            "latency_breakdown": latency_breakdown.model_dump(mode="json"),
        }

    def _assemble_run_result(
        self,
        response: dict[str, Any],
        updated_messages: list[Any],
        debug_info: dict[str, Any],
    ) -> dict[str, Any]:
        final_answer = ""
        file_path = ""

        for message in reversed(updated_messages):
            if isinstance(message, HumanMessage):
                break

            if not final_answer and isinstance(message, AIMessage):
                final_answer = self._extract_text_content(message.content)
            elif (
                not file_path
                and isinstance(message, ToolMessage)
                and message.name == "save_text"
            ):
                try:
                    tool_result_dict: Dict[str, Any] = json.loads(
                        self._extract_text_content(message.content)
                    )
                    extracted_path = tool_result_dict.get("file_path")
                    if extracted_path and os.path.exists(extracted_path):
                        file_path = extracted_path
                except json.JSONDecodeError:
                    continue

            if final_answer and file_path:
                break

        raw_response_payload = response.get("response_payload")
        if isinstance(raw_response_payload, dict):
            response_payload = dict(raw_response_payload)
        else:
            response_payload = build_empty_response_payload(answer=final_answer).model_dump(mode="json")

        return {
            "message": final_answer,
            "filepath": file_path,
            "response": response,
            "response_payload": response_payload,
            "debug": debug_info,
        }

    def run_agent_flow(self, user_input: str, upload_file_path: Optional[str] = None) -> dict:
        if user_input.lower() in {"exit", "종료", "quit", "q"}:
            self.close()
            reset_message = "Chat session has been reset. Start again."
            return {
                "message": reset_message,
                "filepath": "",
                "response": None,
                "response_payload": {
                    "answer": reset_message,
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
                "debug": {
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "model_name": None,
                    "errors": [],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            }

        try:
            state, upload_retriever_build_ms = self._prepare_graph_state(user_input, upload_file_path)
            response, graph_total_ms = self._invoke_graph(state)
            updated_messages = response["messages"]
            self.messages = updated_messages
            debug_info = self._extract_debug_info(
                response,
                updated_messages,
                graph_total_ms,
                upload_retriever_build_ms,
            )
            return self._assemble_run_result(response, updated_messages, debug_info)

        except Exception as exc:
            self._cleanup_upload_retriever()
            self.upload_file_path = None
            log_event(logger, logging.ERROR, "agent_execution_error", error=exc)
            message = str(exc)
            return {
                "message": message,
                "filepath": "",
                "response": None,
                "response_payload": {
                    "answer": message,
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
                "debug": {
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "token_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                    },
                    "model_name": None,
                    "errors": [message],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            }
