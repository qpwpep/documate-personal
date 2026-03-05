import json
import os
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from .evidence import dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from .graph_builder import build_agent_graph
from .settings import AppSettings, get_settings
from .upload_helpers import build_temp_retriever


class AgentFlowManager:
    """Manages per-session LangGraph execution and message state."""

    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.graph = build_agent_graph(self.settings)
        self.messages: List[Any] = []
        self.retriever = None
        self.upload_file_path: Optional[str] = None

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

    def run_agent_flow(self, user_input: str, upload_file_path: Optional[str] = None) -> dict:
        current_messages = self.messages

        if user_input.lower() in {"exit", "종료", "quit", "q"}:
            self.messages = []
            self.retriever = None
            self.upload_file_path = None
            reset_message = "Chat session has been reset. Start again."
            return {
                "message": reset_message,
                "filepath": "",
                "response": None,
                "response_payload": {
                    "answer": reset_message,
                    "evidence": [],
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
                },
            }

        try:
            state = {
                "user_input": user_input,
                "messages": current_messages,
            }

            if upload_file_path is not None:
                if self.upload_file_path != upload_file_path:
                    self.upload_file_path = upload_file_path
                    self.retriever = build_temp_retriever(upload_file_path)

                if self.retriever is not None:
                    state["retriever"] = self.retriever
            else:
                self.retriever = None
                self.upload_file_path = None

            response = self.graph.invoke(state)

            updated_messages = response["messages"]
            self.messages = updated_messages

            final_answer = ""
            file_path = ""
            tool_calls: list[str] = []
            total_prompt_tokens = 0
            total_completion_tokens = 0
            total_tokens = 0
            model_name: str | None = None
            debug_errors: list[str] = []

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

            observed_evidence = self._extract_observed_evidence(
                current_turn_messages,
                errors=debug_errors,
            )

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

            if total_tokens <= 0:
                total_tokens = total_prompt_tokens + total_completion_tokens

            response_payload = {
                "answer": final_answer,
                "evidence": observed_evidence,
            }

            return {
                "message": final_answer,
                "filepath": file_path,
                "response": response,
                "response_payload": response_payload,
                "debug": {
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
                },
            }

        except Exception as exc:
            print(f"Agent execution error: {exc}")
            message = str(exc)
            return {
                "message": message,
                "filepath": "",
                "response": None,
                "response_payload": {
                    "answer": message,
                    "evidence": [],
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
                },
            }
