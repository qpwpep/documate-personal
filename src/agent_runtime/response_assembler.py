from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..answer_schema import build_empty_response_payload
from ..contracts.boundary.response import get_response_state


class ResponseAssembler:
    @staticmethod
    def _resolve_file_path(extracted_path: str) -> str:
        candidate = Path(str(extracted_path or "").strip())
        if not candidate:
            return ""
        if candidate.is_absolute():
            return str(candidate)
        resolved = (Path.cwd() / candidate).resolve()
        if resolved.exists():
            return str(resolved)
        return str(candidate)

    @staticmethod
    def extract_text_content(content: Any) -> str:
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

    def assemble(
        self,
        *,
        response: dict[str, Any],
        updated_messages: list[Any],
        debug_info: dict[str, Any],
    ) -> dict[str, Any]:
        response_state = get_response_state(response)
        final_answer = response_state.final_answer
        file_path = ""
        latest_ai_answer = ""

        for message in reversed(updated_messages):
            if isinstance(message, HumanMessage):
                break

            if not latest_ai_answer and isinstance(message, AIMessage):
                latest_ai_answer = self.extract_text_content(message.content)
            elif (
                not file_path
                and isinstance(message, ToolMessage)
                and message.name == "save_text"
            ):
                try:
                    tool_result_dict = json.loads(self.extract_text_content(message.content))
                    extracted_path = tool_result_dict.get("file_path")
                    if extracted_path:
                        file_path = self._resolve_file_path(str(extracted_path))
                except json.JSONDecodeError:
                    continue

            if latest_ai_answer and file_path:
                break

        if latest_ai_answer:
            final_answer = latest_ai_answer
        elif not final_answer:
            final_answer = str(response_state.payload.answer or "").strip()

        response_payload = response_state.payload.model_dump(mode="json")
        if not response_payload:
            response_payload = build_empty_response_payload(answer=final_answer).model_dump(mode="json")
        elif final_answer:
            response_payload["answer"] = final_answer

        return {
            "message": final_answer,
            "filepath": file_path,
            "response": response,
            "response_payload": response_payload,
            "debug": debug_info,
        }
