from __future__ import annotations

import json
import os
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ..answer_schema import build_empty_response_payload
from ..contracts.graph_state import response_state


class ResponseAssembler:
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
        final_answer = response_state(response).final_answer
        file_path = ""

        for message in reversed(updated_messages):
            if isinstance(message, HumanMessage):
                break

            if not final_answer and isinstance(message, AIMessage):
                final_answer = self.extract_text_content(message.content)
            elif (
                not file_path
                and isinstance(message, ToolMessage)
                and message.name == "save_text"
            ):
                try:
                    tool_result_dict = json.loads(self.extract_text_content(message.content))
                    extracted_path = tool_result_dict.get("file_path")
                    if extracted_path and os.path.exists(extracted_path):
                        file_path = extracted_path
                except json.JSONDecodeError:
                    continue

            if final_answer and file_path:
                break

        response_payload = response_state(response).payload.model_dump(mode="json")
        if not response_payload:
            response_payload = build_empty_response_payload(answer=final_answer).model_dump(mode="json")

        return {
            "message": final_answer,
            "filepath": file_path,
            "response": response,
            "response_payload": response_payload,
            "debug": debug_info,
        }
