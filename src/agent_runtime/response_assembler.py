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
        raw_path = str(extracted_path or "").strip()
        if not raw_path:
            return ""

        candidate = Path(raw_path)
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

    @staticmethod
    def _current_turn_messages(updated_messages: list[Any]) -> list[Any]:
        current_turn: list[Any] = []
        for message in reversed(updated_messages):
            if isinstance(message, HumanMessage):
                break
            current_turn.append(message)
        current_turn.reverse()
        return current_turn

    @staticmethod
    def _parse_tool_payload(message: ToolMessage) -> dict[str, Any]:
        try:
            parsed = json.loads(ResponseAssembler.extract_text_content(message.content))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _format_save_receipt(payload: dict[str, Any], *, file_path: str) -> str:
        status = str(payload.get("status") or "").strip().lower()
        if status == "error":
            error = str(payload.get("error") or "").strip()
            return f"저장 실패: {error}" if error else "저장 실패"
        if file_path:
            return f"저장 완료: {file_path}"
        message = str(payload.get("message") or "").strip()
        return f"저장 완료: {message}" if message else "저장 완료"

    @staticmethod
    def _format_slack_receipt(payload: dict[str, Any]) -> str:
        status = str(payload.get("status") or "").strip().lower()
        if status == "error":
            error = str(payload.get("error") or "").strip()
            return f"전송 실패: {error}" if error else "전송 실패"
        if status == "skipped":
            reason = str(payload.get("reason") or "").strip()
            return f"전송 보류: {reason}" if reason else "전송 보류"

        channel_id = str(payload.get("channel_id") or "").strip()
        target_type = str(payload.get("target_type") or "").strip().lower()
        if target_type == "dm" and channel_id:
            return f"전송 완료: Slack DM ({channel_id})"
        if channel_id:
            return f"전송 완료: Slack 채널 ({channel_id})"
        return "전송 완료: Slack"

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

        current_turn_messages = self._current_turn_messages(updated_messages)
        latest_ai_answer = ""
        receipts: list[str] = []

        for message in reversed(current_turn_messages):
            if isinstance(message, AIMessage):
                latest_ai_answer = self.extract_text_content(message.content)
                break

        for message in current_turn_messages:
            if not isinstance(message, ToolMessage):
                continue

            payload = self._parse_tool_payload(message)
            if message.name == "save_text":
                extracted_path = payload.get("file_path")
                if extracted_path:
                    file_path = self._resolve_file_path(str(extracted_path))
                receipts.append(self._format_save_receipt(payload, file_path=file_path))
            elif message.name == "slack_notify":
                receipts.append(self._format_slack_receipt(payload))

        receipts = [receipt for receipt in receipts if receipt.strip()]
        if latest_ai_answer:
            final_answer = latest_ai_answer
        elif not final_answer:
            final_answer = str(response_state.payload.answer or "").strip()

        if receipts:
            receipt_block = "\n".join(receipts)
            if final_answer.strip():
                if receipt_block not in final_answer:
                    final_answer = f"{final_answer}\n\n{receipt_block}"
            else:
                final_answer = receipt_block

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
