from __future__ import annotations

from typing import Any

from src.core.contracts import SlackDestination


def tool_status_is_success(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    status = str(result.get("status") or "").strip().lower()
    return status in {"", "ok", "success"}


def build_save_receipt(save_result: Any) -> str | None:
    if not isinstance(save_result, dict):
        return None
    file_path = str(save_result.get("file_path") or "").strip()
    if tool_status_is_success(save_result) and file_path:
        return f"저장 완료: {file_path}"
    error = str(save_result.get("error") or save_result.get("message") or "").strip()
    if error:
        return f"저장 실패: {error}"
    return None


def build_slack_target_label(*, destinations: SlackDestination, slack_result: Any) -> str:
    if isinstance(slack_result, dict):
        for key in ("channel_id", "user_id", "email"):
            value = str(slack_result.get(key) or "").strip()
            if value:
                if key == "channel_id":
                    return f"Slack ({value})"
                return f"Slack DM ({value})"
    if destinations.channel_id:
        return f"Slack ({destinations.channel_id})"
    if destinations.user_id:
        return f"Slack DM ({destinations.user_id})"
    if destinations.email:
        return f"Slack DM ({destinations.email})"
    return "Slack"


def build_slack_receipt(*, slack_result: Any, destinations: SlackDestination) -> str | None:
    if not isinstance(slack_result, dict):
        return None
    target_label = build_slack_target_label(destinations=destinations, slack_result=slack_result)
    if tool_status_is_success(slack_result):
        return f"전송 완료: {target_label}"
    error = str(slack_result.get("error") or slack_result.get("message") or "").strip()
    if error:
        return f"전송 실패: {error}"
    return None


def compose_action_response_text(*, delivery_body: str, receipts: list[str]) -> str:
    body = str(delivery_body or "").strip()
    receipt_block = "\n".join(receipt for receipt in receipts if receipt.strip()).strip()
    if body and receipt_block:
        return f"{body}\n\n{receipt_block}"
    return body or receipt_block
