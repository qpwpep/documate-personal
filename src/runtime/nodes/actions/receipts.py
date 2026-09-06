from __future__ import annotations

from typing import Any

from src.core.answer_schema import ActionReceipt
from src.core.contracts import SlackDestination


def _status(result: Any) -> str:
    return str(result.get("status") or "").strip().lower() if isinstance(result, dict) else ""


def _failure(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("error") or result.get("message") or "실행 결과를 확인할 수 없습니다.").strip()
    return "실행 결과를 확인할 수 없습니다."


def build_save_receipt(save_result: Any) -> ActionReceipt:
    status = _status(save_result)
    path = str(save_result.get("file_path") or "").strip() if isinstance(save_result, dict) else ""
    if status in {"success", "ok"} and path:
        return ActionReceipt(kind="save_text", status="success", file_path=path)
    if status == "skipped":
        return ActionReceipt(kind="save_text", status="skipped", message=str(save_result.get("reason") or "저장을 보류했습니다."))
    return ActionReceipt(kind="save_text", status="error", error=_failure(save_result))


def build_slack_receipt(*, slack_result: Any, destinations: SlackDestination) -> ActionReceipt:
    status = _status(slack_result)
    target = None
    for key in ("channel_id", "user_id", "email"):
        value = slack_result.get(key) if isinstance(slack_result, dict) else None
        if value:
            target = str(value)
            break
    target = target or destinations.channel_id or destinations.user_id or destinations.email
    if status in {"success", "ok"}:
        return ActionReceipt(kind="slack_notify", status="success", target=target)
    if status == "skipped":
        return ActionReceipt(kind="slack_notify", status="skipped", target=target, message=str(slack_result.get("reason") or "전송을 보류했습니다."))
    return ActionReceipt(kind="slack_notify", status="error", target=target, error=_failure(slack_result))
