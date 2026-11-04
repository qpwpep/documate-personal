from __future__ import annotations

from src.core.answer_schema import ActionReceipt
from src.core.slack_contract import SlackDelivery


def build_slack_receipt(delivery: SlackDelivery) -> ActionReceipt:
    if delivery.status == "sent":
        return ActionReceipt(kind="slack_notify", status="success", slack=delivery, invocation_id=delivery.invocation_id)
    if delivery.failure is None:
        raise ValueError("cannot report an unattempted Slack delivery")
    status = ("unknown" if delivery.status == "unknown" else
              "skipped" if delivery.failure.stage in {"input", "selection"} else "error")
    return ActionReceipt(
        kind="slack_notify", status=status, slack=delivery, invocation_id=delivery.invocation_id,
        message=delivery.failure.message if status == "skipped" else None,
        error=delivery.failure.message if status != "skipped" else None,
        error_code=f"SLACK_{delivery.failure.code.upper()}",
    )
