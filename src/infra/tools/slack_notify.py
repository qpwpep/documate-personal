from __future__ import annotations

from collections.abc import Callable
from typing import Any

from slack_sdk.errors import SlackApiError

from src.infra.settings import AppSettings
from src.infra.slack_utils import create_slack_client, resolve_destination


def build_slack_notify_tool(settings: AppSettings) -> Callable[..., dict[str, Any]]:
    slack_client = create_slack_client(settings.slack_bot_token)

    def slack_notify(
        text: str,
        user_id: str | None = None,
        email: str | None = None,
        channel_id: str | None = None,
    ) -> dict[str, Any]:
        if not slack_client:
            return {
                "status": "skipped",
                "reason": "SLACK_BOT_TOKEN not set",
                "error_code": "SLACK_AUTH_FAILED",
            }

        resolved_id, target_type = resolve_destination(
            slack_client=slack_client,
            channel_id=channel_id,
            user_id=user_id,
            email=email,
            default_user_id=settings.slack_default_user_id,
            default_email=settings.slack_default_dm_email,
        )

        if not resolved_id:
            return {
                "status": "skipped",
                "reason": "No valid Slack destination resolved",
                "error_code": "SLACK_DESTINATION_MISSING",
            }

        try:
            slack_client.chat_postMessage(channel=resolved_id, text=text)
            return {"status": "ok", "channel_id": resolved_id, "target_type": target_type}
        except SlackApiError as exc:
            return {"status": "error", "error": str(exc), "error_code": "SLACK_AUTH_FAILED"}

    return slack_notify
