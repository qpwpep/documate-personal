from __future__ import annotations

from typing import Any

from langchain_core.tools import StructuredTool
from slack_sdk.errors import SlackApiError

from ..settings import AppSettings
from ..slack_utils import create_slack_client, resolve_destination
from ._common import SlackArgs


def build_slack_notify_tool(settings: AppSettings) -> Any:
    slack_client = create_slack_client(settings.slack_bot_token)

    def slack_notify(
        text: str,
        user_id: str | None = None,
        email: str | None = None,
        channel_id: str | None = None,
        target: str = "auto",
    ) -> dict:
        _ = target

        if not slack_client:
            return {"status": "skipped", "reason": "SLACK_BOT_TOKEN not set"}

        resolved_id, target_type = resolve_destination(
            slack_client=slack_client,
            channel_id=channel_id,
            user_id=user_id,
            email=email,
            default_user_id=settings.slack_default_user_id,
            default_email=settings.slack_default_dm_email,
        )

        if not resolved_id:
            return {"status": "skipped", "reason": "No valid Slack destination resolved"}

        try:
            slack_client.chat_postMessage(channel=resolved_id, text=text)
            return {"status": "ok", "channel_id": resolved_id, "target_type": target_type}
        except SlackApiError as exc:
            return {"status": "error", "error": str(exc)}

    return StructuredTool.from_function(
        name="slack_notify",
        description=(
            "Send a message to Slack. Use when the user asks to DM or post the answer to Slack. "
            "Provide either channel_id (C/G/D...) or a user_id/email for DM. "
            "If neither is present, the tool tries environment defaults."
        ),
        func=slack_notify,
        args_schema=SlackArgs,
    )
