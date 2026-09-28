from __future__ import annotations

from collections.abc import Callable
from typing import Any

from slack_sdk.errors import SlackApiError
from slack_sdk.web import WebClient

from src.core.slack_contract import SlackDelivery, SlackTarget
from src.infra.slack_utils import (
    api_failure,
    create_slack_client,
    failed_delivery,
    protocol_failure,
    resolve_destination,
    transport_failure,
    update_delivery,
)


def _send_message(slack_client: WebClient, *, text: str, target: SlackTarget) -> Any:
    """The sender accepts a resolved channel, never a recipient choice."""
    return slack_client.chat_postMessage(channel=target.channel_id, text=text)


def build_slack_notify_tool(token: str | None) -> Callable[..., SlackDelivery]:
    slack_client = create_slack_client(token)

    def slack_notify(*, text: str, delivery: SlackDelivery) -> SlackDelivery:
        # Validate even for internal direct callers before touching Slack.
        delivery = SlackDelivery.model_validate(delivery.model_dump(mode="json"))
        if delivery.status in {"sent", "unknown"}:
            return delivery
        if delivery.selection is None:
            # Policy rejection cannot become a fresh recipient choice or be
            # erased by authentication checking at the transport boundary.
            return resolve_destination(slack_client, delivery)
        delivery = update_delivery(delivery, status="pending", failure=None, message_ts=None)
        if slack_client is None:
            return failed_delivery(
                delivery, stage="selection", code="authentication_failed",
                message="Slack 토큰이 설정되지 않아 전송하지 않았습니다. 토큰을 설정한 뒤 같은 수신자로 재시도해 주세요.",
                next_action="fix_configuration",
            )

        delivery = resolve_destination(slack_client, delivery)
        if delivery.status != "pending" or delivery.target is None:
            return delivery
        try:
            response = _send_message(slack_client, text=text, target=delivery.target)
        except SlackApiError as error:
            return api_failure(delivery, "send", error)
        except Exception:
            return transport_failure(delivery, "send")

        if not isinstance(response.data, dict):
            return protocol_failure(delivery, "send")
        channel_id, message_ts = response.get("channel"), response.get("ts")
        if (channel_id != delivery.target.channel_id
                or not isinstance(message_ts, str) or not message_ts.strip()):
            return protocol_failure(delivery, "send")
        return update_delivery(delivery, status="sent", failure=None, message_ts=message_ts)

    return slack_notify
