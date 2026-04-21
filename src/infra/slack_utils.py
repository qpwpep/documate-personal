from typing import Optional, Tuple

from slack_sdk.errors import SlackApiError
from slack_sdk.web import WebClient


def create_slack_client(token: Optional[str]) -> Optional[WebClient]:
    return WebClient(token=token) if token else None


def _classify_channel(channel_id: str) -> str:
    if channel_id.startswith("D"):
        return "DM"
    if channel_id.startswith("C"):
        return "Public Channel"
    if channel_id.startswith("G"):
        return "Private Channel"
    return "Unknown Channel"


def _resolve_user_id(
    slack_client: Optional[WebClient],
    user_id: Optional[str],
    email: Optional[str],
    default_user_id: Optional[str] = None,
    default_email: Optional[str] = None,
    require_user_prefix: bool = True,
) -> Optional[str]:
    if user_id and (not require_user_prefix or user_id.startswith("U")):
        return user_id

    if email and slack_client:
        try:
            response = slack_client.users_lookupByEmail(email=email)
            return response["user"]["id"]
        except SlackApiError:
            pass

    if default_user_id and (not require_user_prefix or default_user_id.startswith("U")):
        return default_user_id

    if default_email and slack_client:
        try:
            response = slack_client.users_lookupByEmail(email=default_email)
            return response["user"]["id"]
        except SlackApiError:
            pass

    return None


def _open_dm_channel(slack_client: Optional[WebClient], user_id: str) -> Optional[str]:
    if not slack_client:
        return None

    try:
        response = slack_client.conversations_open(users=user_id)
        return response["channel"]["id"]
    except SlackApiError:
        return None


def resolve_destination(
    slack_client: Optional[WebClient],
    channel_id: Optional[str] = None,
    user_id: Optional[str] = None,
    email: Optional[str] = None,
    default_user_id: Optional[str] = None,
    default_email: Optional[str] = None,
    require_user_prefix: bool = True,
) -> Tuple[Optional[str], str]:
    if channel_id:
        return channel_id, _classify_channel(channel_id)

    resolved_user_id = _resolve_user_id(
        slack_client=slack_client,
        user_id=user_id,
        email=email,
        default_user_id=default_user_id,
        default_email=default_email,
        require_user_prefix=require_user_prefix,
    )
    if not resolved_user_id:
        return None, "Unknown"

    dm_channel_id = _open_dm_channel(slack_client, resolved_user_id)
    if not dm_channel_id:
        return None, "Unknown"

    return dm_channel_id, "DM"
