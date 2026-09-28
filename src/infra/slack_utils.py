from __future__ import annotations

from typing import Any, Literal

from slack_sdk.errors import SlackApiError
from slack_sdk.web import WebClient

from src.core.slack_contract import RecipientSelector, SlackDelivery, SlackFailure, SlackTarget


SlackStage = Literal["input", "selection", "lookup", "open_dm", "send"]


def create_slack_client(token: str | None) -> WebClient | None:
    # A send whose response is lost must not be repeated by a hidden SDK retry.
    return WebClient(token=token, retry_handlers=[]) if token else None


def update_delivery(delivery: SlackDelivery, **updates: Any) -> SlackDelivery:
    return SlackDelivery.model_validate({**delivery.model_dump(mode="json"), **updates})


def failed_delivery(
    delivery: SlackDelivery,
    *,
    stage: SlackStage,
    code: str,
    message: str,
    next_action: str,
    unknown: bool = False,
    slack_error: str | None = None,
    retry_after_seconds: int | None = None,
) -> SlackDelivery:
    failure = SlackFailure.model_validate({
        "stage": stage,
        "code": code,
        "message": message,
        "next_action": next_action,
        "slack_error": slack_error,
        "retry_after_seconds": retry_after_seconds,
    })
    return update_delivery(
        delivery, status="unknown" if unknown else "not_sent", failure=failure, message_ts=None,
    )


def api_failure(delivery: SlackDelivery, stage: SlackStage, error: SlackApiError) -> SlackDelivery:
    response = error.response
    data = response.data if isinstance(response.data, dict) else {}
    slack_error = data.get("error")
    slack_error = slack_error if isinstance(slack_error, str) else None
    status = response.status_code or 0

    if stage == "send" and status >= 500:
        return failed_delivery(
            delivery, stage=stage, code="delivery_unknown", unknown=True,
            message="Slack 서버 오류로 전송 여부를 확인하지 못했습니다. 지정한 대화에서 메시지를 먼저 확인해 주세요.",
            next_action="verify_delivery", slack_error=slack_error,
        )
    if slack_error in {
        "not_authed", "invalid_auth", "token_revoked", "token_expired", "account_inactive",
        "not_allowed_token_type", "two_factor_setup_required",
    }:
        code, message, action = "authentication_failed", "Slack 인증에 실패했습니다. 토큰과 계정 인증 설정을 확인한 뒤 같은 수신자로 재시도해 주세요.", "fix_configuration"
    elif slack_error in {
        "missing_scope", "no_permission", "restricted_action", "ekm_access_denied", "cannot_dm_bot",
        "access_denied", "accesslimited", "team_access_not_granted", "enterprise_is_restricted",
        "not_in_channel", "app_access_restricted", "restricted_action_read_only_channel",
        "restricted_action_thread_only_channel",
    }:
        code, message, action = "permission_denied", "Slack 접근 또는 전송 권한이 제한되었습니다. 앱 권한과 대상 접근 설정을 확인한 뒤 같은 수신자로 재시도해 주세요.", "fix_configuration"
    elif slack_error in {"users_not_found", "user_not_found"}:
        code, message, action = "target_not_found", "지정한 Slack 수신자를 조회하지 못했습니다. 사용자 ID 또는 이메일과 워크스페이스를 확인해 주세요.", "correct_input"
    elif slack_error in {"channel_not_found", "user_not_visible", "is_archived", "user_disabled", "user_not_active"}:
        code, message, action = "target_unavailable", "지정한 Slack 대상이 없거나 현재 앱에서 접근할 수 없습니다. 대상과 접근 권한을 확인해 주세요.", "correct_input"
    elif slack_error in {"ratelimited", "rate_limited"} or status == 429:
        retry_after = next((value for key, value in (response.headers or {}).items() if key.lower() == "retry-after"), None)
        try:
            seconds = max(0, int(retry_after)) if retry_after is not None else None
        except (TypeError, ValueError):
            seconds = None
        return failed_delivery(
            delivery, stage=stage, code="rate_limited",
            message="Slack 호출 한도에 도달해 아직 전송하지 않았습니다. 잠시 후 같은 수신자로 재시도해 주세요.",
            next_action="retry_same_target", slack_error=slack_error, retry_after_seconds=seconds,
        )
    elif stage == "send":
        # Internal errors and malformed/unknown responses do not establish that
        # Slack rejected the message before delivery.
        return failed_delivery(
            delivery, stage=stage, code="delivery_unknown", unknown=True,
            message="Slack 전송 여부를 확인하지 못했습니다. 지정한 대화에서 메시지를 확인한 뒤 다시 전송해 주세요.",
            next_action="verify_delivery", slack_error=slack_error,
        )
    else:
        code, message, action = "temporary_failure", "Slack 대상 조회 중 오류가 발생해 아직 전송하지 않았습니다. 같은 수신자로 재시도해 주세요.", "retry_same_target"
    return failed_delivery(
        delivery, stage=stage, code=code, message=message, next_action=action, slack_error=slack_error,
    )


def transport_failure(delivery: SlackDelivery, stage: SlackStage) -> SlackDelivery:
    if stage == "send":
        return failed_delivery(
            delivery, stage=stage, code="delivery_unknown", unknown=True,
            message="Slack 전송 응답을 받지 못해 전달 여부가 불명확합니다. 지정한 대화에서 메시지를 먼저 확인해 주세요.",
            next_action="verify_delivery",
        )
    return failed_delivery(
        delivery, stage=stage, code="temporary_failure",
        message="Slack 대상 조회에 연결하지 못해 아직 전송하지 않았습니다. 같은 수신자로 재시도해 주세요.",
        next_action="retry_same_target",
    )


def protocol_failure(delivery: SlackDelivery, stage: SlackStage) -> SlackDelivery:
    sending = stage == "send"
    return failed_delivery(
        delivery, stage=stage, code="protocol_error", unknown=sending,
        message=("Slack 응답이 요청 대상의 전송 성공을 확인하지 못했습니다. 지정한 대화에서 메시지를 먼저 확인해 주세요."
                 if sending else "Slack 대상 조회 응답이 올바르지 않아 전송하지 않았습니다. 같은 수신자로 재시도해 주세요."),
        next_action="verify_delivery" if sending else "retry_same_target",
    )


def resolve_destination(slack_client: WebClient | None, delivery: SlackDelivery) -> SlackDelivery:
    """Resolve one selected identity; defaults and replacement targets do not exist here."""
    if delivery.target is not None:
        return delivery
    selection = delivery.selection
    if selection is None:
        if delivery.failure is not None:
            return delivery
        unresolved = delivery.intent.state == "unresolved"
        code = {
            "invalid": "recipient_invalid", "ambiguous": "recipient_ambiguous",
            "unverified": "recipient_invalid", "conflict": "recipient_conflict",
        }.get(delivery.intent.reason, "recipient_missing") if unresolved else "recipient_missing"
        return failed_delivery(
            delivery, stage="selection", code=code,
            message="Slack 수신자를 하나의 올바른 사용자 ID, 채널 ID 또는 이메일로 지정해 주세요.",
            next_action="correct_input",
        )
    selector = selection.selector
    if slack_client is None:
        return failed_delivery(
            delivery, stage="selection", code="authentication_failed",
            message="Slack 토큰을 설정한 뒤 같은 수신자로 재시도해 주세요.", next_action="fix_configuration",
        )
    if selector.kind == "channel":
        return update_delivery(delivery, target=SlackTarget(channel_id=selector.value))

    user_id = delivery.resolved_user_id
    if user_id is None and selector.kind == "user":
        user_id = selector.value
    if user_id is None:
        try:
            response = slack_client.users_lookupByEmail(email=selector.value)
        except SlackApiError as error:
            return api_failure(delivery, "lookup", error)
        except Exception:
            return transport_failure(delivery, "lookup")
        try:
            user_id = RecipientSelector(kind="user", value=response["user"]["id"]).value
        except (TypeError, KeyError, ValueError):
            return protocol_failure(delivery, "lookup")

    # Persist the user even if opening the DM fails. Re-resolving an email on
    # retry could bind a different user if the email's ownership has changed.
    delivery = update_delivery(delivery, resolved_user_id=user_id)
    try:
        response = slack_client.conversations_open(users=user_id)
    except SlackApiError as error:
        return api_failure(delivery, "open_dm", error)
    except Exception:
        return transport_failure(delivery, "open_dm")
    try:
        channel_id = response["channel"]["id"]
        if not isinstance(channel_id, str) or not channel_id.startswith("D"):
            return protocol_failure(delivery, "open_dm")
        target = SlackTarget(channel_id=channel_id, user_id=user_id)
    except (TypeError, KeyError, ValueError):
        return protocol_failure(delivery, "open_dm")
    return update_delivery(delivery, target=target)
