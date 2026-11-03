from __future__ import annotations

from pydantic import ValidationError

from src.core.slack_contract import (
    ExplicitRecipient, RecipientInput, RecipientIntent, RecipientSelection, RecipientSelector,
    SlackDefault, SlackDelivery, SlackFailure, UnresolvedRecipient,
)


def _same_intent(left: RecipientIntent, right: RecipientIntent) -> bool:
    return left.model_dump(exclude={"evidence_ids"}) == right.model_dump(exclude={"evidence_ids"})


def select_slack_delivery(
    *, request_id: str, intent: RecipientIntent, request_recipient: RecipientInput | None,
    default: SlackDefault, previous: SlackDelivery | None = None,
) -> SlackDelivery:
    """Select once. Resolution failure never becomes an omitted recipient."""
    if previous is not None and previous.status == "unknown":
        if previous.selection.request_id != request_id:
            raise ValueError("Slack delivery belongs to another request")
        # A correction cannot resolve whether the original send happened.
        # Only a separate new request can authorize a new delivery obligation.
        return previous
    selector_input = None
    if request_recipient is not None:
        try:
            selector_input = RecipientSelector.model_validate(request_recipient.model_dump())
        except ValidationError:
            pass
    if previous is not None and previous.intent.state == "unresolved":
        if (previous.failure is not None and previous.failure.stage == "input"
                and selector_input is not None):
            # A corrected form can resolve its own never-selected recipient.
            # Text ambiguity and an already selected identity remain pinned.
            previous = None
        else:
            # Input disappearance is not a new recipient decision.
            return previous
    if previous is not None and (intent.state == "omitted" or _same_intent(intent, previous.intent)):
        if previous.selection is not None:
            if previous.selection.request_id != request_id:
                raise ValueError("Slack delivery belongs to another request")
            return previous
        # A genuinely omitted, never-selected target can recover after input or
        # configuration is corrected. Unresolved explicit intent stays blocked.
        if previous.intent.state != "omitted" or intent.state != "omitted":
            return previous
    if intent.state == "unresolved":
        code = {"invalid": "recipient_invalid", "ambiguous": "recipient_ambiguous",
                "unverified": "recipient_invalid", "conflict": "recipient_conflict"}[intent.reason]
        return SlackDelivery(intent=intent, status="not_sent", failure=SlackFailure(
            stage="selection", code=code, next_action="correct_input",
            message="명시한 Slack 수신자를 확정하지 못했습니다. 보낼 대상 하나의 ID 또는 이메일을 확인해 주세요.",
        ))
    if request_recipient is not None and selector_input is None:
        return SlackDelivery(
            intent=UnresolvedRecipient(raw_input=request_recipient.value, reason="invalid"),
            status="not_sent", failure=SlackFailure(
                stage="input", code="recipient_invalid", next_action="correct_input",
                message="Slack 전송을 위해 사이드바의 수신자를 확인해 주세요. 선택한 유형에 맞는 채널 ID, 사용자 ID 또는 이메일을 입력한 뒤 다시 전송할 수 있습니다. 기존 답변은 유지됩니다.",
            ),
        )
    if intent.state == "explicit":
        if selector_input is not None and selector_input != intent.selector:
            unresolved = UnresolvedRecipient(
                raw_input=f"{intent.selector.value} / {selector_input.value}",
                reason="conflict", evidence_ids=intent.evidence_ids,
            )
            return select_slack_delivery(request_id=request_id, intent=unresolved,
                                         request_recipient=None, default=default)
        selector, source = intent.selector, "user_text"
    elif selector_input is not None:
        selector, source = selector_input, "request_input"
        intent = ExplicitRecipient(selector=selector)
    elif default.failure is not None:
        return SlackDelivery(intent=intent, status="not_sent", failure=default.failure)
    elif default.selector is None:
        return SlackDelivery(intent=intent, status="not_sent", failure=SlackFailure(
            stage="selection", code="recipient_missing", next_action="correct_input",
            message="Slack으로 보낼 대상의 채널 ID, 사용자 ID 또는 이메일을 알려 주세요.",
        ))
    else:
        selector, source = default.selector, "configured_default"
    return SlackDelivery(intent=intent, selection=RecipientSelection(
        request_id=request_id, source=source, selector=selector,
    ))
