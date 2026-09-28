from __future__ import annotations

from src.core.slack_contract import (
    ExplicitRecipient, RecipientIntent, RecipientSelection, RecipientSelector,
    SlackDefault, SlackDelivery, SlackFailure, UnresolvedRecipient,
)


def _same_intent(left: RecipientIntent, right: RecipientIntent) -> bool:
    return left.model_dump(exclude={"evidence_ids"}) == right.model_dump(exclude={"evidence_ids"})


def select_slack_delivery(
    *, request_id: str, intent: RecipientIntent, request_recipient: RecipientSelector | None,
    default: SlackDefault, previous: SlackDelivery | None = None,
) -> SlackDelivery:
    """Select once. Resolution failure never becomes an omitted recipient."""
    if previous is not None and previous.intent.state == "unresolved":
        # Accepted current-turn confirmation/correction clears this record in
        # the planner. Input disappearance is not a new recipient decision.
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
    if intent.state == "explicit":
        if request_recipient is not None and request_recipient != intent.selector:
            unresolved = UnresolvedRecipient(
                raw_input=f"{intent.selector.value} / {request_recipient.value}",
                reason="conflict", evidence_ids=intent.evidence_ids,
            )
            return select_slack_delivery(request_id=request_id, intent=unresolved,
                                         request_recipient=None, default=default)
        selector, source = intent.selector, "user_text"
    elif request_recipient is not None:
        selector, source = request_recipient, "request_input"
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
