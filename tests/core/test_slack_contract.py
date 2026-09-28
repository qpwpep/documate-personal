from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.slack_contract import ExplicitRecipient, SlackDelivery


def _serialized_user_delivery(**updates):
    """A serialized pending result as received from an internal caller/storage."""
    return {
        "intent": {"state": "explicit", "selector": {"kind": "user", "value": "UINTENDED"}},
        "selection": {"request_id": "request", "source": "request_input", "selector": {"kind": "user", "value": "UINTENDED"}},
        "resolved_user_id": "UINTENDED",
        "target": {"channel_id": "DINTENDED", "user_id": "UINTENDED"},
        "status": "pending",
        **updates,
    }


@pytest.mark.parametrize("updates", [
    {"selection": {"request_id": "request", "source": "configured_default", "selector": {"kind": "user", "value": "UINTENDED"}}},
    {"selection": {"request_id": "request", "source": "request_input", "selector": {"kind": "user", "value": "UOTHER"}}},
    {"resolved_user_id": "UOTHER"},
    {"target": {"channel_id": "C123", "user_id": "UINTENDED"}},
    {"target": {"channel_id": "DINTENDED", "user_id": "UOTHER"}},
    {"target": {"channel_id": "DINTENDED"}},
    {"selection": None},
    {"intent": {"state": "unresolved", "raw_input": "our team", "reason": "ambiguous"}},
])
def test_serialized_delivery_cannot_change_the_explicit_recipient(updates):
    with pytest.raises(ValidationError):
        SlackDelivery.model_validate(_serialized_user_delivery(**updates))


@pytest.mark.parametrize("updates", [
    {"status": "sent"},
    {"status": "sent", "message_ts": ""},
    {"status": "sent", "message_ts": "   "},
    {"status": "sent", "message_ts": "1.0", "target": None},
    {"status": "not_sent"},
    {"status": "unknown"},
    {"status": "pending", "message_ts": "1.0"},
    {"status": "unknown", "failure": {
        "stage": "send", "code": "delivery_unknown", "message": "Unconfirmed delivery.", "next_action": "retry_same_target",
    }},
])
def test_serialized_result_cannot_claim_an_unproven_success_or_safe_retry(updates):
    with pytest.raises(ValidationError):
        SlackDelivery.model_validate(_serialized_user_delivery(**updates))


@pytest.mark.parametrize("status,stage,code,next_action", [
    ("not_sent", "send", "delivery_unknown", "retry_same_target"),
    ("not_sent", "send", "delivery_unknown", "verify_delivery"),
    ("not_sent", "send", "protocol_error", "verify_delivery"),
    ("unknown", "lookup", "delivery_unknown", "verify_delivery"),
    ("unknown", "selection", "authentication_failed", "verify_delivery"),
])
def test_serialized_failure_cannot_hide_delivery_uncertainty_or_invent_a_send(status, stage, code, next_action):
    with pytest.raises(ValidationError):
        SlackDelivery.model_validate(_serialized_user_delivery(
            status=status,
            failure={"stage": stage, "code": code, "message": "Unconfirmed delivery.", "next_action": next_action},
        ))


def test_confirmed_delivery_roundtrip_preserves_identity_and_slack_acknowledgment():
    result = SlackDelivery.model_validate(_serialized_user_delivery(status="sent", message_ts="123.456"))

    restored = SlackDelivery.model_validate_json(result.model_dump_json())

    assert restored.status == "sent"
    assert restored.intent.selector.value == restored.selection.selector.value == restored.resolved_user_id == "UINTENDED"
    assert restored.target.channel_id == "DINTENDED"
    assert restored.target.user_id == "UINTENDED"
    assert restored.message_ts == "123.456"


@pytest.mark.parametrize("kind,value", [
    ("user", ""), ("email", " "), ("user", "C123"), ("channel", "U123"),
    ("user", "not-a-user"), ("email", "one@example.com,two@example.com"),
])
def test_invalid_explicit_input_is_rejected_instead_of_becoming_omitted(kind, value):
    with pytest.raises(ValidationError):
        ExplicitRecipient.model_validate({"state": "explicit", "selector": {"kind": kind, "value": value}})
