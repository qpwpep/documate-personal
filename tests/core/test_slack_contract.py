from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.slack_contract import (
    ExplicitRecipient, OmittedRecipient, RecipientSelector, SlackDefault, SlackDelivery, UnresolvedRecipient,
)
from src.infra.settings import AppSettings
from src.core.contracts.graph_state import SessionMetadata
from src.runtime.nodes.actions.policy import select_slack_delivery
from tests.core.test_pending_action_delivery import delivery_tools
from tests.core.test_request_contract_retries import _SaveConversation, _fail_save_publication


def test_unresolved_pending_recipient_cannot_be_replaced_by_new_metadata():
    original = select_slack_delivery(
        request_id="request", intent=UnresolvedRecipient(raw_input="our team", reason="ambiguous"),
        request_recipient=None, default=SlackDefault(),
    )

    followup = select_slack_delivery(
        request_id="request", intent=OmittedRecipient(),
        request_recipient=RecipientSelector(kind="user", value="UMETADATA"),
        default=SlackDefault(selector=RecipientSelector(kind="user", value="UDEFAULT")), previous=original,
    )

    assert followup.intent == original.intent
    assert followup.status == "not_sent"
    assert followup.selection is None
    assert followup.failure == original.failure


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


def _settings(user_id=None, email=None):
    return AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        slack_default_user_id=user_id, slack_default_dm_email=email,
    )


@pytest.mark.parametrize("user_id,email", [(None, None), ("", " ")])
def test_absent_default_requires_a_recipient(user_id, email):
    default = _settings(user_id, email).slack_default_recipient()

    result = select_slack_delivery(request_id="request", intent=OmittedRecipient(), request_recipient=None, default=default)

    assert default.selector is None
    assert default.failure is None
    assert result.selection is None
    assert result.status == "not_sent"
    assert result.failure.code == "recipient_missing"


@pytest.mark.parametrize("user_id,email,kind,value", [
    (" UDEFAULT ", None, "user", "UDEFAULT"),
    (None, " default@example.invalid ", "email", "default@example.invalid"),
])
def test_one_valid_default_can_be_selected_only_for_an_omitted_recipient(user_id, email, kind, value):
    default = _settings(user_id, email).slack_default_recipient()

    result = select_slack_delivery(request_id="request", intent=OmittedRecipient(), request_recipient=None, default=default)

    assert result.intent.state == "omitted"
    assert result.selection.source == "configured_default"
    assert result.selection.selector == RecipientSelector(kind=kind, value=value)
    assert result.failure is None


@pytest.mark.parametrize("user_id,email", [
    ("UDEFAULT", "default@example.invalid"), ("wrong-user", None), (None, "wrong-email"),
])
def test_invalid_default_blocks_default_delivery_but_keeps_valid_explicit_selection_usable(user_id, email):
    default = _settings(user_id, email).slack_default_recipient()
    intended = ExplicitRecipient(selector=RecipientSelector(kind="channel", value="CINTENDED"))

    omitted = select_slack_delivery(request_id="omitted", intent=OmittedRecipient(), request_recipient=None, default=default)
    explicit = select_slack_delivery(request_id="explicit", intent=intended, request_recipient=None, default=default)

    assert default.selector is None
    assert default.failure.code == "configuration_error"
    assert omitted.selection is None
    assert omitted.status == "not_sent"
    assert omitted.failure.next_action == "fix_configuration"
    assert explicit.selection.selector == intended.selector
    assert explicit.selection.source == "user_text"
    assert explicit.status == "pending"
    assert explicit.failure is None


def test_conflicting_text_and_structured_input_require_a_new_recipient_decision():
    intended = ExplicitRecipient(selector=RecipientSelector(kind="email", value="intended@example.invalid"))

    result = select_slack_delivery(
        request_id="request", intent=intended,
        request_recipient=RecipientSelector(kind="user", value="UMETADATA"),
        default=_settings("UDEFAULT").slack_default_recipient(),
    )

    assert result.selection is None
    assert result.status == "not_sent"
    assert result.intent.state == "unresolved"
    assert result.intent.reason == "conflict"
    assert "intended@example.invalid" in result.intent.raw_input
    assert "UMETADATA" in result.intent.raw_input
    assert result.failure.next_action == "correct_input"


def test_correcting_invalid_default_configuration_can_resume_a_truly_omitted_recipient():
    original = select_slack_delivery(
        request_id="request", intent=OmittedRecipient(), request_recipient=None,
        default=_settings("UDEFAULT", "default@example.invalid").slack_default_recipient(),
    )
    assert original.failure.code == "configuration_error"
    assert original.selection is None

    corrected = select_slack_delivery(
        request_id="request", intent=OmittedRecipient(), request_recipient=None,
        default=_settings("UDEFAULT").slack_default_recipient(), previous=original,
    )

    assert corrected.status == "pending"
    assert corrected.selection.source == "configured_default"
    assert corrected.selection.selector.value == "UDEFAULT"
    assert corrected.intent.state == "omitted"
    assert corrected.failure is None


@pytest.mark.parametrize("change_body", [False, True])
def test_body_correction_with_new_send_instruction_cannot_reuse_previous_slack_success(delivery_tools, monkeypatch, change_body):
    """A new authorized body must actually be delivered, even after partial success."""
    _save, slack, requests, _output = delivery_tools
    conversation = _SaveConversation(["처음 전송한 본문", "명시적으로 수정한 새 본문"], slack=slack)
    with monkeypatch.context() as failure:
        _fail_save_publication(failure)
        first = conversation.turn(
            "설명을 작성하고 저장한 뒤 C123으로 보내줘",
            body={"kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"]},
            actions={name: {"intent": "requested", "evidence_ids": ["request"]} for name in ("save_text", "slack_notify")},
            slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]},
        )
    pending = first["runtime"].pending_action
    assert pending is not None
    assert pending.completed_actions == ("slack_notify",)
    assert pending.slack_delivery.status == "sent"

    body = ({"kind": "transform_answer", "source": {"ref": "pending"},
             "instruction": "새 본문으로 수정한다.", "evidence_ids": ["request"]} if change_body else
            {"kind": "copy_answer", "source": {"ref": "pending"}})
    second = conversation.turn(
        "본문을 수정하고 같은 수신자에게 다시 보내줘" if change_body else "같은 본문을 같은 수신자에게 다시 보내줘",
        relation="correction", body=body,
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
    )

    sent = [request["payload"] for request in requests if request["path"] == "/chat.postMessage"]
    assert sent == [
        {"channel": "C123", "text": "처음 전송한 본문"},
        {"channel": "C123", "text": "명시적으로 수정한 새 본문" if change_body else "처음 전송한 본문"},
    ]
    receipt = next(item for item in second["response"].result.actions if item.kind == "slack_notify")
    assert receipt.status == "success"
    assert receipt.slack.selection == pending.slack_delivery.selection


@pytest.mark.parametrize("delivery_tools", [{"/chat.postMessage": None}], indirect=True)
def test_implicit_pending_followup_does_not_repeat_an_unknown_delivery(delivery_tools, monkeypatch):
    _save, slack, requests, _output = delivery_tools
    conversation = _SaveConversation(["전달 여부가 불확실한 본문"], slack=slack)
    with monkeypatch.context() as failure:
        _fail_save_publication(failure)
        first = conversation.turn(
            "설명을 작성하고 저장한 뒤 C123으로 보내줘",
            body={"kind": "compose", "instruction": "설명을 작성한다.", "evidence_ids": ["request"]},
            actions={name: {"intent": "requested", "evidence_ids": ["request"]} for name in ("save_text", "slack_notify")},
            slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]},
        )
    original = first["runtime"].pending_action.slack_delivery
    assert original.status == "unknown"

    second = conversation.turn(
        "남은 작업을 마저 처리해줘", relation="supplement", actions={},
        body={"kind": "copy_answer", "source": {"ref": "pending"}},
    )

    receipt = next(item for item in second["response"].result.actions if item.kind == "slack_notify")
    assert receipt.status == "unknown"
    assert receipt.slack == original
    assert len([request for request in requests if request["path"] == "/chat.postMessage"]) == 1


def test_conflicting_recipient_needs_explicit_confirmation_before_a_pending_send(delivery_tools):
    _save, slack, requests, _output = delivery_tools

    conversation = _SaveConversation(["확인 후 전달할 본문"], slack=slack)
    first = conversation.turn("본문을 작성해서 C123으로 보내줘",
        session_metadata=SessionMetadata(slack_recipient=RecipientSelector(kind="channel", value="COTHER")),
        body={"kind": "compose", "instruction": "본문을 작성해", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
        slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]})
    assert requests == []
    assert first["runtime"].pending_action.slack_delivery.failure.code == "recipient_conflict"

    second = conversation.turn("남은 전송을 다시 시도해줘", relation="supplement", actions={})
    assert requests == []
    assert second["runtime"].pending_action.slack_delivery.failure.code == "recipient_conflict"

    third = conversation.turn("수신자는 C123이 맞아", relation="supplement", actions={},
        slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]})
    assert [request["payload"] for request in requests if request["path"] == "/chat.postMessage"] == [
        {"channel": "C123", "text": "확인 후 전달할 본문"},
    ]
    assert third["runtime"].pending_action is None


@pytest.mark.parametrize("delivery_tools", [{
    "/users.lookupByEmail": [{"ok": True, "user": {"id": "UINTENDED"}}, {"ok": True, "user": {"id": "UOTHER"}}],
    "/conversations.open": [{"ok": False, "error": "missing_scope"}],
}], indirect=True)
def test_current_email_reconfirmation_preserves_the_resolved_pending_user(delivery_tools):
    _save, slack, requests, _output = delivery_tools
    conversation = _SaveConversation(["원래 사용자에게 보낼 본문"], slack=slack)
    intent = {"state": "explicit", "selector": {"kind": "email", "value": "intended@example.invalid"},
              "evidence_ids": ["request"]}
    first = conversation.turn("intended@example.invalid로 보내줘",
        body={"kind": "compose", "instruction": "본문을 작성해", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}}, slack_recipient=intent)
    original = first["runtime"].pending_action
    assert original.slack_delivery.resolved_user_id == "UINTENDED"
    assert all(request["path"] != "/chat.postMessage" for request in requests)

    second = conversation.turn("수신자는 intended@example.invalid가 맞아", relation="supplement", actions={},
                               slack_recipient=intent)

    receipt = next(item for item in second["response"].result.actions if item.kind == "slack_notify")
    assert receipt.status == "success"
    assert receipt.slack.selection == original.slack_delivery.selection
    assert receipt.slack.target.user_id == "UINTENDED"
    assert [request["payload"]["channel"] for request in requests if request["path"] == "/chat.postMessage"] == ["DINTENDED"]
    assert set(original.contract.slack_recipient.evidence_ids) < set(second["runtime"].request_contract.slack_recipient.evidence_ids)
