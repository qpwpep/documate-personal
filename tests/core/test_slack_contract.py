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












def _settings(user_id=None, email=None):
    return AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        slack_default_user_id=user_id, slack_default_dm_email=email,
    )














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
