from __future__ import annotations

import hashlib

import pytest

from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.slack_contract import RecipientSelector
from src.core.contracts.graph_state import SessionMetadata
from tests.core.test_pending_request_delivery import delivery_tools
from tests.core.test_request_contract_retries import _SaveConversation, _fail_save_publication


@pytest.mark.parametrize("delivery_tools", [{"/chat.postMessage": None}], indirect=True)
@pytest.mark.parametrize("corrected_channel", ["C123", "C456"])
def test_body_correction_preserves_unknown_delivery_for_the_original_payload(delivery_tools, corrected_channel):
    _save, slack, requests, _output = delivery_tools
    conversation = _SaveConversation(["원래 본문", "수정 본문", "원래 본문"], slack=slack)
    recipient = {"state": "explicit", "selector": {"kind": "channel", "value": "C123"},
                 "evidence_ids": ["request"]}
    send = {"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}}
    first = conversation.turn("첫 본문을 작성해서 C123으로 보내줘",
        body={"kind": "compose", "instruction": "첫 본문 작성", "evidence_ids": ["request"]},
        actions=send, slack_recipient=recipient)
    original = first["runtime"].pending_request.slack_delivery

    corrected_recipient = {**recipient, "selector": {"kind": "channel", "value": corrected_channel}}
    second = conversation.turn(f"본문을 새 내용으로 바꿔서 {corrected_channel}으로 보내줘", relation="correction",
        body={"kind": "compose", "instruction": "새 내용 작성", "evidence_ids": ["request"]},
        actions=send, slack_recipient=corrected_recipient)
    corrected = AnswerResponse.model_validate(second["response"].result.model_dump(mode="json"))
    receipt = corrected.actions[0]
    assert receipt.status == "skipped"
    assert receipt.slack is None
    assert receipt.previous_slack == original
    assert "수정된 본문은 전송하지 않았습니다" in receipt.message
    assert export_answer_text(corrected) == "수정 본문"
    assert export_answer_text(second["runtime"].pending_request.response) == "수정 본문"
    assert second["runtime"].pending_request.slack_delivery == original
    payload = "원래 본문".encode("utf-8")
    assert original.operation.payload_sha256 == hashlib.sha256(payload).hexdigest()
    assert original.operation.byte_count == len(payload)
    mismatched = corrected.model_dump(mode="json")
    mismatched["actions"] = [first["response"].result.actions[0].model_dump(mode="json")]
    with pytest.raises(ValueError, match="current Slack delivery must match"):
        AnswerResponse.model_validate(mismatched)

    repeated = conversation.turn("수정 본문 그대로 다시 보내줘", relation="supplement", actions={},
        body={"kind": "copy_answer", "source": {"ref": "pending"}})
    assert repeated["response"].result.actions[0].previous_slack == original
    assert repeated["response"].result.actions[0].status == "skipped"

    restored = conversation.turn("원래 본문으로 다시 바꿔서 C123으로 보내줘", relation="correction",
        body={"kind": "compose", "instruction": "첫 본문 작성", "evidence_ids": ["request"]},
        actions=send, slack_recipient=recipient)
    assert restored["response"].result.actions[0].slack == original
    assert restored["response"].result.actions[0].status == "unknown"
    assert [request["payload"]["text"] for request in requests if request["path"] == "/chat.postMessage"] == ["원래 본문"]

    fresh = conversation.turn("별도 새 요청으로 이 본문을 C123으로 다시 보내줘", relation="new",
        body={"kind": "copy_answer", "source": {"ref": "previous"}},
        actions=send, slack_recipient=recipient)
    new_delivery = fresh["response"].result.actions[0].slack
    assert new_delivery.status == "sent"
    assert new_delivery.operation.request_id != original.operation.request_id
    assert new_delivery.operation.operation_id != original.operation.operation_id
    assert new_delivery.operation.payload_sha256 == original.operation.payload_sha256
    assert fresh["response"].result.actions[0].previous_slack is None
    assert fresh["runtime"].pending_request is None
    assert len([request for request in requests if request["path"] == "/chat.postMessage"]) == 2


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
    original = first["runtime"].pending_request.slack_delivery
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
    assert first["runtime"].pending_request.slack_delivery.failure.code == "recipient_conflict"

    second = conversation.turn("남은 전송을 다시 시도해줘", relation="supplement", actions={})
    assert requests == []
    assert second["runtime"].pending_request.slack_delivery.failure.code == "recipient_conflict"

    third = conversation.turn("수신자는 C123이 맞아", relation="supplement", actions={},
        slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]})
    assert [request["payload"] for request in requests if request["path"] == "/chat.postMessage"] == [
        {"channel": "C123", "text": "확인 후 전달할 본문"},
    ]
    assert third["runtime"].pending_request is None


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
    original = first["runtime"].pending_request
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


@pytest.mark.parametrize("value", ["", "not-a-channel"])
def test_incomplete_sidebar_recipient_blocks_only_send_and_recovers_after_correction(delivery_tools, value):
    from src.core.answer_schema import export_answer_text

    _save, slack, requests, _output = delivery_tools
    conversation = _SaveConversation(["수신자를 고쳐도 유지할 본문"], slack=slack)
    first = conversation.turn("설명을 작성해 Slack으로 보내줘",
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": value}),
        body={"kind": "compose", "instruction": "설명을 작성해", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}})

    assert requests == []
    pending = first["runtime"].pending_request
    assert pending.body_prepared
    assert pending.slack_delivery.failure.code == "recipient_invalid"
    assert pending.slack_delivery.failure.next_action == "correct_input"
    assert export_answer_text(pending.response) == "수신자를 고쳐도 유지할 본문"

    repeated = conversation.turn("다시 보내줘", relation="supplement", actions={},
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": value}))
    assert repeated["runtime"].pending_request.response.content == pending.response.content
    assert requests == []

    omitted = conversation.turn("다시 보내줘", relation="supplement", actions={})
    assert omitted["runtime"].pending_request.slack_delivery.failure.code == "recipient_invalid"
    assert requests == []

    second = conversation.turn("수신자를 고쳤으니 다시 보내줘", relation="supplement", actions={},
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "C123"}))

    assert second["runtime"].pending_request is None
    receipt = second["response"].result.actions[0]
    assert receipt.status == "success"
    assert [request["payload"] for request in requests] == [
        {"channel": "C123", "text": "수신자를 고쳐도 유지할 본문"},
    ]


@pytest.mark.parametrize("delivery_tools", [{"/chat.postMessage": None}], indirect=True)
def test_changing_sidebar_recipient_never_resends_an_unknown_delivery(delivery_tools):
    _save, slack, requests, _output = delivery_tools
    conversation = _SaveConversation(["전달 여부를 확인할 본문"], slack=slack)
    first = conversation.turn("설명을 작성해 Slack으로 보내줘",
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "C123"}),
        body={"kind": "compose", "instruction": "설명을 작성해", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}})
    original = first["runtime"].pending_request.slack_delivery
    assert original.status == "unknown"

    second = conversation.turn("다시 보내줘", relation="supplement", actions={},
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "COTHER"}))

    assert second["runtime"].pending_request.slack_delivery == original
    assert second["response"].result.actions[0].status == "unknown"
    assert len(requests) == 1
    assert requests[0]["payload"]["channel"] == "C123"
