"""Configured recipient references cross the real planner, action and Slack HTTP boundaries."""
from __future__ import annotations

import json

import pytest

from src.core.answer_schema import export_answer_text
from src.core.contracts.graph_state import SessionMetadata
from tests.core.test_pending_request_delivery import delivery_tools
from tests.core.test_request_contract_retries import _SaveConversation


class RecipientConversation(_SaveConversation):
    def invoke(self, messages):
        context = next(message for message in messages if message.name == "request_context")
        self.planner_context = json.loads(context.content.split("\n", 1)[1])
        return super().invoke(messages)


@pytest.mark.parametrize("selector,reference,channel", [
    ({"kind": "channel", "value": "C123BENCH"}, "지정된 벤치마크 팀 채널", "C123BENCH"),
    ({"kind": "user", "value": "U123BENCH"}, "선택한 수신자", "D123BENCH"),
    ({"kind": "email", "value": "chosen@example.invalid"}, "설정된 수신자", "DINTENDED"),
])
def test_configured_reference_sends_to_the_exact_offered_destination(delivery_tools, selector, reference, channel):
    _save, slack, requests, _output = delivery_tools
    conversation = RecipientConversation(["전달할 본문"], slack=slack)

    result = conversation.turn(
        f"본문을 작성해서 {reference}에 보내줘.",
        session_metadata=SessionMetadata(slack_recipient=selector),
        body={"kind": "compose", "instruction": "본문을 작성한다.", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
        slack_recipient={"state": "configured", "selector": selector, "raw_input": reference,
                         "evidence_ids": ["request"]},
    )

    assert conversation.planner_context["configured_recipient"] == selector
    delivery = result["response"].result.actions[0].slack
    assert delivery.status == "sent"
    assert delivery.intent.state == "configured"
    assert delivery.selection.source == "request_input"
    assert delivery.selection.selector.model_dump() == selector
    assert [request["payload"] for request in requests if request["path"] == "/chat.postMessage"] == [
        {"channel": channel, "text": "전달할 본문"},
    ]


@pytest.mark.parametrize("actual,selected", [
    (None, {"kind": "channel", "value": "C123BENCH"}),
    ({"kind": "user", "value": "U123BENCH"}, {"kind": "channel", "value": "C123BENCH"}),
    ({"kind": "channel", "value": "C123BENCH"}, {"kind": "channel", "value": "COTHER"}),
])
def test_unavailable_or_mismatched_configured_reference_preserves_body_without_sending(delivery_tools, actual, selected):
    _save, slack, requests, _output = delivery_tools
    conversation = RecipientConversation(["확인 후 보낼 본문"], slack=slack)

    result = conversation.turn(
        "본문을 작성해서 설정된 수신자에게 보내줘.",
        session_metadata=SessionMetadata(slack_recipient=actual),
        body={"kind": "compose", "instruction": "본문을 작성한다.", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
        slack_recipient={"state": "configured", "selector": selected, "raw_input": "설정된 수신자",
                         "evidence_ids": ["request"]},
    )

    assert requests == []
    pending = result["runtime"].pending_request
    assert pending.body_prepared
    assert export_answer_text(pending.response) == "확인 후 보낼 본문"
    assert pending.contract.slack_recipient.state == "unresolved"
    assert pending.contract.slack_recipient.raw_input == "설정된 수신자"
    assert pending.slack_delivery.selection is None
    assert pending.slack_delivery.failure.code == "recipient_invalid"


def test_named_unresolved_recipient_does_not_select_an_available_configuration(delivery_tools):
    _save, slack, requests, _output = delivery_tools
    conversation = RecipientConversation(["대상을 확인할 본문"], slack=slack)
    result = conversation.turn(
        "본문을 작성해서 운영팀에 보내줘.",
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "C123BENCH"}),
        body={"kind": "compose", "instruction": "본문을 작성한다.", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
        slack_recipient={"state": "unresolved", "raw_input": "운영팀", "reason": "ambiguous",
                         "evidence_ids": ["request"]},
    )
    assert requests == []
    assert result["runtime"].pending_request.slack_delivery.failure.code == "recipient_ambiguous"


def test_configured_reference_requires_its_exact_current_user_phrase(delivery_tools):
    _save, slack, requests, _output = delivery_tools
    conversation = RecipientConversation(["확인 후 보낼 본문"], slack=slack)
    result = conversation.turn(
        "본문을 작성해서 운영팀에 보내줘.",
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "C123"}),
        body={"kind": "compose", "instruction": "본문을 작성한다.", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
        slack_recipient={"state": "configured", "selector": {"kind": "channel", "value": "C123"},
                         "raw_input": "설정된 채널", "evidence_ids": ["request"]},
    )
    assert requests == []
    assert result["runtime"].pending_request.contract.slack_recipient.state == "unresolved"


def test_configured_destination_alone_does_not_authorize_sending(delivery_tools):
    _save, slack, requests, _output = delivery_tools
    conversation = RecipientConversation(["화면에만 표시할 본문"], slack=slack)
    result = conversation.turn(
        "선택한 채널에 보여줄 본문을 작성하되 아직 전송하지 마.",
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "C123"}),
        body={"kind": "compose", "instruction": "본문을 작성한다.", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "forbidden", "evidence_ids": ["request"]}},
        slack_recipient={"state": "configured", "selector": {"kind": "channel", "value": "C123"},
                         "raw_input": "선택한 채널", "evidence_ids": ["request"]},
    )
    assert requests == []
    assert export_answer_text(result["response"].result) == "화면에만 표시할 본문"
    assert result["response"].result.actions == []


@pytest.mark.parametrize("delivery_tools", [{"/chat.postMessage": [{"ok": False, "error": "ratelimited"}]}], indirect=True)
def test_configured_reference_remains_pinned_when_retry_uses_changed_settings(delivery_tools):
    _save, slack, requests, _output = delivery_tools
    conversation = RecipientConversation(["같은 대상에게 재시도할 본문"], slack=slack)
    first = conversation.turn(
        "본문을 작성해서 선택한 채널에 보내줘.",
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "C123"}),
        body={"kind": "compose", "instruction": "본문을 작성한다.", "evidence_ids": ["request"]},
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["request"]}},
        slack_recipient={"state": "configured", "selector": {"kind": "channel", "value": "C123"},
                         "raw_input": "선택한 채널", "evidence_ids": ["request"]},
    )
    original = first["runtime"].pending_request.slack_delivery
    second = conversation.turn(
        "같은 본문을 같은 대상에게 다시 보내줘.", relation="supplement", actions={},
        session_metadata=SessionMetadata(slack_recipient={"kind": "channel", "value": "COTHER"}),
    )
    delivery = second["response"].result.actions[0].slack
    assert delivery.status == "sent"
    assert delivery.selection == original.selection
    assert [request["payload"]["channel"] for request in requests] == ["C123", "C123"]
