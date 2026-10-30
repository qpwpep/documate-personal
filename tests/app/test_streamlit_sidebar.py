"""Sidebar guidance and navigation through the actual Streamlit widgets."""
from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest
import requests
from streamlit.testing.v1 import AppTest


@pytest.fixture(autouse=True)
def slack_default_http(monkeypatch):
    state = {"payload": {"selector": None, "failure": None}, "unavailable": False, "requests": []}

    def send(_adapter, request, **_kwargs):
        assert request.method == "GET"
        assert urlparse(request.url).path == "/slack/default-recipient"
        state["requests"].append(request.url)
        if state["unavailable"]:
            raise requests.ConnectionError("default recipient service unavailable")
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.url = request.url
        response._content = json.dumps(state["payload"]).encode()
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return state


def _sidebar_app() -> None:
    import streamlit as st

    from src.app.client import AgentRequestContext, AgentSessionClient
    from src.app.web.streamlit_sidebar import render_sidebar
    from src.app.web.streamlit_state import Conversation
    from src.core.uploads import UploadManifest

    if "conversations" not in st.session_state:
        st.session_state["conversations"] = {
            str(index): Conversation(
                id=str(index),
                client=AgentSessionClient(AgentRequestContext("http://unused.local", str(index))),
                messages=[{"role": "user", "content": f"질문 {index}"}],
            )
            for index in range(1, 5)
        }
        st.session_state["active_conversation_id"] = "4"
    result = render_sidebar(manifest=UploadManifest(epoch="test", revision=0, files=[]))
    st.json({
        "selected_conversation": result.selected_conversation,
        "recipient": result.slack_recipient.model_dump() if result.slack_recipient else None,
    })


def test_all_conversations_picker_is_secondary_but_keeps_older_conversations_reachable() -> None:
    app = AppTest.from_function(_sidebar_app).run()

    assert not app.exception
    all_conversations = next(item for item in app.expander if item.label == "모든 대화 보기")
    assert not all_conversations.proto.expanded
    picker = next(item for item in all_conversations.selectbox if item.label == "대화 선택")
    assert picker.options == ["4. 질문 4", "3. 질문 3", "2. 질문 2", "1. 질문 1"]
    picker.select("1").run()
    assert json.loads(app.json[0].value)["selected_conversation"] == "1"


def test_sidebar_keeps_secondary_settings_collapsed_without_losing_preferences() -> None:
    app = AppTest.from_function(_sidebar_app).run()

    appearance = next(item for item in app.expander if item.label == "화면 설정")
    delivery = next(item for item in app.expander if item.label == "Slack 수신자 설정")
    assert not appearance.proto.expanded
    assert not delivery.proto.expanded
    appearance.radio[0].set_value("다크").run()
    next(item for item in app.radio if item.label == "수신자 유형").set_value("사용자").run()
    next(item for item in app.text_input if item.label == "Slack 수신자").set_value("U0123456789").run()
    next(item for item in app.radio if item.label == "수신자 유형").set_value("채널").run()
    next(item for item in app.radio if item.label == "수신자 유형").set_value("사용자").run()

    assert not app.exception
    assert next(item for item in app.radio if item.label == "테마").value == "다크"
    assert next(item for item in app.text_input if item.label == "Slack 수신자").value == "U0123456789"


def test_slack_guidance_explains_destination_and_explicit_send_without_discarding_input() -> None:
    app = AppTest.from_function(_sidebar_app).run()
    next(item for item in app.radio if item.label == "수신자 유형").set_value("채널").run()

    guidance = "\n".join(item.value for item in app.caption)
    assert "채널 이름 대신" in guidance
    assert "채널 ID" in guidance
    assert "방금 답변을 Slack으로 보내줘" in guidance
    assert any(item.label == "수신자 ID 찾는 방법" for item in app.expander)
    field = next(item for item in app.text_input if item.label == "Slack 수신자")
    field.set_value("general").run()

    assert not app.exception
    assert json.loads(app.json[0].value)["recipient"] == {"kind": "channel", "value": "general"}
    assert any("C, G 또는 D" in item.value for item in app.caption)
    feedback = next(item.value for item in app.markdown if 'id="documate-slack-recipient-help"' in item.value)
    assert 'data-slack-invalid="true"' in feedback
    assert 'role="status"' in feedback
    assert "채널 이름은 사용할 수 없습니다" in feedback
    field = next(item for item in app.text_input if item.label == "Slack 수신자")
    field.set_value("C0123456789").run()
    assert json.loads(app.json[0].value)["recipient"] == {"kind": "channel", "value": "C0123456789"}
    feedback = next(item.value for item in app.markdown if 'id="documate-slack-recipient-help"' in item.value)
    assert 'data-slack-invalid="false"' in feedback
    assert "수신자 형식을 확인했습니다" in feedback

    next(item for item in app.radio if item.label == "수신자 유형").set_value("미지정").run()
    assert not app.text_input
    assert not any('id="documate-slack-recipient-help"' in item.value for item in app.markdown)


@pytest.mark.parametrize("kind,value,label", [
    ("user", "U0123456789", "사용자"),
    ("email", "recipient@example.com", "이메일"),
])
def test_omitted_recipient_displays_the_server_default_without_selecting_an_explicit_target(
    slack_default_http, kind, value, label,
) -> None:
    slack_default_http["payload"] = {"selector": {"kind": kind, "value": value}, "failure": None}

    app = AppTest.from_function(_sidebar_app).run()

    assert not app.exception
    recipient_type = next(item for item in app.radio if item.label == "수신자 유형")
    assert recipient_type.options[0] == "직접 지정 안 함"
    guidance = "\n".join(item.value for item in app.markdown)
    assert f"기본 수신자: {label}" in guidance
    assert value in guidance
    assert any("질문에도 수신자를 지정하지 않은 새 전송" in item.value for item in app.caption)
    assert json.loads(app.json[0].value)["recipient"] is None

    next(item for item in app.radio if item.label == "수신자 유형").set_value("채널").run()
    next(item for item in app.text_input if item.label == "Slack 수신자").set_value("C0123456789").run()
    assert not any("기본 수신자:" in item.value for item in app.markdown)
    assert json.loads(app.json[0].value)["recipient"] == {"kind": "channel", "value": "C0123456789"}
    assert len(slack_default_http["requests"]) == 1


def test_no_default_requires_a_recipient_but_preserves_ordinary_chat() -> None:
    app = AppTest.from_function(_sidebar_app).run()

    assert not app.exception
    assert any("기본 수신자가 없습니다" in item.value for item in app.info)
    assert any("질문에 수신자를 지정" in item.value for item in app.info)
    assert json.loads(app.json[0].value)["recipient"] is None
    assert not app.button(key="documate_new_chat").disabled


def test_invalid_default_is_distinct_from_missing_and_explicit_input_still_works(slack_default_http) -> None:
    slack_default_http["payload"] = {
        "selector": None,
        "failure": {"stage": "selection", "code": "configuration_error",
                    "message": "기본 Slack 수신자 설정이 잘못되었습니다.", "next_action": "fix_configuration"},
    }
    app = AppTest.from_function(_sidebar_app).run()

    assert not app.exception
    assert any("기본 Slack 수신자 설정이 잘못되었습니다" in item.value for item in app.warning)
    assert not any("기본 수신자가 없습니다" in item.value for item in app.info)
    assert json.loads(app.json[0].value)["recipient"] is None

    next(item for item in app.radio if item.label == "수신자 유형").set_value("사용자").run()
    next(item for item in app.text_input if item.label == "Slack 수신자").set_value("U9876543210").run()
    assert not app.warning
    assert json.loads(app.json[0].value)["recipient"] == {"kind": "user", "value": "U9876543210"}


def test_unavailable_default_can_be_rechecked_without_being_reported_as_missing(slack_default_http) -> None:
    slack_default_http["unavailable"] = True
    app = AppTest.from_function(_sidebar_app).run()

    assert not app.exception
    assert any("기본 수신자를 확인하지 못했습니다" in item.value for item in app.warning)
    assert not any("기본 수신자가 없습니다" in item.value for item in app.info)
    assert json.loads(app.json[0].value)["recipient"] is None

    slack_default_http["unavailable"] = False
    slack_default_http["payload"] = {"selector": {"kind": "user", "value": "U0123456789"}, "failure": None}
    next(item for item in app.button if item.label == "기본 수신자 다시 확인").click().run()
    assert not app.exception
    assert not app.warning
    assert any("U0123456789" in item.value for item in app.markdown)


@pytest.mark.parametrize("unavailable", [False, True])
def test_default_lookup_does_not_repeat_during_unrelated_sidebar_reruns(slack_default_http, unavailable) -> None:
    slack_default_http["unavailable"] = unavailable
    app = AppTest.from_function(_sidebar_app).run()
    app.run()
    next(item for item in app.radio if item.label == "테마").set_value("다크").run()
    app.button(key="documate_new_chat").click().run()

    assert not app.exception
    assert slack_default_http["requests"] == ["http://unused.local/slack/default-recipient"]
    assert bool(app.warning) is unavailable


def test_default_lookup_refreshes_explicitly_and_keeps_each_server_separate(slack_default_http) -> None:
    slack_default_http["payload"] = {"selector": {"kind": "user", "value": "U0123456789"}, "failure": None}
    app = AppTest.from_function(_sidebar_app).run()
    slack_default_http["payload"] = {"selector": {"kind": "email", "value": "updated@example.com"}, "failure": None}
    app.run()

    assert any("U0123456789" in item.value for item in app.markdown)
    assert len(slack_default_http["requests"]) == 1

    next(item for item in app.button if item.label == "기본 수신자 다시 확인").click().run()
    assert any("updated@example.com" in item.value for item in app.markdown)
    assert len(slack_default_http["requests"]) == 2

    conversation = app.session_state["conversations"]["4"]
    conversation.client.context.fastapi_url = "http://another-server.local"
    slack_default_http["payload"] = {"selector": None, "failure": None}
    app.run()
    assert any("기본 수신자가 없습니다" in item.value for item in app.info)
    assert slack_default_http["requests"][-1] == "http://another-server.local/slack/default-recipient"
    assert len(slack_default_http["requests"]) == 3

    conversation.client.context.fastapi_url = "http://unused.local"
    app.run()
    assert any("updated@example.com" in item.value for item in app.markdown)
    assert len(slack_default_http["requests"]) == 3
