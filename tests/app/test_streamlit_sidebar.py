"""Sidebar guidance and navigation through the actual Streamlit widgets."""
from __future__ import annotations

import json

from streamlit.testing.v1 import AppTest


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
    field = next(item for item in app.text_input if item.label == "Slack 수신자")
    field.set_value("C0123456789").run()
    assert json.loads(app.json[0].value)["recipient"] == {"kind": "channel", "value": "C0123456789"}
