"""Real SDK, graph, service, HTTP/SSE, client and UI share the failure contract."""
from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI
from streamlit.testing.v1 import AppTest

from src.app.client import AgentRequestContext, AgentSessionClient
from tests.web.test_agent_stream_integration import agent_server


def test_provider_schema_rejection_is_one_failed_turn_from_sdk_to_ui(agent_server, monkeypatch):
    endpoint, server_app, _ = agent_server
    provider_requests = []

    def provider(request):
        provider_requests.append(request)
        return httpx.Response(400, json={"error": {
            "message": "PRIVATE invalid response_format oneOf; credential=secret",
            "type": "invalid_request_error", "param": "response_format", "code": "invalid_json_schema",
        }}, headers={"x-request-id": "private-provider-request"})

    with httpx.Client(transport=httpx.MockTransport(provider)) as provider_client:
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(
            **kwargs, http_client=provider_client, base_url="https://provider.test/v1",
        ))
        session_id = "schema-failure-delivery"
        client = AgentSessionClient(AgentRequestContext(fastapi_url=endpoint, session_id=session_id))
        manifest = client.refresh_uploads()
        session = server_app.state.session_store.get_or_create(session_id)._ensure_session()
        before = session.snapshot_conversation_memory()

        app = AppTest.from_string(f'''
import streamlit as st
from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web.streamlit_chat import process_chat_prompt, render_chat_history

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.requests_sent = 0
    st.session_state.client = AgentSessionClient(AgentRequestContext(
        fastapi_url={endpoint!r}, session_id={session_id!r}, include_debug=False,
    ))

def stream_agent(prompt):
    st.session_state.requests_sent += 1
    events = list(st.session_state.client.stream(prompt))
    st.session_state.events = events
    return events

if not st.session_state.messages:
    process_chat_prompt("pandas merge 사용법을 공식 문서 기준으로 설명해줘",
                        st.session_state.messages.append, st.session_state.messages.append, stream_agent)
else:
    render_chat_history(st.session_state.messages, {endpoint!r})
''').run(timeout=15)

    assert not app.exception
    events = app.session_state.events
    finals = [event for event in events if event.event == "final_response"]
    assert len(finals) == 1
    assert not [event for event in events if event.event == "error"]
    final = finals[0]
    assert final.result.status == "failed"
    assert final.result.problem.code == "provider_schema_invalid"
    assert final.result.problem.next_action == "fix_configuration"
    assert final.result.response is None
    assert final.result.upload_manifest == manifest
    assert final.result.request_id == events[0].data["request_id"]
    assert final.result.request_id
    assert final.data["debug"] is None
    assert len(provider_requests) == 1
    assert session.snapshot_conversation_memory() == before
    assert session.previous_response is None
    assert session.pending_action is None
    stages = [event.data["stage"] for event in events if event.event == "stage_started"]
    assert stages == ["planner"]
    assert [item.value for item in app.error] == [final.result.problem.message]
    assert any(final.result.request_id in item.value for item in app.caption)
    visible = " ".join(item.value for group in (app.error, app.warning, app.caption, app.markdown) for item in group)
    for private in ("PRIVATE", "oneOf", "credential=secret", "private-provider-request"):
        assert private not in visible
        assert private not in str(final.data)
    assert app.session_state.messages[-1]["result"].response is None
    app.run()
    assert app.session_state.requests_sent == 1
    assert len(provider_requests) == 1
    assert [item.value for item in app.error] == [final.result.problem.message]
