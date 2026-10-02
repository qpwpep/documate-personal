from __future__ import annotations

import httpx
from langchain_openai import ChatOpenAI

from src.app.agent_manager import AgentFlowManager
from src.infra.settings import AppSettings


def test_schema_rejection_is_a_service_failure_without_a_fake_user_contract(monkeypatch):
    """A valid question must not become a clarification because OpenAI rejected our schema."""
    requests = []

    def provider(request):
        requests.append(request)
        return httpx.Response(400, json={"error": {
            "message": "Invalid schema for response_format: oneOf is not permitted",
            "type": "invalid_request_error", "param": "response_format", "code": "invalid_json_schema",
        }}, headers={"x-request-id": "provider-schema-test"})

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    with httpx.Client(transport=httpx.MockTransport(provider)) as client:
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kw: ChatOpenAI(
            **kw, http_client=client, base_url="https://provider.test/v1",
        ))
        manager = AgentFlowManager(AppSettings(
            _env_file=None, openai_api_key="test-key", tavily_api_key="test-key", verbose=False,
        ))
        before = manager._ensure_session().snapshot_conversation_memory()
        result = manager.run_agent_flow("pandas merge 사용법을 공식 문서 기준으로 설명해줘")

    assert result["status"] == "failed"
    assert result["problem"]["code"] == "provider_schema_invalid"
    assert result["problem"]["next_action"] == "fix_configuration"
    assert result["response"] is None
    assert "질문을 수정할 필요는 없습니다" in result["message"]
    assert len(requests) == 1
    assert manager._ensure_session().snapshot_conversation_memory() == before
    assert result["debug"]["tool_calls"] == []
