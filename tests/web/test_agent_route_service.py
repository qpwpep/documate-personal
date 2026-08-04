from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.infra.settings import AppSettings
from src.app.web.agent_request_service import AgentRequestResult
from src.app.web.app import create_app
from src.app.web.schemas import AgentDebugInfo, AgentResponsePayload, AgentRequest, AgentStreamEvent
from src.core.conversation_memory import DEFAULT_QUERY_MAX_CHARS


class _FakeAgentRequestService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.stream_calls: list[dict[str, object]] = []

    async def run(self, *, request_id: str, request_data: AgentRequest) -> AgentRequestResult:
        self.calls.append(
            {
                "request_id": request_id,
                "request_data": request_data,
            }
        )
        return AgentRequestResult(
            response=AgentResponsePayload.model_validate(
                {
                    "answer": "delegated answer",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                }
            ),
            trace=f"trace-{request_id}",
            file_path="output/result.txt",
            debug=(
                AgentDebugInfo(schema_version=3, observability_status="ok")
                if request_data.include_debug
                else None
            ),
        )

    def stream(self, *, request_id: str, request_data: AgentRequest):
        self.stream_calls.append(
            {
                "request_id": request_id,
                "request_data": request_data,
            }
        )

        async def event_stream():
            yield AgentStreamEvent(
                event="request_started",
                data={"request_id": request_id},
            )
            yield AgentStreamEvent(
                event="final_response",
                data={
                    "response": {
                        "answer": "delegated answer",
                        "claims": [],
                        "evidence": [],
                        "confidence": None,
                    },
                    "trace": f"trace-{request_id}",
                    "file_path": "output/result.txt",
                    "debug": None,
                },
            )
            yield AgentStreamEvent(event="done", data={})

        return event_stream()


class AgentRouteServiceDelegationTest(unittest.TestCase):
    def test_oversized_queries_are_rejected_before_route_delegation(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        fake_service = _FakeAgentRequestService()
        oversized_query = "x" * (DEFAULT_QUERY_MAX_CHARS + 1)
        with patch("src.app.web.app.get_settings", return_value=settings):
            with TestClient(create_app()) as client:
                client.app.state.agent_request_service = fake_service
                client.app.state.session_store = None
                client.app.state.runtime_cleaner = None

                response = client.post(
                    "/agent",
                    json={"query": oversized_query, "session_id": "demo-session"},
                )
                stream_response = client.post(
                    "/agent/stream",
                    json={"query": oversized_query, "session_id": "demo-session"},
                )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(stream_response.status_code, 422)
        self.assertEqual(fake_service.calls, [])
        self.assertEqual(fake_service.stream_calls, [])

    def test_agent_route_only_delegates_to_service(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        fake_service = _FakeAgentRequestService()
        with patch("src.app.web.app.get_settings", return_value=settings):
            with TestClient(create_app()) as client:
                client.app.state.agent_request_service = fake_service
                client.app.state.session_store = None
                client.app.state.runtime_cleaner = None

                response_without_debug = client.post(
                    "/agent",
                    json={
                        "query": "hello",
                        "session_id": "demo-session",
                        "include_debug": False,
                    },
                )
                response_with_debug = client.post(
                    "/agent",
                    json={
                        "query": "hello",
                        "session_id": "demo-session",
                        "include_debug": True,
                    },
                )

        self.assertEqual(response_without_debug.status_code, 200)
        self.assertEqual(response_with_debug.status_code, 200)
        self.assertEqual(
            response_without_debug.json()["response"],
            response_with_debug.json()["response"],
        )
        self.assertIsNone(response_without_debug.json()["debug"])
        self.assertIsNotNone(response_with_debug.json()["debug"])
        self.assertEqual(len(fake_service.calls), 2)
        self.assertEqual(fake_service.calls[0]["request_data"].query, "hello")
        self.assertEqual(fake_service.calls[1]["request_data"].include_debug, True)
        self.assertEqual(len(str(fake_service.calls[0]["request_id"])), 8)

    def test_agent_stream_route_streams_sse_events(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        fake_service = _FakeAgentRequestService()
        with patch("src.app.web.app.get_settings", return_value=settings):
            with TestClient(create_app()) as client:
                client.app.state.agent_request_service = fake_service
                client.app.state.session_store = None
                client.app.state.runtime_cleaner = None

                with client.stream(
                    "POST",
                    "/agent/stream",
                    json={
                        "query": "hello",
                        "session_id": "demo-session",
                    },
                ) as response:
                    body = "".join(response.iter_text())

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: request_started", body)
        self.assertIn("event: final_response", body)
        self.assertIn("event: done", body)
        self.assertEqual(len(fake_service.stream_calls), 1)
        self.assertEqual(fake_service.stream_calls[0]["request_data"].query, "hello")


if __name__ == "__main__":
    unittest.main()
