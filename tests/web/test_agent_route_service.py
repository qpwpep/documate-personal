from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from src.settings import AppSettings
from src.web.agent_request_service import AgentRequestResult
from src.web.app import create_app
from src.web.schemas import AgentDebugInfo, AgentResponsePayload, AgentRequest


class _FakeAgentRequestService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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


class AgentRouteServiceDelegationTest(unittest.TestCase):
    def test_agent_route_only_delegates_to_service(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        fake_service = _FakeAgentRequestService()
        with patch("src.web.app.get_settings", return_value=settings):
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


if __name__ == "__main__":
    unittest.main()
