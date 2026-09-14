from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from src.app.web.app import create_app
from src.app.web.schemas import AgentRequest, AgentStreamEvent
from src.infra.sse import iter_sse_events
from tests.web.answer_fixtures import response_payload


class _FakeAgentRequestService:
    def __init__(self) -> None:
        self.stream_calls: list[dict[str, object]] = []

    def stream(self, *, request_id: str, request_data: AgentRequest):
        self.stream_calls.append({"request_id": request_id, "request_data": request_data})

        async def event_stream():
            yield AgentStreamEvent(event="request_started", data={"request_id": request_id})
            yield AgentStreamEvent(
                event="final_response",
                data={
                    "response": response_payload("delegated answer"),
                    "trace": f"trace-{request_id}",
                    "debug": None,
                },
            )
            yield AgentStreamEvent(event="done", data={})

        return event_stream()


class AgentRouteServiceDelegationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.service = _FakeAgentRequestService()
        app = create_app()
        app.state.agent_request_service = self.service
        # Route contracts do not require the production lifespan or file cleanup.
        self.client = TestClient(app, raise_server_exceptions=False)
        self.addCleanup(self.client.close)

    def test_agent_stream_route_streams_sse_events(self) -> None:
        """The HTTP response exposes framed progress and final response events."""
        response = self.client.post("/agent/stream", json={"query": "hello", "session_id": "demo"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "text/event-stream; charset=utf-8")
        self.assertEqual(response.headers["x-accel-buffering"], "no")
        events = list(iter_sse_events([response.content]))
        self.assertEqual([event.event for event in events], ["request_started", "final_response", "done"])
        self.assertEqual(events[1].data["response"], response_payload("delegated answer"))
        self.assertEqual(len(self.service.stream_calls), 1)

    def test_openapi_describes_stream_events_and_final_response_schema(self) -> None:
        """API consumers can discover the SSE wire format and complete final payload."""
        schema = self.client.get("/openapi.json").json()
        self.assertNotIn("/agent", schema["paths"])
        response = schema["paths"]["/agent/stream"]["post"]["responses"]["200"]
        self.assertEqual(set(response["content"]), {"text/event-stream"})
        stream = response["content"]["text/event-stream"]
        self.assertEqual(stream["schema"]["type"], "string")
        self.assertEqual(set(stream["x-sse-events"]), {
            "request_started", "stage_started", "stage_completed", "heartbeat",
            "progress_snapshot", "final_response", "error", "done",
        })
        self.assertEqual(stream["x-sse-events"]["final_response"], {"$ref": "#/components/schemas/AgentResponse"})
        final = schema["components"]["schemas"]["AgentResponse"]
        self.assertEqual(set(final["properties"]), {"response", "trace", "debug", "upload_manifest"})
        self.assertIn("AgentDebugInfo", schema["components"]["schemas"])
        self.assertIn("UploadManifest", schema["components"]["schemas"])


if __name__ == "__main__":
    unittest.main()
