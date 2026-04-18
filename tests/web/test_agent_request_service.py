from __future__ import annotations

import asyncio
import unittest

from src.web.agent_request_service import AgentRequestService
from src.web.schemas import AgentRequest


class _FakeCleaner:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def run_once(self, *, force: bool, current_session_id: str | None = None) -> dict[str, int | bool]:
        self.calls.append(
            {
                "force": force,
                "current_session_id": current_session_id,
            }
        )
        return {"errors": 0}


class _FakeSessionStore:
    def __init__(self, agent_answer: dict[str, object]) -> None:
        self.agent_answer = agent_answer
        self.agent_manager = object()
        self.get_calls: list[str] = []
        self.run_calls: list[dict[str, object]] = []

    def get_or_create(self, session_id: str):
        self.get_calls.append(session_id)
        return self.agent_manager

    def run_session_request(
        self,
        *,
        session_id: str,
        session_metadata,
        user_input: str,
        upload_file_path: str | None = None,
    ):
        self.run_calls.append(
            {
                "session_id": session_id,
                "session_metadata": session_metadata,
                "user_input": user_input,
                "upload_file_path": upload_file_path,
            }
        )
        return self.agent_manager, dict(self.agent_answer), 12


class AgentRequestServiceTest(unittest.TestCase):
    def test_include_debug_only_changes_debug_field(self) -> None:
        cleaner = _FakeCleaner()
        store = _FakeSessionStore(
            {
                "message": "fallback answer",
                "filepath": "output/result.txt",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "errors": [],
                    "observed_evidence": [],
                },
            }
        )
        service = AgentRequestService(runtime_cleaner=cleaner, session_store=store)

        without_debug = asyncio.run(
            service.run(
                request_id="req00001",
                request_data=AgentRequest(
                    query="hello",
                    session_id="demo-session",
                    include_debug=False,
                ),
            )
        )
        with_debug = asyncio.run(
            service.run(
                request_id="req00002",
                request_data=AgentRequest(
                    query="hello",
                    session_id="demo-session",
                    include_debug=True,
                ),
            )
        )

        self.assertEqual(without_debug.response.model_dump(), with_debug.response.model_dump())
        self.assertIsNone(without_debug.debug)
        self.assertIsNotNone(with_debug.debug)
        self.assertEqual(without_debug.response.answer, "fallback answer")
        self.assertEqual(cleaner.calls[0]["current_session_id"], "demo-session")
        self.assertEqual(store.get_calls, ["demo-session", "demo-session"])

    def test_service_builds_session_metadata_snapshot_before_dispatch(self) -> None:
        cleaner = _FakeCleaner()
        store = _FakeSessionStore(
            {
                "message": "structured answer",
                "response_payload": {
                    "answer": "structured answer",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "errors": [],
                    "observed_evidence": [],
                },
            }
        )
        service = AgentRequestService(runtime_cleaner=cleaner, session_store=store)

        result = asyncio.run(
            service.run(
                request_id="req00003",
                request_data=AgentRequest(
                    query="share this",
                    session_id="demo-session",
                    slack_channel_id="C123BENCH",
                    include_debug=False,
                ),
            )
        )

        self.assertEqual(result.response.answer, "structured answer")
        self.assertEqual(store.run_calls[0]["user_input"], "share this")
        self.assertEqual(
            store.run_calls[0]["session_metadata"].slack_destination.channel_id,
            "C123BENCH",
        )


if __name__ == "__main__":
    unittest.main()
