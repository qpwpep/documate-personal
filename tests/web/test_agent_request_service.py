from __future__ import annotations

import asyncio
import unittest

import pytest
from pydantic import ValidationError

from src.app.web.agent_request_service import AgentRequestService
from src.app.web.schemas import AgentRequest, AgentResponse
from src.core.answer_schema import export_answer_text
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION, DebugPayload
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.tool_execution import ToolExecutionEvidence
from src.core.uploads import UploadManifest
from tests.web.answer_fixtures import response_payload


@pytest.mark.parametrize("recipient", [
    {"kind": "email", "value": "   "},
    {"kind": "email", "value": "not-an-email"},
    {"kind": "user", "value": "U123", "email": "other@example.com"},
    [{"kind": "user", "value": "U123"}, {"kind": "email", "value": "other@example.com"}],
])
def test_invalid_explicit_recipient_is_rejected_before_dispatch(recipient):
    with pytest.raises(ValidationError):
        AgentRequest(query="보내줘", session_id="s1", slack_recipient=recipient)


@pytest.mark.parametrize("field", ["slack_user_id", "slack_email", "slack_channel_id"])
def test_obsolete_recipient_fields_are_not_silently_treated_as_unspecified(field):
    with pytest.raises(ValidationError):
        AgentRequest.model_validate({"query": "보내줘", "session_id": "s1", field: "specified"})


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
        self.run_calls: list[dict[str, object]] = []

    def run_session_request(
        self,
        *,
        session_id: str,
        session_metadata,
        user_input: str,
        upload_file_path: str | None = None,
        progress_emitter=None,
    ):
        self.run_calls.append(
            {
                "session_id": session_id,
                "session_metadata": session_metadata,
                "user_input": user_input,
                "upload_file_path": upload_file_path,
                "progress_emitter": progress_emitter,
            }
        )
        if progress_emitter is not None:
            progress_emitter.emit_stage_started(stage="planner", attempt=1)
            progress_emitter.emit_stage_completed(
                stage="planner",
                attempt=1,
                latency_ms=10,
                status="llm",
            )
            progress_emitter.emit_progress_snapshot(
                stage="retrieval",
                summary="근거 요약: docs 1건",
                evidence_count=1,
            )
        return self.agent_manager, dict(self.agent_answer), 12, UploadManifest(epoch="session-epoch", revision=0, files=[])


async def _final_response(service: AgentRequestService, *, request_id: str, request_data: AgentRequest):
    events = [event async for event in service.stream(request_id=request_id, request_data=request_data)]
    return AgentResponse.model_validate(next(event.data for event in events if event.event == "final_response"))


@pytest.mark.parametrize("diagnostics", [None, 1, [{"status": []}], [{}], "missing", []])
def test_stream_preserves_invalid_retrieval_observations_as_critical_diagnostic_failures(diagnostics):
    """Wire normalization cannot replace malformed runtime evidence with an apparently complete empty list."""
    response = response_payload("retained answer")
    debug = DebugPayload(
        answer_provenance=AnswerProvenance(body_kind="compose", response_hash=response["content_hash"], evidence_packet=[]),
        execution_evidence=ToolExecutionEvidence(schema_version=1, request_id="diagnostic-request", status="complete", events=[]),
    ).model_dump(mode="json")
    if diagnostics == "missing":
        debug.pop("retrieval_diagnostics")
    else:
        debug["retrieval_diagnostics"] = diagnostics
    store = _FakeSessionStore({"response": response, "debug": debug})
    service = AgentRequestService(runtime_cleaner=_FakeCleaner(), session_store=store)
    result = asyncio.run(_final_response(
        service, request_id="diagnostic-request",
        request_data=AgentRequest(query="hello", session_id="s1", include_debug=True),
    ))

    assert result.response.model_dump(mode="json") == response
    assert result.debug.retrieval_diagnostics == []
    assert result.debug.execution_evidence.request_id == "diagnostic-request"
    if diagnostics == []:
        assert result.debug.observability_status == "ok"
        assert result.debug.errors == result.debug.missing_required_debug_fields == []
    else:
        assert result.debug.observability_status == "failed"
        assert "retrieval_diagnostics" in result.debug.missing_required_debug_fields
        assert "DEBUG_NORMALIZATION_FAILED" in result.debug.error_codes
        assert any("retrieval_diagnostics" in error for error in result.debug.errors)


class AgentRequestServiceTest(unittest.TestCase):
    def test_invalid_runtime_response_does_not_silently_fall_back_to_message(self) -> None:
        """A broken response contract is reported instead of discarding source metadata."""
        service = AgentRequestService(
            runtime_cleaner=_FakeCleaner(),
            session_store=_FakeSessionStore({"response": {"answer": "obsolete"}, "message": "fallback"}),
        )
        async def collect_events():
            return [event async for event in service.stream(
                request_id="bad", request_data=AgentRequest(query="hello", session_id="s1"),
            )]

        events = asyncio.run(collect_events())
        self.assertEqual([event.event for event in events][-2:], ["error", "done"])
        self.assertFalse(any(event.event == "final_response" for event in events))
        self.assertIn("validation error", events[-2].data["message"])

    def test_include_debug_only_changes_debug_field(self) -> None:
        cleaner = _FakeCleaner()
        store = _FakeSessionStore(
            {
                "response": response_payload("fallback answer"),
                "debug": {
                    "schema_version": DEBUG_SCHEMA_VERSION,
                    "route_decisions": [],
                    "memory_compactions": [],
                    "observability_status": "ok",
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "errors": [],
                    "observed_hits": [],
                },
            }
        )
        service = AgentRequestService(runtime_cleaner=cleaner, session_store=store)

        without_debug = asyncio.run(
            _final_response(service,
                request_id="req00001",
                request_data=AgentRequest(
                    query="hello",
                    session_id="demo-session",
                    include_debug=False,
                ),
            )
        )
        with_debug = asyncio.run(
            _final_response(service,
                request_id="req00002",
                request_data=AgentRequest(
                    query="hello",
                    session_id="demo-session",
                    include_debug=True,
                ),
            )
        )

        self.assertEqual(without_debug.response.model_dump(), with_debug.response.model_dump())
        self.assertEqual(without_debug.upload_manifest, UploadManifest(epoch="session-epoch", revision=0, files=[]))
        self.assertEqual(without_debug.upload_manifest, with_debug.upload_manifest)
        self.assertIsNone(without_debug.debug)
        self.assertIsNotNone(with_debug.debug)
        self.assertEqual(export_answer_text(without_debug.response), "fallback answer")
        self.assertEqual(cleaner.calls[0]["current_session_id"], "demo-session")
        self.assertIsNotNone(store.run_calls[0]["progress_emitter"])

    def test_service_builds_session_metadata_snapshot_before_dispatch(self) -> None:
        cleaner = _FakeCleaner()
        store = _FakeSessionStore(
            {
                "response": response_payload("structured answer"),
                "debug": {
                    "schema_version": DEBUG_SCHEMA_VERSION,
                    "route_decisions": [],
                    "memory_compactions": [],
                    "observability_status": "ok",
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "errors": [],
                    "observed_hits": [],
                },
            }
        )
        service = AgentRequestService(runtime_cleaner=cleaner, session_store=store)

        result = asyncio.run(
            _final_response(service,
                request_id="req00003",
                request_data=AgentRequest(
                    query="share this",
                    session_id="demo-session",
                    slack_recipient={"kind": "channel", "value": "C123BENCH"},
                    include_debug=False,
                ),
            )
        )

        self.assertEqual(export_answer_text(result.response), "structured answer")
        self.assertEqual(store.run_calls[0]["user_input"], "share this")
        self.assertEqual(
            store.run_calls[0]["session_metadata"].slack_recipient.value,
            "C123BENCH",
        )

    def test_stream_emits_progress_then_final_response_then_done(self) -> None:
        cleaner = _FakeCleaner()
        store = _FakeSessionStore(
            {
                "response": response_payload("streamed answer"),
                "debug": {
                    "schema_version": DEBUG_SCHEMA_VERSION,
                    "route_decisions": [],
                    "memory_compactions": [],
                    "observability_status": "ok",
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "errors": [],
                    "observed_hits": [],
                },
            }
        )
        service = AgentRequestService(runtime_cleaner=cleaner, session_store=store)

        async def collect_events():
            return [
                event
                async for event in service.stream(
                    request_id="reqstream",
                    request_data=AgentRequest(
                        query="hello",
                        session_id="demo-session",
                        include_debug=False,
                    ),
                )
            ]

        events = asyncio.run(collect_events())

        self.assertEqual(
            [event.event for event in events],
            [
                "request_started",
                "stage_started",
                "stage_completed",
                "progress_snapshot",
                "final_response",
                "done",
            ],
        )
        self.assertEqual(events[1].data["stage"], "planner")
        self.assertEqual(events[2].data["status"], "llm")
        self.assertEqual(events[3].data["summary"], "근거 요약: docs 1건")
        self.assertEqual(events[4].data["response"], response_payload("streamed answer"))
        self.assertIsNotNone(store.run_calls[0]["progress_emitter"])


if __name__ == "__main__":
    unittest.main()
