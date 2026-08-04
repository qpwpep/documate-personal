from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from src.app.agent_manager import AgentFlowManager
from src.app.web.session_store import InMemorySessionStore
from src.app.web.schemas import AgentRequest
from src.core.contracts import ResponseState
from src.core.conversation_memory import (
    DEFAULT_QUERY_MAX_CHARS,
    DEFAULT_QUERY_MAX_UTF8_BYTES,
)
from src.infra.settings import AppSettings


def _response(answer: str) -> ResponseState:
    return ResponseState(
        final_answer=answer,
        payload={
            "answer": answer,
            "claims": [],
            "evidence": [],
            "confidence": None,
        },
    )


class _RollingSummaryGraph:
    def __init__(self, *, include_save_receipt: bool = False) -> None:
        self.states: list[dict] = []
        self.include_save_receipt = include_save_receipt

    def invoke(self, state: dict) -> dict:
        self.states.append(dict(state))
        turn = len(self.states)
        runtime = state["runtime"]
        answer = f"answer-{turn}"
        messages = [
            *state.get("messages", []),
            HumanMessage(content=runtime.user_input),
            AIMessage(content=answer),
        ]
        if self.include_save_receipt:
            messages.append(
                ToolMessage(
                    content=json.dumps(
                        {
                            "status": "success",
                            "file_path": "output/result.txt",
                            "raw": "TRANSIENT_TOOL_PAYLOAD",
                        }
                    ),
                    name="save_text",
                    tool_call_id=f"save-{turn}",
                )
            )
        return {
            "messages": messages,
            "runtime": runtime.model_copy(
                update={"memory_summary": f"summary-{turn}"}
            ),
            "response": _response(answer),
        }


class _RuntimeOmittingGraph:
    def invoke(self, state: dict) -> dict:
        runtime = state["runtime"]
        return {
            "messages": [
                *state.get("messages", []),
                HumanMessage(content=runtime.user_input),
                AIMessage(content="legacy answer"),
            ],
            "response": _response("legacy answer"),
        }


class _MutatingFailureGraph:
    def invoke(self, state: dict) -> dict:
        state["messages"][0].content = "mutated by failed graph"
        state["messages"][0].id = "mutated-id"
        raise RuntimeError("graph failed after mutating its input")


def _make_manager(graph) -> AgentFlowManager:
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(
        openai_api_key="test-key",
        tavily_api_key="test-key",
    )
    manager.graph = graph
    manager.messages = []
    manager.upload_retriever_handle = None
    manager.upload_file_path = None
    return manager


class ConversationMemoryLifecycleTest(unittest.TestCase):
    def test_next_request_receives_the_summary_committed_by_the_previous_request(self) -> None:
        graph = _RollingSummaryGraph()
        manager = _make_manager(graph)

        manager.run_agent_flow("first")
        manager.run_agent_flow("second")

        self.assertEqual(
            [state["runtime"].memory_summary for state in graph.states],
            [None, "summary-1"],
        )
        self.assertEqual(manager.memory_summary, "summary-2")
        self.assertEqual(
            [str(message.content) for message in graph.states[1]["messages"]],
            ["first", "answer-1"],
        )

    def test_graph_without_runtime_preserves_the_existing_summary(self) -> None:
        manager = _make_manager(_RuntimeOmittingGraph())
        manager.memory_summary = "stable summary"

        manager.run_agent_flow("request")

        self.assertEqual(manager.memory_summary, "stable summary")

    def test_tool_payload_is_used_for_the_response_but_not_persisted(self) -> None:
        manager = _make_manager(_RollingSummaryGraph(include_save_receipt=True))

        result = manager.run_agent_flow("save this")

        self.assertTrue(
            any(
                isinstance(message, ToolMessage)
                for message in result["response"]["messages"]
            )
        )
        self.assertIn("저장 완료:", result["message"])
        self.assertTrue(result["filepath"])
        self.assertFalse(any(isinstance(message, ToolMessage) for message in manager.messages))
        self.assertEqual(
            [(type(message), str(message.content)) for message in manager.messages],
            [
                (HumanMessage, "save this"),
                (AIMessage, result["message"]),
            ],
        )

    def test_response_assembly_failure_preserves_the_previous_snapshot(self) -> None:
        manager = _make_manager(_RollingSummaryGraph(include_save_receipt=True))
        manager.messages = [
            HumanMessage(content="stable request"),
            AIMessage(content="stable answer"),
        ]
        manager.memory_summary = "stable summary"
        before = manager._ensure_session().snapshot_conversation_memory()

        with patch(
            "src.runtime.agent_runtime.response_assembler.Path.resolve",
            side_effect=OSError("filesystem unavailable"),
        ):
            result = manager.run_agent_flow("new request")

        self.assertEqual(
            manager._ensure_session().snapshot_conversation_memory(),
            before,
        )
        self.assertEqual(result["debug"]["observability_status"], "failed")
        self.assertIn("filesystem unavailable", result["message"])

    def test_failed_graph_cannot_mutate_the_previous_snapshot_through_shared_messages(self) -> None:
        manager = _make_manager(_MutatingFailureGraph())
        manager.messages = [
            HumanMessage(content="stable request"),
            AIMessage(content="stable answer"),
        ]
        manager.memory_summary = "stable summary"

        result = manager.run_agent_flow("new request")

        self.assertEqual(
            [(message.content, message.id) for message in manager.messages],
            [("stable request", None), ("stable answer", None)],
        )
        self.assertEqual(manager.memory_summary, "stable summary")
        self.assertEqual(result["debug"]["observability_status"], "failed")

    def test_exported_messages_cannot_mutate_the_owned_session_snapshot(self) -> None:
        manager = _make_manager(_RuntimeOmittingGraph())
        manager.messages = [HumanMessage(content="owned by session")]

        exported = manager.messages
        exported[0].content = "mutated by caller"

        self.assertEqual(manager.messages[0].content, "owned by session")

    def test_close_discards_messages_and_summary_together(self) -> None:
        manager = _make_manager(_RuntimeOmittingGraph())
        manager.messages = [
            HumanMessage(content="secret request"),
            AIMessage(content="secret answer"),
        ]
        manager.memory_summary = "secret summary"

        manager.close()

        snapshot = manager._ensure_session().snapshot_conversation_memory()
        self.assertEqual(snapshot.messages, ())
        self.assertIsNone(snapshot.memory_summary)

    def test_expired_session_discards_its_conversation_memory(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test-key")
        store = InMemorySessionStore(
            settings=settings,
            agent_factory=lambda: _make_manager(_RuntimeOmittingGraph()),
        )
        manager = store.get_or_create("stale-session")
        manager.messages = [
            HumanMessage(content="old request"),
            AIMessage(content="old answer"),
        ]
        manager.memory_summary = "old summary"
        last_accessed = store.active_agents["stale-session"].last_accessed_monotonic

        removed = store.cleanup_expired(now=last_accessed + 2, ttl_seconds=1)

        self.assertEqual(removed, 1)
        self.assertNotIn("stale-session", store.active_session_ids())
        self.assertEqual(manager.messages, [])
        self.assertIsNone(manager.memory_summary)

    def test_different_sessions_do_not_share_conversation_memory(self) -> None:
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test-key")
        graphs = iter([_RollingSummaryGraph(), _RollingSummaryGraph()])
        store = InMemorySessionStore(
            settings=settings,
            agent_factory=lambda: _make_manager(next(graphs)),
        )
        manager_a = store.get_or_create("session-a")
        manager_b = store.get_or_create("session-b")

        manager_a.run_agent_flow("alpha")
        manager_b.run_agent_flow("beta")
        manager_a.memory_summary = "session-a-only"

        self.assertEqual(manager_a.memory_summary, "session-a-only")
        self.assertEqual(manager_b.memory_summary, "summary-1")
        self.assertEqual(
            [str(message.content) for message in manager_b.messages],
            ["beta", "answer-1"],
        )


class AgentRequestMemoryBoundaryTest(unittest.TestCase):
    def test_query_at_the_character_limit_is_accepted(self) -> None:
        request = AgentRequest(
            query="x" * DEFAULT_QUERY_MAX_CHARS,
            session_id="session",
        )

        self.assertEqual(len(request.query), DEFAULT_QUERY_MAX_CHARS)

    def test_query_over_the_character_limit_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(
                query="x" * (DEFAULT_QUERY_MAX_CHARS + 1),
                session_id="session",
            )

    def test_query_over_the_utf8_byte_limit_is_rejected(self) -> None:
        query = "😀" * (DEFAULT_QUERY_MAX_UTF8_BYTES // 4 + 1)
        self.assertLessEqual(len(query), DEFAULT_QUERY_MAX_CHARS)

        with self.assertRaises(ValidationError):
            AgentRequest(query=query, session_id="session")

    def test_blank_query_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(query="   ", session_id="session")


if __name__ == "__main__":
    unittest.main()
