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

    def test_query_over_the_character_limit_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRequest(
                query="x" * (DEFAULT_QUERY_MAX_CHARS + 1),
                session_id="session",
            )




if __name__ == "__main__":
    unittest.main()
