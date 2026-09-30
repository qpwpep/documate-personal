from __future__ import annotations

import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from pydantic import ValidationError

from src.app.agent_manager import AgentFlowManager
from src.app.web.session_store import InMemorySessionStore
from src.app.web.schemas import AgentRequest
from src.core.contracts import ResponseState
from src.core.answer_schema import ActionReceipt, AnswerResponse, export_answer_text, finalize_answer, text_document
from src.core.conversation_memory import (
    DEFAULT_QUERY_MAX_CHARS,
    DEFAULT_QUERY_MAX_UTF8_BYTES,
)
from src.infra.settings import AppSettings
from src.runtime.nodes.session import add_user_message


def _response(answer: str, *, actions: list[ActionReceipt] | None = None) -> ResponseState:
    return ResponseState(result=finalize_answer(text_document(answer), [], actions=actions), kind="answer")


class _RollingSummaryGraph:
    def __init__(self, *, include_save_receipt: bool = False) -> None:
        self.states: list[dict] = []
        self.include_save_receipt = include_save_receipt

    def stream(self, state: dict, *, stream_mode: str):
        assert stream_mode == "values"
        self.states.append(dict(state))
        turn = len(self.states)
        added = add_user_message(state)
        runtime = added["runtime"]
        answer = f"answer-{turn}"
        messages = [
            *state.get("messages", []),
            *added["messages"],
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
        yield {
            "route_decisions": [],
            "messages": messages,
            "runtime": runtime.model_copy(
                update={"memory_summary": f"summary-{turn}"}
            ),
            "response": _response(answer, actions=[ActionReceipt(kind="save_text", status="success", file_path="output/result.txt")] if self.include_save_receipt else []),
        }


class _RuntimeOmittingGraph:
    def stream(self, state: dict, *, stream_mode: str):
        runtime = state["runtime"]
        yield {
            "route_decisions": [],
            "messages": [
                *state.get("messages", []),
                HumanMessage(content=runtime.user_input),
                AIMessage(content="legacy answer"),
            ],
            "response": _response("legacy answer"),
        }


class _MutatingFailureGraph:
    def stream(self, state: dict, *, stream_mode: str):
        state["messages"][0].content = "mutated by failed graph"
        state["messages"][0].id = "mutated-id"
        raise RuntimeError("graph failed after mutating its input")


class _InvalidResponseGraph(_RollingSummaryGraph):
    def stream(self, state: dict, *, stream_mode: str):
        result = next(super().stream(state, stream_mode=stream_mode))
        # A graph boundary can return a valid-looking revision whose body changed.
        response = result["response"].model_dump(mode="json")
        response["result"]["content"]["blocks"][0]["content"][0]["text"] = "unvalidated replacement"
        result["response"] = response
        yield result


class _InvalidRuntimeGraph(_RollingSummaryGraph):
    def stream(self, state: dict, *, stream_mode: str):
        result = next(super().stream(state, stream_mode=stream_mode))
        result["runtime"] = {"previous_response": {"content": text_document("unchecked runtime body").model_dump(mode="json")}}
        yield result


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




    def test_invalid_returned_runtime_preserves_conversation_and_previous_answer(self) -> None:
        """잘못된 반환 runtime은 대화 기록 일부만 새 요청으로 교체하지 않는다."""
        manager = _make_manager(_InvalidRuntimeGraph())
        manager.messages = [HumanMessage(content="stable request"), AIMessage(content="stable answer")]
        manager.memory_summary = "stable summary"
        previous = finalize_answer(text_document("stable answer"), [])
        manager._ensure_session().previous_response = previous
        before = manager._ensure_session().snapshot_conversation_memory()

        result = manager.run_agent_flow("new request")

        self.assertEqual(result["debug"]["observability_status"], "failed")
        self.assertEqual(manager._ensure_session().snapshot_conversation_memory(), before)
        self.assertEqual(manager._ensure_session().previous_response, previous)





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
    pass




if __name__ == "__main__":
    unittest.main()
