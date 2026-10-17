from __future__ import annotations

import unittest
from copy import deepcopy

from langchain_core.messages import AIMessage

from src.app.agent_manager import AgentFlowManager
from src.app.web.session_store import InMemorySessionStore
from src.core.contracts import ResponseState
from src.core.answer_schema import AnswerResponse, finalize_answer, text_document
from src.infra.settings import AppSettings
from src.runtime.nodes.session import add_user_message


def _response(answer: str) -> ResponseState:
    return ResponseState(result=finalize_answer(text_document(answer), []), kind="answer")


class _RollingSummaryGraph:
    def __init__(self) -> None:
        self.states: list[dict] = []

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
        yield {
            "route_decisions": [],
            "messages": messages,
            "runtime": runtime.model_copy(
                update={"memory_summary": f"summary-{turn}"}
            ),
            "response": _response(answer),
        }


class _InvalidRuntimeGraph(_RollingSummaryGraph):
    def stream(self, state: dict, *, stream_mode: str):
        result = next(super().stream(state, stream_mode=stream_mode))
        result["runtime"] = {"previous_response": {"content": text_document("unchecked runtime body").model_dump(mode="json")}}
        yield result


def _make_manager(graph) -> AgentFlowManager:
    manager = AgentFlowManager(settings=AppSettings(
        _env_file=None,
        openai_api_key="test-key",
        tavily_api_key="test-key",
    ))
    manager.graph = graph
    return manager


class ConversationMemoryLifecycleTest(unittest.TestCase):
    def test_invalid_returned_runtime_preserves_conversation_and_previous_answer(self) -> None:
        """잘못된 반환 runtime은 대화 기록 일부만 새 요청으로 교체하지 않는다."""
        graph = _RollingSummaryGraph()
        manager = _make_manager(graph)
        previous = AnswerResponse.model_validate(manager.run_agent_flow("stable request")["response"])
        before = deepcopy(manager.messages)
        summary = manager.memory_summary
        manager.graph = _InvalidRuntimeGraph()

        result = manager.run_agent_flow("new request")

        self.assertEqual(result["debug"]["observability_status"], "failed")
        self.assertEqual(manager.messages, before)
        self.assertEqual(manager.memory_summary, summary)
        manager.graph = graph
        manager.run_agent_flow("continue")
        self.assertEqual(graph.states[1]["runtime"].previous_response, previous)
        self.assertEqual(graph.states[1]["messages"], before)
        self.assertEqual([turn.text for turn in graph.states[1]["runtime"].user_turns], ["stable request"])


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


if __name__ == "__main__":
    unittest.main()
