import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.make_graph import build_graph
from src.nodes.planner import make_planner_node
from src.nodes.retrieval import make_retrieve_dispatch_node
from src.nodes.session import add_user_message
from src.nodes.state import State
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput, RetrievalTask

from .helpers import (
    _CapturePlannerLLM,
    _FailingPlannerLLM,
    _ToolWrapper,
    _tool_payload,
)


class GraphRoutingTest(unittest.TestCase):
    def test_short_conversation_skips_summary_node(self) -> None:
        summary_calls = {"count": 0}

        def _summarize(state):
            summary_calls["count"] += 1
            return state

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=_summarize,
            planner_node=lambda state: {"planner_output": PlannerOutput(use_retrieval=False, tasks=[])},
            retrieve_dispatch_node=lambda state: self.fail("retrieve_dispatch should not run"),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "final_answer": "final answer",
                "synthesis_attempt": 1,
                "needs_retry": False,
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "question", "messages": []})
        self.assertEqual(summary_calls["count"], 0)
        self.assertEqual(result.get("final_answer"), "final answer")

    def test_long_conversation_runs_summary_node(self) -> None:
        summary_calls = {"count": 0}
        long_history = [
            HumanMessage(content=f"user-{index}") if index % 2 == 0 else AIMessage(content=f"ai-{index}")
            for index in range(14)
        ]

        def _summarize(state):
            summary_calls["count"] += 1
            return state

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=_summarize,
            planner_node=lambda state: {"planner_output": PlannerOutput(use_retrieval=False, tasks=[])},
            retrieve_dispatch_node=lambda state: self.fail("retrieve_dispatch should not run"),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "final_answer": "final answer",
                "synthesis_attempt": 1,
                "needs_retry": False,
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "question", "messages": long_history})
        self.assertEqual(summary_calls["count"], 1)
        self.assertEqual(result.get("final_answer"), "final answer")

    def test_planner_skips_retrieval_dispatch_when_not_required(self) -> None:
        dispatch_calls = {"count": 0}
        synth_calls = {"count": 0}

        def _dispatch(_state):
            dispatch_calls["count"] += 1
            return {}

        def _synthesize(state):
            synth_calls["count"] += 1
            return {
                "messages": [AIMessage(content="final answer")],
                "final_answer": "final answer",
                "synthesis_attempt": int(state.get("synthesis_attempt", 0)) + 1,
                "needs_retry": False,
            }

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=lambda state: {"planner_output": PlannerOutput(use_retrieval=False, tasks=[])},
            retrieve_dispatch_node=_dispatch,
            synthesize_node=_synthesize,
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "question", "messages": []})
        self.assertEqual(dispatch_calls["count"], 0)
        self.assertEqual(synth_calls["count"], 1)
        self.assertEqual(result.get("final_answer"), "final answer")

    def test_graph_forces_retrieval_when_planner_fails_for_docs_query(self) -> None:
        docs_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            return _tool_payload(
                [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://fastapi.tiangolo.com/reference/response/",
                        "url_or_path": "https://fastapi.tiangolo.com/reference/response/",
                        "title": "FastAPI Response Reference",
                        "snippet": "response model docs",
                        "score": 0.91,
                    }
                ],
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query=query,
            )

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=make_planner_node(_FailingPlannerLLM(), verbose=False),
            retrieve_dispatch_node=make_retrieve_dispatch_node(
                _ToolWrapper(_docs_search),
                _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
                _ToolWrapper(lambda query, k: _tool_payload([], tool="rag_search", route="local", status="no_result", message="", query=query)),
                verbose=False,
            ),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "final_answer": "final answer",
                "synthesis_attempt": 1,
                "needs_retry": False,
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "공식 문서 기준으로 FastAPI response_model 설명해줘", "messages": []})
        self.assertEqual(docs_calls["count"], 1)
        self.assertTrue(any(message.name == "tavily_search" for message in result["messages"] if isinstance(message, ToolMessage)))

    def test_retry_path_replans_and_reruns_retrieval(self) -> None:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        capture_planner = _CapturePlannerLLM(planner_output)
        planner_node = make_planner_node(capture_planner, verbose=False)

        docs_calls = {"count": 0}
        synth_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            if docs_calls["count"] == 1:
                return _tool_payload(
                    [],
                    tool="tavily_search",
                    route="docs",
                    status="no_result",
                    message="no official docs found",
                    query=query,
                )
            return _tool_payload(
                [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "official docs",
                        "score": 0.92,
                    }
                ],
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query=query,
            )

        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(_docs_search),
            _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
            _ToolWrapper(lambda query, k: _tool_payload([], tool="rag_search", route="local", status="no_result", message="", query=query)),
            verbose=False,
        )

        def _synthesize(state):
            synth_calls["count"] += 1
            answer = f"answer-{synth_calls['count']}"
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
                "synthesis_attempt": int(state.get("synthesis_attempt", 0)) + 1,
                "needs_retry": False,
            }

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=planner_node,
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=_synthesize,
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "question", "messages": []})
        self.assertEqual(capture_planner.call_count, 2)
        self.assertEqual(docs_calls["count"], 2)
        self.assertEqual(synth_calls["count"], 2)
        self.assertEqual(result.get("final_answer"), "answer-2")

    def test_retry_path_keeps_required_routes_when_planner_repeats_false(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        docs_calls = {"count": 0}
        synth_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            if docs_calls["count"] == 1:
                return _tool_payload(
                    [],
                    tool="tavily_search",
                    route="docs",
                    status="no_result",
                    message="no official docs found",
                    query=query,
                )
            return _tool_payload(
                [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://fastapi.tiangolo.com/reference/response/",
                        "url_or_path": "https://fastapi.tiangolo.com/reference/response/",
                        "title": "FastAPI Response Reference",
                        "snippet": "response model docs",
                        "score": 0.93,
                    }
                ],
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query=query,
            )

        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(_docs_search),
            _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
            _ToolWrapper(lambda query, k: _tool_payload([], tool="rag_search", route="local", status="no_result", message="", query=query)),
            verbose=False,
        )

        def _synthesize(state):
            synth_calls["count"] += 1
            answer = f"answer-{synth_calls['count']}"
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
                "synthesis_attempt": int(state.get("synthesis_attempt", 0)) + 1,
                "needs_retry": False,
            }

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=planner_node,
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=_synthesize,
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "FastAPI response_model을 공식 문서 기준으로 설명해줘", "messages": []})
        self.assertEqual(capture_planner.call_count, 2)
        self.assertEqual(docs_calls["count"], 2)
        self.assertEqual(synth_calls["count"], 2)
        self.assertEqual(result.get("final_answer"), "answer-2")
        self.assertTrue(result["planner_diagnostics"]["intent_required"])
        self.assertEqual(result["planner_diagnostics"]["required_routes"], ["docs"])
