import unittest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.contracts import GraphState, LLMCallMetadata, PlannerState, ResponseState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.runtime.graph_builder import _instrument_stage_node
from src.runtime.make_graph import build_graph
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message
from src.runtime.nodes.validation import make_validate_evidence_node
from src.core.planner_schema import PlannerOutput, RetrievalTask

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
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=_summarize,
            planner_node=lambda state: {"planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[]))},
            retrieve_dispatch_node=lambda state: self.fail("retrieve_dispatch should not run"),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "response": ResponseState(final_answer="final answer", synthesis_attempt=1),
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=[]))
        self.assertEqual(summary_calls["count"], 0)
        self.assertEqual(result["response"].final_answer, "final answer")

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
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=_summarize,
            planner_node=lambda state: {"planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[]))},
            retrieve_dispatch_node=lambda state: self.fail("retrieve_dispatch should not run"),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "response": ResponseState(final_answer="final answer", synthesis_attempt=1),
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=long_history))
        self.assertEqual(summary_calls["count"], 1)
        self.assertEqual(result["response"].final_answer, "final answer")

    def test_tool_messages_do_not_force_summary_when_turn_count_fits_window(self) -> None:
        summary_calls = {"count": 0}
        history = [
            HumanMessage(content="user-1"),
            AIMessage(content="answer-1"),
            ToolMessage(content='{"status":"ok"}', name="tavily_search", tool_call_id="tool-1"),
            AIMessage(content="saved to output/response-1.txt"),
            HumanMessage(content="user-2"),
            AIMessage(content="answer-2"),
            ToolMessage(content='{"status":"ok"}', name="save_text", tool_call_id="tool-2"),
            AIMessage(content="saved to output/response-2.txt"),
        ]

        def _summarize(state):
            summary_calls["count"] += 1
            return state

        graph = build_graph(
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=_summarize,
            planner_node=lambda state: {"planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[]))},
            retrieve_dispatch_node=lambda state: self.fail("retrieve_dispatch should not run"),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "response": ResponseState(final_answer="final answer", synthesis_attempt=1),
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=2,
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=history))
        self.assertEqual(summary_calls["count"], 0)
        self.assertEqual(result["response"].final_answer, "final answer")

    def test_planner_skips_retrieval_dispatch_when_not_required(self) -> None:
        dispatch_calls = {"count": 0}

        graph = build_graph(
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=lambda state: {"planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[]))},
            retrieve_dispatch_node=lambda state: dispatch_calls.__setitem__("count", dispatch_calls["count"] + 1),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "response": ResponseState(final_answer="final answer", synthesis_attempt=1),
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=[]))
        self.assertEqual(dispatch_calls["count"], 0)
        self.assertEqual(result["response"].final_answer, "final answer")

    def test_graph_uses_deterministic_docs_route_before_planner_llm(self) -> None:
        docs_calls = {"count": 0}
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))

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
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=make_planner_node(capture_planner, verbose=False),
            retrieve_dispatch_node=make_retrieve_dispatch_node(
                _ToolWrapper(_docs_search),
                _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
                _ToolWrapper(lambda query, k: _tool_payload([], tool="rag_search", route="local", status="no_result", message="", query=query)),
                verbose=False,
            ),
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "response": ResponseState(final_answer="final answer", synthesis_attempt=1),
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain FastAPI response_model from official docs.",
                messages=[],
            )
        )
        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual(docs_calls["count"], 1)
        self.assertTrue(
            any(message.name == "tavily_search" for message in result["messages"] if isinstance(message, ToolMessage))
        )

    def test_retry_path_reruns_docs_retrieval_and_synthesis(self) -> None:
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
                    message="no docs yet",
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
            attempt = get_response_state(state).synthesis_attempt + 1
            if synth_calls["count"] == 1:
                return {
                    "messages": [AIMessage(content=answer)],
                    "response": ResponseState(final_answer=answer, synthesis_attempt=attempt),
                }
            return {
                "messages": [AIMessage(content=f"{answer} [1]")],
                "response": ResponseState(
                    final_answer=f"{answer} [1]",
                    payload={
                        "answer": f"{answer} [1]",
                        "claims": [
                            {
                                "text": answer,
                                "evidence_ids": ["url:https://numpy.org/doc/stable/"],
                                "confidence": 0.92,
                            }
                        ],
                        "evidence": get_retrieval_state(state).evidence_log,
                        "confidence": 0.92,
                    },
                    synthesis_attempt=attempt,
                ),
            }

        graph = build_graph(
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=planner_node,
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=_synthesize,
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain NumPy broadcasting from official docs.",
                messages=[],
            )
        )
        self.assertEqual(capture_planner.call_count, 0)
        self.assertEqual(docs_calls["count"], 2)
        self.assertEqual(synth_calls["count"], 2)
        self.assertEqual(result["response"].final_answer, "answer-2 [1]")

    def test_debug_survives_validation_and_action_stage_instrumentation(self) -> None:
        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(
                lambda query: _tool_payload(
                    [
                        {
                            "kind": "official",
                            "tool": "tavily_search",
                            "source_id": "url:https://numpy.org/doc/stable/",
                            "document_id": "url:https://numpy.org/doc/stable/",
                            "url_or_path": "https://numpy.org/doc/stable/",
                            "title": "NumPy docs",
                            "snippet": "broadcasting official reference",
                            "score": 0.94,
                        }
                    ],
                    tool="tavily_search",
                    route="docs",
                    status="success",
                    message="",
                    query=query,
                )
            ),
            _ToolWrapper(
                lambda query, k, retriever=None: _tool_payload(
                    [],
                    tool="upload_search",
                    route="upload",
                    status="no_result",
                    message="",
                    query=query,
                )
            ),
            _ToolWrapper(
                lambda query, k: _tool_payload(
                    [],
                    tool="rag_search",
                    route="local",
                    status="no_result",
                    message="",
                    query=query,
                )
            ),
            verbose=False,
        )

        def _synthesize(state):
            retrieval_state = get_retrieval_state(state)
            debug = get_debug_state(state)
            answer = "NumPy broadcasting keeps compatible dimensions aligned [1]"
            return {
                "messages": [AIMessage(content=answer)],
                "response": ResponseState(
                    final_answer=answer,
                    payload={
                        "answer": answer,
                        "claims": [
                            {
                                "text": "NumPy broadcasting keeps compatible dimensions aligned.",
                                "evidence_ids": ["url:https://numpy.org/doc/stable/"],
                                "confidence": 0.94,
                            }
                        ],
                        "evidence": retrieval_state.evidence_log,
                        "confidence": 0.94,
                    },
                    synthesis_attempt=1,
                ),
                "debug": debug.model_copy(
                    update={
                        "llm_calls": [
                            *debug.llm_calls,
                            LLMCallMetadata(
                                stage="synthesis",
                                attempt=1,
                                path="structured",
                                response_metadata={"model_name": "gpt-5-mini"},
                                usage_metadata={"input_tokens": 20, "output_tokens": 8, "total_tokens": 28},
                            ),
                        ]
                    }
                ),
            }

        validate_node = _instrument_stage_node("validation", make_validate_evidence_node(verbose=False))
        action_node = _instrument_stage_node(
            "action_postprocess",
            make_action_postprocess_node(
                save_text_tool=_ToolWrapper(lambda content, filename_prefix: {"status": "ok"}),
                slack_notify_tool=_ToolWrapper(lambda text, **kwargs: {"status": "ok"}),
                verbose=False,
            ),
        )

        graph = build_graph(
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=lambda state: {
                "planner": PlannerState(
                    output=PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="docs", query="numpy broadcasting official docs", k=3)],
                    )
                )
            },
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=_synthesize,
            validate_evidence_node=validate_node,
            action_postprocess_node=action_node,
            summary_max_turns=6,
        )

        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain NumPy broadcasting from official docs.",
                messages=[],
            )
        )

        debug = get_debug_state(result)
        self.assertEqual([item.tool for item in debug.retrieval_diagnostics], ["tavily_search"])
        self.assertEqual([item.stage for item in debug.llm_calls], ["synthesis"])
        stage_events = [item for item in debug.latency_trace if item.get("kind") == "stage"]
        self.assertTrue(any(item.get("stage") == "action_postprocess" for item in stage_events))
        self.assertEqual(result["response"].final_answer, "NumPy broadcasting keeps compatible dimensions aligned [1]")


if __name__ == "__main__":
    unittest.main()
