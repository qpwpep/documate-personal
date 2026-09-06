import unittest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.core.conversation_memory import ConversationMemoryPolicy
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
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from src.core.planner_schema import PlannerOutput, RetrievalTask

from .helpers import (
    _CapturePlannerLLM,
    _tool_payload,
)


def _official_hit(*, uri, title, excerpt, score):
    snapshot = build_snapshot(
        source_uri=uri, title=title, media_type="text/html", source_type="official",
        content=excerpt, parser="fixture", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=excerpt))
    return SearchHit(evidence=evidence, score=RetrievalScore(metric="relevance", raw=score, direction="higher"), rank=1).model_dump(mode="json")


def _response(text, *, state=None, attempt=1):
    evidence = [SearchHit.model_validate(hit).evidence for hit in get_retrieval_state(state).hit_log] if state else []
    document = text_document(text, basis="source" if evidence else "interaction", refs=[item.id for item in evidence])
    return ResponseState(result=finalize_answer(document, evidence), evidence_packet=evidence, synthesis_attempt=attempt)


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
                "response": _response("final answer", state=state),
            },
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(verbose=False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(verbose=False),
            action_postprocess_node=lambda state: {},
            memory_policy=ConversationMemoryPolicy(),
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=[]))
        self.assertEqual(summary_calls["count"], 0)
        self.assertEqual(export_answer_text(result["response"].result), "final answer")

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
                "response": _response("final answer", state=state),
            },
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(verbose=False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(verbose=False),
            action_postprocess_node=lambda state: {},
            memory_policy=ConversationMemoryPolicy(),
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=long_history))
        self.assertEqual(summary_calls["count"], 1)
        self.assertEqual(export_answer_text(result["response"].result), "final answer")

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
                "response": _response("final answer", state=state),
            },
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(verbose=False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(verbose=False),
            action_postprocess_node=lambda state: {},
            memory_policy=ConversationMemoryPolicy(high_water_turns=4, low_water_turns=3),
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=history))
        self.assertEqual(summary_calls["count"], 0)
        self.assertEqual(export_answer_text(result["response"].result), "final answer")

    def test_planner_skips_retrieval_dispatch_when_not_required(self) -> None:
        dispatch_calls = {"count": 0}

        graph = build_graph(
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=lambda state: {"planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[]))},
            retrieve_dispatch_node=lambda state: dispatch_calls.__setitem__("count", dispatch_calls["count"] + 1),
            synthesize_node=lambda state: {
                "response": _response("final answer", state=state),
            },
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(verbose=False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(verbose=False),
            action_postprocess_node=lambda state: {},
            memory_policy=ConversationMemoryPolicy(),
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=[]))
        self.assertEqual(dispatch_calls["count"], 0)
        self.assertEqual(export_answer_text(result["response"].result), "final answer")

    def test_graph_retrieves_docs_selected_by_planner(self) -> None:
        docs_calls = {"count": 0}
        capture_planner = _CapturePlannerLLM(PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="FastAPI response_model", k=4)],
        ))

        def _docs_search(query: str):
            docs_calls["count"] += 1
            return _tool_payload(
                [
                    _official_hit(uri='https://fastapi.tiangolo.com/reference/response/', title='FastAPI Response Reference', excerpt='response model docs', score=0.91)
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
                _docs_search,
                lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query),
                verbose=False,
            ),
            synthesize_node=lambda state: {
                "response": _response("final answer", state=state),
            },
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(verbose=False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(verbose=False),
            action_postprocess_node=lambda state: {},
            memory_policy=ConversationMemoryPolicy(),
        )

        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain FastAPI response_model from official docs.",
                messages=[],
            )
        )
        self.assertEqual(capture_planner.call_count, 1)
        self.assertEqual(docs_calls["count"], 1)
        self.assertTrue(
            any(message.name == "tavily_search" for message in result["messages"] if isinstance(message, ToolMessage))
        )

    def test_retry_path_reruns_docs_retrieval_and_synthesis(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="NumPy broadcasting", k=4)],
        ))
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
                    _official_hit(uri='https://numpy.org/doc/stable/', title='NumPy Docs', excerpt='official docs', score=0.92)
                ],
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query=query,
            )

        retrieve_dispatch = make_retrieve_dispatch_node(
            _docs_search,
            lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query),
            verbose=False,
        )

        def _synthesize(state):
            synth_calls["count"] += 1
            answer = f"answer-{synth_calls['count']}"
            attempt = get_response_state(state).synthesis_attempt + 1
            return {
                "response": _response(answer, state=state, attempt=attempt),
            }

        graph = build_graph(
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=planner_node,
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=_synthesize,
            pre_synthesis_validation_node=make_pre_synthesis_validation_node(verbose=False),
            post_synthesis_validation_node=make_post_synthesis_validation_node(verbose=False),
            action_postprocess_node=lambda state: {},
            memory_policy=ConversationMemoryPolicy(),
        )

        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain NumPy broadcasting from official docs.",
                messages=[],
            )
        )
        self.assertEqual(capture_planner.call_count, 2)
        self.assertEqual(docs_calls["count"], 2)
        self.assertEqual(synth_calls["count"], 1)
        self.assertEqual(export_answer_text(result["response"].result), "answer-1 [1]")

    def test_debug_survives_validation_and_action_stage_instrumentation(self) -> None:
        retrieve_dispatch = make_retrieve_dispatch_node(
            lambda query: _tool_payload(
                [
                    _official_hit(uri='https://numpy.org/doc/stable/', title='NumPy docs', excerpt='broadcasting official reference', score=0.94)
                ],
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query=query,
            ),
            lambda query, k, retriever=None: _tool_payload(
                [],
                tool="upload_search",
                route="upload",
                status="no_result",
                message="",
                query=query,
            ),
            verbose=False,
        )

        def _synthesize(state):
            debug = get_debug_state(state)
            answer = "NumPy broadcasting keeps compatible dimensions aligned"
            return {
                "response": _response(answer, state=state),
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

        pre_validate_node = _instrument_stage_node(
            "pre_synthesis_validation",
            make_pre_synthesis_validation_node(verbose=False),
        )
        validate_node = _instrument_stage_node(
            "post_synthesis_validation",
            make_post_synthesis_validation_node(verbose=False),
        )
        action_node = _instrument_stage_node(
            "action_postprocess",
            make_action_postprocess_node(
                save_text_tool=lambda content, filename_prefix: {"status": "ok"},
                slack_notify_tool=lambda text, **kwargs: {"status": "ok"},
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
            pre_synthesis_validation_node=pre_validate_node,
            post_synthesis_validation_node=validate_node,
            action_postprocess_node=action_node,
            memory_policy=ConversationMemoryPolicy(),
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
        self.assertTrue(any(item.get("stage") == "pre_synthesis_validation" for item in stage_events))
        self.assertTrue(any(item.get("stage") == "post_synthesis_validation" for item in stage_events))
        self.assertTrue(any(item.get("stage") == "action_postprocess" for item in stage_events))
        self.assertEqual(export_answer_text(result["response"].result), "NumPy broadcasting keeps compatible dimensions aligned [1]")


if __name__ == "__main__":
    unittest.main()


def test_repair_resynthesis_uses_the_same_packet_without_searching_again():
    """공식·업로드·혼합 답변의 본문 오류는 같은 packet으로 한 번만 재합성한다."""
    for routes, defect, max_retries, persistent in (
        (["docs"], "references", 1, False),
        (["upload"], "code", 1, False),
        (["docs", "upload"], "coverage", 1, False),
        (["docs"], "references", 0, True),
        (["docs"], "references", 3, True),
    ):
        _assert_repair_flow(routes, defect, max_retries, persistent)


def _assert_repair_flow(routes, defect, max_retries, persistent):
    import json

    from src.core.answer_schema import AnswerDocument
    from src.runtime.nodes.synthesis import make_synthesize_node
    from tests.core.test_answer_validation import _evidence, _hit

    hits = {
        route: _hit(_evidence(source_type="official" if route == "docs" else "upload")).model_dump(mode="json")
        for route in routes
    }
    searches = {"docs": 0, "upload": 0}
    packets = []

    def search(route, query):
        searches[route] += 1
        return _tool_payload(
            [hits[route]], tool="tavily_search" if route == "docs" else "upload_search",
            route=route, status="success", message="", query=query,
        )

    class RepairingLLM:
        def with_structured_output(self, *args, **kwargs):
            return self

        def invoke(self, messages):
            raw = next(str(message.content) for message in messages if str(message.content).startswith("[Evidence Packet]"))
            packet = json.loads(raw.split("\n", 2)[2])
            packets.append(packet)
            refs = [item["id"] for item in packet]
            broken = len(packets) == 1 or persistent
            if broken and defect == "references":
                refs = ["unknown-ref"]
            if broken and defect == "coverage":
                refs = refs[:1]
            if defect == "code" and not broken:
                document = {"blocks": [{
                    "type": "code", "language": "python",
                    "content": {"text": "retries = 5", "basis": "example", "refs": refs},
                }]}
            else:
                document = {"blocks": [{
                    "type": "paragraph",
                    "content": [{"text": "수정 전 본문" if broken else "확인한 설명", "basis": "source", "refs": refs}],
                }]}
            return AnswerDocument.model_validate(document).model_dump(mode="json")

    plan = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route=route, query=route, k=3) for route in routes])
    planner_llm = _CapturePlannerLLM(plan)
    request = "업로드 코드 예시를 설명해줘" if defect == "code" else (
        "공식 문서와 업로드 자료를 비교해줘" if len(routes) == 2 else "공식 문서를 설명해줘"
    )
    graph = build_graph(
        state_type=GraphState, add_user_node=add_user_message,
        summarize_node=lambda state: {},
        planner_node=make_planner_node(planner_llm, verbose=False),
        retrieve_dispatch_node=make_retrieve_dispatch_node(
            lambda query: search("docs", query),
            lambda query, k, retriever=None: search("upload", query),
            verbose=False,
        ),
        synthesize_node=make_synthesize_node(RepairingLLM(), verbose=False),
        pre_synthesis_validation_node=make_pre_synthesis_validation_node(False),
        post_synthesis_validation_node=make_post_synthesis_validation_node(False),
        action_postprocess_node=lambda state: {},
        memory_policy=ConversationMemoryPolicy(),
    )

    state = graph.invoke(build_graph_state_input(
        user_input=request, messages=[], retriever=object() if "upload" in routes else None,
        retry={"max_retries": max_retries},
    ))

    expected_attempts = 1 if max_retries == 0 else 2
    assert len(packets) == expected_attempts, (routes, defect, max_retries)
    assert all(packet == packets[0] for packet in packets)
    assert planner_llm.call_count == 1
    assert searches == {route: int(route in routes) for route in searches}
    result = state["response"].result
    assert all(check.reference_status != "missing" for check in result.checks)
    assert [message.content for message in state["messages"] if isinstance(message, AIMessage)] == [export_answer_text(result)]
    if persistent:
        assert any(issue.code == "source_excerpt_fallback" for issue in result.issues)
        assert "수정 전 본문" not in export_answer_text(result)
    else:
        assert not any(issue.code == "source_excerpt_fallback" for issue in result.issues)
