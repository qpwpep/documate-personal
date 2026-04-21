import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage

from src.core.contracts import PlannerState, ResponseState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.runtime.graph_builder import _instrument_stage_node, build_agent_graph
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.settings import AppSettings

from .helpers import _ToolWrapper


class _NeverCallLLM:
    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, _messages):
        raise AssertionError("LLM should not be invoked in this deterministic graph test")


def _docs_payload(query: str) -> dict:
    return {
        "evidence": [
            {
                "kind": "official",
                "tool": "tavily_search",
                "source_id": "url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                "document_id": "url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                "url_or_path": "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                "title": "NumPy concatenate",
                "snippet": "Join a sequence of arrays along an existing axis.",
                "score": 0.94,
            }
        ],
        "diagnostics": {
            "tool": "tavily_search",
            "route": "docs",
            "status": "success",
            "message": "",
            "query": query,
            "attempt": 1,
        },
    }


def _upload_payload(query: str) -> dict:
    return {
        "evidence": [
            {
                "kind": "local",
                "tool": "upload_search",
                "source_id": "path:uploads/demo/sample_pipeline.ipynb#cell=1;chunk=0;start=0;end=64",
                "document_id": "path:uploads/demo/sample_pipeline.ipynb",
                "url_or_path": "uploads/demo/sample_pipeline.ipynb",
                "snippet": "X = np.concatenate([train, test], axis=0)",
                "score": 0.88,
                "cell_id": 1,
                "chunk_id": 0,
                "start_offset": 0,
                "end_offset": 64,
            }
        ],
        "diagnostics": {
            "tool": "upload_search",
            "route": "upload",
            "status": "success",
            "message": "",
            "query": query,
            "attempt": 1,
        },
    }


class GraphBuilderDebugTest(unittest.TestCase):
    def test_partial_debug_patch_preserves_existing_debug_fields(self) -> None:
        wrapped = _instrument_stage_node(
            "validation",
            lambda _state: {
                "debug": {
                    "validation_errors": ["unsupported evidence id detected"],
                }
            },
        )
        state = build_graph_state_input(
            user_input="question",
            messages=[],
            response={"synthesis_attempt": 1},
            debug={
                "retrieval_diagnostics": [
                    {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "success",
                        "message": "",
                        "query": "question",
                        "attempt": 1,
                    }
                ],
                "latency_trace": [
                    {
                        "kind": "retrieval_route",
                        "route": "docs",
                        "tool": "tavily_search",
                        "attempt": 1,
                        "latency_ms": 18,
                        "status": "success",
                    }
                ],
            },
        )

        updates = wrapped(state)
        debug = get_debug_state(updates)

        self.assertEqual(len(debug.retrieval_diagnostics), 1)
        self.assertEqual(debug.retrieval_diagnostics[0].route, "docs")
        self.assertEqual(debug.validation_errors, ["unsupported evidence id detected"])
        self.assertTrue(any(item.get("stage") == "validation" for item in debug.latency_trace))

    @patch("src.runtime.graph_builder.make_synthesize_node")
    @patch("src.runtime.graph_builder.make_planner_node")
    @patch("src.runtime.graph_builder.build_llm_registry")
    @patch("src.runtime.graph_builder.build_tool_registry")
    def test_debug_survives_validation_and_action_postprocess(
        self,
        mock_build_tool_registry,
        mock_build_llm_registry,
        mock_make_planner_node,
        mock_make_synthesize_node,
    ) -> None:
        mock_build_llm_registry.return_value = SimpleNamespace(
            llm_summarizer=_NeverCallLLM(),
            llm_planner=_NeverCallLLM(),
            llm_synthesizer=_NeverCallLLM(),
            llm_synthesizer_compact=_NeverCallLLM(),
            verbose=False,
        )
        mock_make_planner_node.return_value = lambda _state: {
            "planner": PlannerState(
                output=PlannerOutput(
                    use_retrieval=True,
                    tasks=[
                        RetrievalTask(route="docs", query="numpy concatenate official docs", k=3),
                        RetrievalTask(route="upload", query="uploaded concat example", k=3),
                    ],
                ),
                status="deterministic",
                diagnostics={
                    "status": "deterministic",
                    "reason": None,
                    "fallback_routes": ["docs", "upload"],
                    "intent_required": True,
                    "required_routes": ["docs", "upload"],
                    "override_applied": False,
                    "override_reason": None,
                },
            )
        }
        mock_make_synthesize_node.return_value = lambda state: {
            "messages": [AIMessage(content="공식 설명과 업로드 비교를 정리했습니다.")],
            "response": ResponseState(
                final_answer="공식 설명과 업로드 비교를 정리했습니다.",
                payload={
                    "answer": "공식 설명과 업로드 비교를 정리했습니다.",
                    "claims": [
                        {
                            "text": "NumPy concatenate는 기존 축을 따라 배열 시퀀스를 결합한다.",
                            "evidence_ids": [
                                "url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html"
                            ],
                            "confidence": 0.94,
                        },
                        {
                            "text": "업로드 파일은 axis=0으로 train/test를 이어 붙이는 예시를 사용한다.",
                            "evidence_ids": [
                                "path:uploads/demo/sample_pipeline.ipynb#cell=1;chunk=0;start=0;end=64"
                            ],
                            "confidence": 0.88,
                        },
                    ],
                    "evidence": get_retrieval_state(state).evidence_log,
                    "confidence": 0.91,
                },
                synthesis_attempt=1,
            ),
        }
        mock_build_tool_registry.return_value = SimpleNamespace(
            tavily_search_tool=_ToolWrapper(lambda query: _docs_payload(query)),
            upload_search_tool=_ToolWrapper(lambda query, k, retriever=None: _upload_payload(query)),
            rag_search_tool=_ToolWrapper(
                lambda query, k: {
                    "evidence": [],
                    "diagnostics": {
                        "tool": "rag_search",
                        "route": "local",
                        "status": "no_result",
                        "message": "",
                        "query": query,
                        "attempt": 1,
                    },
                }
            ),
            save_text_tool=_ToolWrapper(lambda **_kwargs: self.fail("save_text should not run")),
            slack_notify_tool=_ToolWrapper(lambda **_kwargs: self.fail("slack_notify should not run")),
        )

        graph = build_agent_graph(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain from official docs and compare it with the uploaded file example.",
                messages=[],
                retriever=object(),
            )
        )

        debug = get_debug_state(result)
        self.assertEqual(len(debug.retrieval_diagnostics), 2)
        self.assertEqual([item.route for item in debug.retrieval_diagnostics], ["docs", "upload"])
        stage_events = [
            item for item in debug.latency_trace if isinstance(item, dict) and item.get("kind") == "stage"
        ]
        self.assertTrue(any(item.get("stage") == "validation" for item in stage_events))
        self.assertTrue(any(item.get("stage") == "action_postprocess" for item in stage_events))
        self.assertEqual(debug.planner_errors, [])
        self.assertTrue(result["response"].final_answer)


if __name__ == "__main__":
    unittest.main()
