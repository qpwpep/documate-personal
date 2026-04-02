import json
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent_manager import AgentFlowManager
from src.contracts.graph_state import DebugState, PlannerState, ResponseState, RetrievalState
from src.graph_builder import StageExecutionError
from src.settings import AppSettings
from src.tools import build_tool_registry
from src.tools.docs_search import infer_docs_query_hint


class _FakeGraph:
    def __init__(self, evidence_payload: list[dict]):
        self._evidence_payload = evidence_payload

    def invoke(self, state: dict) -> dict:
        runtime = state["runtime"]
        query = runtime.user_input
        return {
            "messages": [
                HumanMessage(content=query),
                ToolMessage(
                    content=json.dumps(
                        {
                            "evidence": self._evidence_payload,
                            "diagnostics": {
                                "tool": "tavily_search",
                                "route": "docs",
                                "status": "success",
                                "message": "",
                                "query": query,
                                "attempt": 1,
                            },
                        },
                        ensure_ascii=False,
                    ),
                    name="tavily_search",
                    tool_call_id="call-1",
                ),
                AIMessage(content="final answer [1]"),
            ],
            "retrieval": RetrievalState(evidence_log=self._evidence_payload),
            "response": ResponseState(
                final_answer="final answer [1]",
                payload={
                    "answer": "final answer [1]",
                    "claims": [
                        {
                            "text": "final answer",
                            "evidence_ids": ["url:https://numpy.org/doc/stable/"],
                            "confidence": 0.88,
                        }
                    ],
                    "evidence": self._evidence_payload,
                    "confidence": 0.88,
                },
            ),
            "planner": PlannerState(
                diagnostics={
                    "status": "heuristic_fallback",
                    "reason": "planner_failed_or_invalid",
                    "fallback_routes": ["docs"],
                    "intent_required": True,
                    "required_routes": ["docs"],
                    "override_applied": False,
                    "override_reason": None,
                }
            ),
            "debug": DebugState(
                retrieval_diagnostics=[
                    {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "success",
                        "message": "",
                        "query": query,
                        "attempt": 1,
                    }
                ],
                latency_trace=[
                    {"kind": "stage", "stage": "planner", "attempt": 1, "latency_ms": 12, "status": "heuristic_fallback"},
                    {"kind": "retrieval_route", "route": "docs", "tool": "tavily_search", "attempt": 1, "latency_ms": 48, "status": "success"},
                    {"kind": "stage", "stage": "retrieval", "attempt": 1, "latency_ms": 50, "status": "success"},
                    {"kind": "synthesis_attempt", "attempt": 1, "mode": "structured_only", "structured_ms": 22, "fallback_ms": None, "total_ms": 22},
                    {"kind": "stage", "stage": "synthesis", "attempt": 1, "latency_ms": 22, "status": "structured_only"},
                    {"kind": "stage", "stage": "validation", "attempt": 1, "latency_ms": 3, "status": "pass"},
                ],
            ),
        }


class _FakeGraphWithLlmCalls:
    def invoke(self, state: dict) -> dict:
        _ = state
        return {
            "messages": [
                HumanMessage(content="question"),
                ToolMessage(content=json.dumps({"evidence": [], "diagnostics": {}}, ensure_ascii=False), name="tavily_search", tool_call_id="call-1"),
                AIMessage(content="final answer"),
            ],
            "response": ResponseState(
                final_answer="final answer",
                payload={"answer": "final answer", "claims": [], "evidence": [], "confidence": None},
            ),
            "debug": DebugState(
                llm_calls=[
                    {
                        "stage": "planner",
                        "attempt": 1,
                        "path": "structured",
                        "response_metadata": {"model_name": "gpt-5-nano"},
                        "usage_metadata": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15},
                    },
                    {
                        "stage": "synthesis",
                        "attempt": 1,
                        "path": "structured",
                        "response_metadata": {"model_name": "gpt-5-mini"},
                        "usage_metadata": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25},
                    },
                ]
            ),
        }


class _FakeGraphWithAiMessageMetadata:
    def invoke(self, state: dict) -> dict:
        _ = state
        return {
            "messages": [
                HumanMessage(content="question"),
                AIMessage(
                    content="final answer",
                    response_metadata={"model_name": "gpt-5-mini"},
                    usage_metadata={"input_tokens": 14, "output_tokens": 6, "total_tokens": 20},
                ),
            ],
            "response": ResponseState(
                final_answer="final answer",
                payload={"answer": "final answer", "claims": [], "evidence": [], "confidence": None},
                synthesis_attempt=1,
            ),
        }


class _FakeGraphWithSave:
    def invoke(self, state: dict) -> dict:
        _ = state
        return {
            "messages": [
                HumanMessage(content="question"),
                AIMessage(content="final answer before save"),
                ToolMessage(
                    content=json.dumps(
                        {
                            "message": "Saved output to response_20260101_010101.txt",
                            "file_path": "output/save_text/response_20260101_010101.txt",
                        },
                        ensure_ascii=False,
                    ),
                    name="save_text",
                    tool_call_id="save-1",
                ),
            ],
            "response": ResponseState(
                final_answer="final answer before save",
                payload={
                    "answer": "final answer before save",
                    "claims": [],
                    "evidence": [],
                    "confidence": None,
                },
            ),
        }


class _FakeVectorStore:
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        _ = (query, k)
        return [
            (
                Document(
                    page_content="uploaded snippet",
                    metadata={
                        "source": "uploads/session/sample_pipeline.ipynb",
                        "cell_id": 2,
                        "chunk_id": 1,
                        "start_offset": 12,
                        "end_offset": 28,
                    },
                ),
                0.87,
            )
        ]


class _FakeDedupVectorStore:
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        _ = (query, k)
        return [
            (
                Document(
                    page_content="first chunk",
                    metadata={
                        "source": "uploads/session/sample_pipeline.ipynb",
                        "cell_id": 0,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 10,
                    },
                ),
                0.81,
            ),
            (
                Document(
                    page_content="second chunk",
                    metadata={
                        "source": "uploads/session/sample_pipeline.ipynb",
                        "cell_id": 0,
                        "chunk_id": 1,
                        "start_offset": 10,
                        "end_offset": 22,
                    },
                ),
                0.79,
            ),
        ]


class _FakeNegativeScoreVectorStore:
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        _ = (query, k)
        return [
            (
                Document(
                    page_content="negative score snippet",
                    metadata={
                        "source": "uploads/session/sample_pipeline.ipynb",
                        "cell_id": 1,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 20,
                    },
                ),
                -0.24,
            )
        ]


class _FakeRetriever:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore or _FakeVectorStore()


class _FakeRetrieverHandle:
    def __init__(self, retriever=None, collection_name: str = "upload-session-session"):
        self.retriever = retriever or _FakeRetriever()
        self.collection_name = collection_name
        self.cleanup_calls = 0

    def cleanup(self) -> None:
        self.cleanup_calls += 1


class _FailingGraph:
    def invoke(self, state: dict) -> dict:
        _ = state
        raise StageExecutionError(stage="synthesis", latency_ms=17, cause=RuntimeError("boom"))


class EvidencePipelineTest(unittest.TestCase):
    def test_extract_observed_evidence_uses_tool_native_payloads(self) -> None:
        tavily_item = {
            "kind": "official",
            "tool": "tavily_search",
            "source_id": "url:https://numpy.org/doc/stable/",
            "document_id": "url:https://numpy.org/doc/stable/",
            "url_or_path": "https://numpy.org/doc/stable/",
            "title": "NumPy docs",
            "snippet": "broadcasting",
            "score": 0.98,
        }
        rag_item = {
            "kind": "local",
            "tool": "rag_search",
            "source_id": "path:uploads/s1/sample.ipynb#cell=0;chunk=1;start=0;end=16",
            "document_id": "path:uploads/s1/sample.ipynb",
            "url_or_path": "uploads/s1/sample.ipynb",
            "title": None,
            "snippet": "local snippet",
            "score": 0.71,
            "chunk_id": 1,
            "cell_id": 0,
            "start_offset": 0,
            "end_offset": 16,
        }

        messages = [
            ToolMessage(
                content=json.dumps(
                    {
                        "evidence": [tavily_item, tavily_item],
                        "diagnostics": {
                            "tool": "tavily_search",
                            "route": "docs",
                            "status": "success",
                            "message": "",
                            "query": "numpy",
                            "attempt": 1,
                        },
                    }
                ),
                name="tavily_search",
                tool_call_id="1",
            ),
            ToolMessage(
                content=json.dumps(
                    {
                        "evidence": [rag_item],
                        "diagnostics": {
                            "tool": "rag_search",
                            "route": "local",
                            "status": "success",
                            "message": "",
                            "query": "local",
                            "attempt": 1,
                        },
                    }
                ),
                name="rag_search",
                tool_call_id="2",
            ),
            ToolMessage(content="not-json", name="upload_search", tool_call_id="3"),
            ToolMessage(content=json.dumps([rag_item]), name="save_text", tool_call_id="4"),
        ]
        errors: list[str] = []

        observed = AgentFlowManager._extract_observed_evidence(messages, errors=errors)

        self.assertEqual(len(observed), 2)
        self.assertTrue(any(item["tool"] == "tavily_search" for item in observed))
        self.assertTrue(any(item["tool"] == "rag_search" for item in observed))
        self.assertTrue(any("tool:upload_search" in error for error in errors))

    def test_response_payload_uses_adopted_evidence_not_observed_evidence_identity(self) -> None:
        evidence_payload = [
            {
                "kind": "official",
                "tool": "tavily_search",
                "source_id": "url:https://numpy.org/doc/stable/",
                "document_id": "url:https://numpy.org/doc/stable/",
                "url_or_path": "https://numpy.org/doc/stable/",
                "title": "NumPy docs",
                "snippet": "broadcasting",
                "score": 0.99,
            }
        ]

        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = None
        manager.graph = _FakeGraph(evidence_payload)
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        self.assertIsNot(result["response_payload"]["evidence"], result["debug"]["observed_evidence"])
        self.assertEqual(result["response_payload"]["answer"], "final answer [1]")
        self.assertEqual(result["response_payload"]["claims"][0]["evidence_ids"], ["url:https://numpy.org/doc/stable/"])
        self.assertEqual(result["response_payload"]["evidence"][0]["url_or_path"], "https://numpy.org/doc/stable/")

    def test_upload_search_returns_typed_chunk_evidence_and_handles_missing_retriever(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        no_retriever = registry.upload_search_tool.func(query="uploaded info", k=3, retriever=None)
        self.assertEqual(no_retriever["diagnostics"]["status"], "unavailable")
        self.assertEqual(no_retriever["evidence"], [])

        with_retriever = registry.upload_search_tool.func(
            query="uploaded info",
            k=3,
            retriever=_FakeRetriever(),
        )
        evidence = with_retriever["evidence"]
        self.assertEqual(with_retriever["diagnostics"]["status"], "success")
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["kind"], "local")
        self.assertEqual(evidence[0]["tool"], "upload_search")
        self.assertEqual(evidence[0]["url_or_path"], "uploads/session/sample_pipeline.ipynb")
        self.assertEqual(evidence[0]["document_id"], "path:uploads/session/sample_pipeline.ipynb")
        self.assertEqual(
            evidence[0]["source_id"],
            "path:uploads/session/sample_pipeline.ipynb#cell=2;chunk=1;start=12;end=28",
        )
        self.assertEqual(evidence[0]["cell_id"], 2)
        self.assertEqual(evidence[0]["chunk_id"], 1)
        self.assertEqual(evidence[0]["start_offset"], 12)
        self.assertEqual(evidence[0]["end_offset"], 28)
        self.assertAlmostEqual(evidence[0]["score"], 0.87)

    def test_upload_search_keeps_multiple_chunks_from_same_document(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.upload_search_tool.func(
            query="uploaded info",
            k=4,
            retriever=_FakeRetriever(vectorstore=_FakeDedupVectorStore()),
        )

        evidence = result["evidence"]
        self.assertEqual(len(evidence), 2)
        self.assertNotEqual(evidence[0]["source_id"], evidence[1]["source_id"])
        self.assertEqual(
            [item["chunk_id"] for item in evidence],
            [0, 1],
        )

    def test_upload_search_clamps_negative_scores(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.upload_search_tool.func(
            query="uploaded info",
            k=4,
            retriever=_FakeRetriever(vectorstore=_FakeNegativeScoreVectorStore()),
        )

        evidence = result["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["score"], 0.0)

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_filters_to_allowed_doc_prefixes(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://fastapi.tiangolo.com/ko/tutorial/response-model",
                    "title": "FastAPI tutorial",
                    "content": "response_model docs",
                    "score": 0.91,
                },
                {
                    "url": "https://huggingface.co/docs/transformers/index",
                    "title": "HF docs",
                    "content": "docs page",
                    "score": 0.8,
                },
                {
                    "url": "https://huggingface.co/datasets/foo/bar",
                    "title": "HF dataset",
                    "content": "dataset page",
                    "score": 0.95,
                },
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="official docs")

        urls = [item["url_or_path"] for item in result["evidence"]]
        self.assertIn("https://fastapi.tiangolo.com/ko/tutorial/response-model", urls)
        self.assertIn("https://huggingface.co/docs/transformers/index", urls)
        self.assertNotIn("https://huggingface.co/datasets/foo/bar", urls)

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_returns_no_result_when_all_urls_are_filtered(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://huggingface.co/datasets/foo/bar",
                    "title": "HF dataset",
                    "content": "dataset page",
                    "score": 0.95,
                }
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="official docs")

        self.assertEqual(result["diagnostics"]["status"], "no_result")
        self.assertEqual(result["evidence"], [])

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_blocks_huggingface_commit_diff_urls(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://huggingface.co/user/repo/commit/abc123.diff?file=tokenizer.json",
                    "title": "HF commit diff",
                    "content": "diff page",
                    "score": 0.99,
                },
                {
                    "url": "https://huggingface.co/docs/transformers/index",
                    "title": "HF docs",
                    "content": "docs page",
                    "score": 0.8,
                },
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="official docs")

        urls = [item["url_or_path"] for item in result["evidence"]]
        self.assertEqual(urls, ["https://huggingface.co/docs/transformers/index"])

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_applies_symbol_based_query_hint(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        registry.tavily_search_tool.func(query="train_test_split 공식 문법을")

        first_call = mock_request_tavily_search.call_args_list[0]
        _, kwargs = first_call
        self.assertEqual(kwargs["include_domains"], ["scikit-learn.org"])
        self.assertEqual(kwargs["query"], "train_test_split 공식 문법을 scikit-learn")
        self.assertEqual(mock_request_tavily_search.call_args_list[1].kwargs["query"], "train_test_split sklearn.model_selection")

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_applies_bare_library_level_query_hint(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        registry.tavily_search_tool.func(query="bare 공식 문서")

        first_call = mock_request_tavily_search.call_args_list[0]
        _, kwargs = first_call
        self.assertEqual(kwargs["include_domains"], ["docs.pears.com"])
        self.assertEqual(kwargs["query"], "bare 공식 문서")
        self.assertEqual(mock_request_tavily_search.call_args_list[1].kwargs["query"], "Bare runtime API")

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_applies_library_level_query_hints_for_common_libraries(
        self,
        mock_request_tavily_search,
    ) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        cases = [
            ("numpy", ["numpy.org"], "numpy user guide"),
            ("pandas", ["pandas.pydata.org"], "pandas user guide"),
            ("fastapi", ["fastapi.tiangolo.com"], "fastapi tutorial"),
            ("numpy 공식 문서", ["numpy.org"], "numpy user guide"),
            ("pandas 공식 문서", ["pandas.pydata.org"], "pandas user guide"),
            ("fastapi 공식 문서", ["fastapi.tiangolo.com"], "fastapi tutorial"),
        ]

        for query, expected_domains, fallback_query in cases:
            with self.subTest(query=query):
                mock_request_tavily_search.reset_mock()
                mock_request_tavily_search.return_value = {"results": []}

                registry.tavily_search_tool.func(query=query)

                first_call = mock_request_tavily_search.call_args_list[0]
                _, kwargs = first_call
                self.assertEqual(kwargs["include_domains"], expected_domains)
                self.assertEqual(kwargs["query"], query)
                self.assertEqual(mock_request_tavily_search.call_args_list[1].kwargs["query"], fallback_query)

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_filters_cross_library_docs_results_for_hinted_queries(
        self,
        mock_request_tavily_search,
    ) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    "title": "numpy.concatenate",
                    "content": "Join a sequence of arrays along an existing axis.",
                    "score": 0.92,
                },
                {
                    "url": "https://fastapi.tiangolo.com/tutorial/response-model/",
                    "title": "FastAPI response model",
                    "content": "FastAPI docs page",
                    "score": 0.95,
                },
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="numpy 공식 문서")

        urls = [item["url_or_path"] for item in result["evidence"]]
        self.assertEqual(
            urls,
            ["https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html"],
        )
        self.assertIn("cross_library_domain_filtered", result["diagnostics"]["warnings"])

    def test_docs_search_word_match_hint_does_not_match_substring(self) -> None:
        self.assertIsNone(infer_docs_query_hint("baremetal 공식 문서"))

    def test_agent_manager_exposes_retrieval_and_planner_diagnostics(self) -> None:
        evidence_payload = [
            {
                "kind": "official",
                "tool": "tavily_search",
                "source_id": "url:https://numpy.org/doc/stable/",
                "document_id": "url:https://numpy.org/doc/stable/",
                "url_or_path": "https://numpy.org/doc/stable/",
                "title": "NumPy docs",
                "snippet": "broadcasting",
                "score": 0.99,
            }
        ]

        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = None
        manager.graph = _FakeGraph(evidence_payload)
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        self.assertEqual(result["debug"]["retrieval_diagnostics"][0]["status"], "success")
        self.assertEqual(result["debug"]["planner_diagnostics"]["status"], "heuristic_fallback")
        self.assertTrue(result["debug"]["planner_diagnostics"]["intent_required"])
        self.assertEqual(result["debug"]["planner_diagnostics"]["required_routes"], ["docs"])

    def test_agent_manager_exposes_latency_breakdown(self) -> None:
        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = AppSettings(openai_api_key="test", tavily_api_key="test")
        manager.graph = _FakeGraph([])
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        latency_breakdown = result["debug"]["latency_breakdown"]
        self.assertIsNotNone(latency_breakdown)
        self.assertGreaterEqual(latency_breakdown["graph_total_ms"], 0)
        self.assertEqual(latency_breakdown["stage_totals_ms"]["planner_ms"], 12)
        self.assertEqual(latency_breakdown["retrieval_routes"][0]["route"], "docs")
        self.assertEqual(latency_breakdown["synthesis_attempts"][0]["mode"], "structured_only")

    def test_agent_manager_exception_path_still_returns_latency_breakdown(self) -> None:
        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = AppSettings(openai_api_key="test", tavily_api_key="test")
        manager.graph = _FailingGraph()
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        latency_breakdown = result["debug"]["latency_breakdown"]
        self.assertIsNotNone(latency_breakdown)
        self.assertGreaterEqual(latency_breakdown["server_total_ms"], 0)
        self.assertEqual(latency_breakdown["stage_attempts"][0]["stage"], "synthesis")
        self.assertEqual(latency_breakdown["stage_attempts"][0]["status"], "error")

    def test_agent_manager_aggregates_llm_calls_into_debug_metadata(self) -> None:
        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = AppSettings(openai_api_key="test", tavily_api_key="test")
        manager.graph = _FakeGraphWithLlmCalls()
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        self.assertEqual(result["debug"]["token_usage"]["prompt_tokens"], 32)
        self.assertEqual(result["debug"]["token_usage"]["completion_tokens"], 8)
        self.assertEqual(result["debug"]["token_usage"]["total_tokens"], 40)
        self.assertEqual(result["debug"]["model_name"], "gpt-5-mini")
        self.assertEqual(result["debug"]["models_used"], ["gpt-5-nano", "gpt-5-mini"])
        self.assertEqual(len(result["debug"]["llm_calls"]), 2)

    def test_agent_manager_falls_back_to_current_turn_ai_message_metadata(self) -> None:
        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = AppSettings(openai_api_key="test", tavily_api_key="test")
        manager.graph = _FakeGraphWithAiMessageMetadata()
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        self.assertEqual(result["debug"]["token_usage"]["prompt_tokens"], 14)
        self.assertEqual(result["debug"]["token_usage"]["completion_tokens"], 6)
        self.assertEqual(result["debug"]["token_usage"]["total_tokens"], 20)
        self.assertEqual(result["debug"]["model_name"], "gpt-5-mini")
        self.assertEqual(result["debug"]["models_used"], ["gpt-5-mini"])
        self.assertEqual(len(result["debug"]["llm_calls"]), 1)
        self.assertEqual(result["debug"]["llm_calls"][0]["path"], "direct")

    @patch("src.agent_manager.build_temp_retriever")
    def test_agent_manager_passes_api_key_to_temp_retriever(self, mock_build_temp_retriever) -> None:
        mock_build_temp_retriever.return_value = _FakeRetrieverHandle()

        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        manager.graph = _FakeGraph([])
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question", upload_file_path="uploads/session/sample_pipeline.ipynb")

        _, kwargs = mock_build_temp_retriever.call_args
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertIsNotNone(result["debug"]["latency_breakdown"]["upload_retriever_build_ms"])

    def test_save_tool_message_does_not_override_final_answer(self) -> None:
        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = None
        manager.graph = _FakeGraphWithSave()
        manager.messages = []
        manager.upload_retriever_handle = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("save this")

        self.assertEqual(result["message"], "final answer before save")
        self.assertEqual(result["response_payload"]["answer"], "final answer before save")
        self.assertEqual(result["response_payload"]["claims"], [])

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_applies_bare_library_hints_for_numpy_pandas_fastapi(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {"results": []}

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        registry.tavily_search_tool.func(query="numpy official docs")
        registry.tavily_search_tool.func(query="pandas official docs")
        registry.tavily_search_tool.func(query="fastapi official docs")

        first_query = mock_request_tavily_search.call_args_list[0].kwargs
        fourth_query = mock_request_tavily_search.call_args_list[3].kwargs
        seventh_query = mock_request_tavily_search.call_args_list[6].kwargs
        self.assertEqual(first_query["include_domains"], ["numpy.org"])
        self.assertEqual(first_query["query"], "numpy official docs")
        self.assertEqual(fourth_query["include_domains"], ["pandas.pydata.org"])
        self.assertEqual(fourth_query["query"], "pandas official docs")
        self.assertEqual(seventh_query["include_domains"], ["fastapi.tiangolo.com"])
        self.assertEqual(seventh_query["query"], "fastapi official docs")

    @patch("src.tools.docs_search.request_tavily_search")
    def test_docs_search_post_filters_cross_library_domains_for_hinted_queries(self, mock_request_tavily_search) -> None:
        mock_request_tavily_search.return_value = {
            "results": [
                {
                    "url": "https://pandas.pydata.org/docs/reference/api/pandas.concat.html",
                    "title": "pandas.concat",
                    "content": "Concatenate pandas objects.",
                    "score": 0.92,
                },
                {
                    "url": "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    "title": "numpy.concatenate",
                    "content": "Join a sequence of arrays.",
                    "score": 0.95,
                },
            ]
        }

        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.tavily_search_tool.func(query="pandas official docs")

        self.assertEqual(
            [item["url_or_path"] for item in result["evidence"]],
            ["https://pandas.pydata.org/docs/reference/api/pandas.concat.html"],
        )

    def test_docs_search_word_match_hints_do_not_match_substrings_for_library_names(self) -> None:
        self.assertIsNone(infer_docs_query_hint("numpydoc official docs"))
        self.assertIsNone(infer_docs_query_hint("fastapiusers official docs"))
        self.assertIsNone(infer_docs_query_hint("pandasai official docs"))


if __name__ == "__main__":
    unittest.main()
