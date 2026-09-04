import json
import math
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

from unittest.mock import patch

import httpx
import requests

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from src.app.agent_manager import AgentFlowManager
from src.core.contracts import GraphState
from src.core.contracts.graph_state import DebugState, PlannerState, ResponseState, RetrievalState
from src.runtime.agent_runtime import DebugCollector, ExecutionRunner, ResponseAssembler, SessionContext
from src.runtime.graph_builder import _instrument_stage_node
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.infra.tools.docs_search import infer_docs_query_hint
from src.infra.tools.docs_search.url_validation import validate_doc_url


def _evidence_response(evidence_payload: list[dict]) -> dict:
    query = "question"
    return {
        "messages": [
            HumanMessage(content=query),
            ToolMessage(
                content=json.dumps(
                    {
                        "evidence": evidence_payload,
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
        "retrieval": RetrievalState(evidence_log=evidence_payload),
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
                "evidence": evidence_payload,
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


def _response_with_llm_calls() -> dict:
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


def _response_with_ai_metadata() -> dict:
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


def _response_with_save_receipt() -> dict:
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


def _assemble_response(response: dict) -> dict:
    debug = DebugCollector().build(
        response=response,
        updated_messages=response["messages"],
        graph_total_ms=100,
        upload_retriever_build_ms=None,
    )
    return ResponseAssembler().assemble(
        response=response, updated_messages=response["messages"], debug_info=debug,
    )


class _FakeVectorStore:
    def similarity_search_with_score(self, query: str, k: int = 4):
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
    def similarity_search_with_score(self, query: str, k: int = 4):
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
    def similarity_search_with_score(self, query: str, k: int = 4):
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


class _FakeSingleChunkLongVectorStore:
    def similarity_search_with_score(self, query: str, k: int = 4):
        _ = (query, k)
        long_chunk = "setup = True " + ("x " * 320) + "target_call(random_state=42)"
        return [
            (
                Document(
                    page_content=long_chunk,
                    metadata={
                        "source": "uploads/session/sample_pipeline.py",
                        "chunk_id": 0,
                        "cell_id": None,
                        "start_offset": 0,
                        "end_offset": len(long_chunk),
                        "document_chunk_count": 1,
                        "document_char_count": len(long_chunk),
                    },
                ),
                0.12,
            )
        ]


class _FakeRetriever:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore or _FakeVectorStore()


class EvidencePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tavily_payload = {"results": []}
        self.search_responder = None
        validate_doc_url.cache_clear()
        self.addCleanup(validate_doc_url.cache_clear)

        def respond(method, url, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response.url = url
            if method.lower() == "post" and url == "https://api.tavily.com/search":
                payload = self.search_responder(kwargs["json"]) if self.search_responder else self.tavily_payload
                response._content = json.dumps(payload).encode("utf-8")
            elif method.lower() == "head":
                response._content = b""
            else:
                raise AssertionError(f"unexpected HTTP request: {method} {url}")
            response.raw = BytesIO(response.content)
            return response

        http_patcher = patch("requests.sessions.Session.request", side_effect=respond)
        http_patcher.start()
        self.addCleanup(http_patcher.stop)

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
        upload_item = {
            "kind": "local",
            "tool": "upload_search",
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
                        "evidence": [upload_item],
                        "diagnostics": {
                            "tool": "upload_search",
                            "route": "upload",
                            "status": "success",
                            "message": "",
                            "query": "uploaded file",
                            "attempt": 1,
                        },
                    }
                ),
                name="upload_search",
                tool_call_id="2",
            ),
            ToolMessage(content="not-json", name="upload_search", tool_call_id="3"),
            ToolMessage(content=json.dumps([upload_item]), name="save_text", tool_call_id="4"),
        ]
        debug = DebugCollector().build(response={}, updated_messages=messages, graph_total_ms=0, upload_retriever_build_ms=None)
        observed = debug["observed_evidence"]
        errors = debug["errors"]

        self.assertEqual(len(observed), 2)
        self.assertTrue(any(item["tool"] == "tavily_search" for item in observed))
        self.assertTrue(any(item["tool"] == "upload_search" for item in observed))
        self.assertTrue(any("tool:upload_search" in error for error in errors))

    def test_response_payload_excludes_observed_evidence_when_it_was_not_adopted(self) -> None:
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

        response = _evidence_response(evidence_payload)
        unused_evidence = {
            **evidence_payload[0],
            "source_id": "url:https://pandas.pydata.org/docs/",
            "document_id": "url:https://pandas.pydata.org/docs/",
            "url_or_path": "https://pandas.pydata.org/docs/",
            "title": "Pandas docs",
        }
        response["messages"][1] = ToolMessage(
            content=json.dumps({"evidence": [*evidence_payload, unused_evidence]}),
            name="tavily_search", tool_call_id="call-1",
        )
        result = _assemble_response(response)

        self.assertEqual(len(result["response_payload"]["evidence"]), 1)
        self.assertEqual(len(result["debug"]["observed_evidence"]), 2)
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
        self.assertEqual(with_retriever["diagnostics"]["metric"], "l2")
        self.assertEqual(with_retriever["diagnostics"]["score_direction"], "lower_is_better")
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
        self.assertAlmostEqual(evidence[0]["score"], 1.0 - (0.87 / math.sqrt(2.0)))

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
            [1, 0],
        )

    def test_upload_search_clamps_negative_raw_distances_to_max_similarity(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.upload_search_tool.func(
            query="uploaded info",
            k=4,
            retriever=_FakeRetriever(vectorstore=_FakeNegativeScoreVectorStore()),
        )

        evidence = result["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0]["score"], 1.0)

    def test_upload_search_uses_query_window_for_single_chunk_files(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        result = registry.upload_search_tool.func(
            query="random_state parameter",
            k=4,
            retriever=_FakeRetriever(vectorstore=_FakeSingleChunkLongVectorStore()),
        )

        evidence = result["evidence"]
        self.assertEqual(len(evidence), 1)
        self.assertLess(len(evidence[0]["snippet"]), 500)
        self.assertIn("target_call(random_state=42)", evidence[0]["snippet"])
        self.assertNotIn("...", evidence[0]["snippet"])

    def test_docs_search_filters_to_allowed_doc_prefixes(self) -> None:
        self.tavily_payload = {
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

    def test_docs_search_returns_no_result_when_all_urls_are_filtered(self) -> None:
        self.tavily_payload = {
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
        self.assertEqual(result["diagnostics"]["provider_result_count"], 1)
        self.assertEqual(result["diagnostics"]["filtered_path_prefix_count"], 1)
        self.assertEqual(result["diagnostics"]["final_evidence_count"], 0)

    def test_docs_search_blocks_huggingface_commit_diff_urls(self) -> None:
        self.tavily_payload = {
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

    def test_docs_search_returns_matching_evidence_when_library_fallback_finds_results(self) -> None:
        cases = [
            ("train_test_split 공식 문법을", ["scikit-learn.org"], "train_test_split sklearn.model_selection", "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html"),
            ("bare 공식 문서", ["docs.pears.com"], "Bare runtime API", "https://docs.pears.com/reference/bare/"),
            ("numpy", ["numpy.org"], "numpy user guide", "https://numpy.org/doc/stable/"),
            ("pandas", ["pandas.pydata.org"], "pandas user guide", "https://pandas.pydata.org/docs/"),
            ("fastapi", ["fastapi.tiangolo.com"], "fastapi tutorial", "https://fastapi.tiangolo.com/tutorial/"),
        ]
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        for query, domains, fallback_query, url in cases:
            with self.subTest(query=query):
                def respond(payload):
                    if payload["query"] != fallback_query or payload["include_domains"] != domains:
                        return {"results": []}
                    return {"results": [{"url": url, "title": fallback_query, "content": fallback_query + " API reference and usage examples.", "score": 0.92}]}
                self.search_responder = respond

                result = registry.tavily_search_tool.func(query=query)

                self.assertEqual(result["diagnostics"]["status"], "success")
                self.assertEqual([item["url_or_path"] for item in result["evidence"]], [url])


    def test_docs_search_filters_cross_library_docs_results_for_hinted_queries(
        self,
    ) -> None:
        self.tavily_payload = {
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
        self.assertEqual(result["diagnostics"]["provider_result_count"], 2)
        self.assertEqual(result["diagnostics"]["filtered_cross_domain_count"], 1)
        self.assertEqual(result["diagnostics"]["final_evidence_count"], 1)

    def test_docs_search_word_match_hint_does_not_match_substring(self) -> None:
        self.assertIsNone(infer_docs_query_hint("baremetal 공식 문서"))

    def test_debug_exposes_retrieval_and_planner_diagnostics(self) -> None:
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

        response = _evidence_response(evidence_payload)
        result = _assemble_response(response)

        self.assertEqual(result["debug"]["retrieval_diagnostics"][0]["status"], "success")
        self.assertEqual(result["debug"]["planner_diagnostics"]["status"], "heuristic_fallback")
        self.assertTrue(result["debug"]["planner_diagnostics"]["intent_required"])
        self.assertEqual(result["debug"]["planner_diagnostics"]["required_routes"], ["docs"])

    def test_debug_exposes_latency_breakdown(self) -> None:
        response = _evidence_response([])
        result = _assemble_response(response)

        latency_breakdown = result["debug"]["latency_breakdown"]
        self.assertIsNotNone(latency_breakdown)
        self.assertGreaterEqual(latency_breakdown["graph_total_ms"], 0)
        self.assertEqual(latency_breakdown["stage_totals_ms"]["planner_ms"], 12)
        self.assertEqual(latency_breakdown["retrieval_routes"][0]["route"], "docs")
        self.assertEqual(latency_breakdown["synthesis_attempts"][0]["mode"], "structured_only")

    def test_agent_manager_returns_error_latency_when_query_is_blank(self) -> None:
        manager = AgentFlowManager(AppSettings(openai_api_key="test", tavily_api_key="test"))
        self.addCleanup(manager.close)

        result = manager.run_agent_flow("   ")

        self.assertEqual(result["response_payload"]["answer"], "query must not be blank")
        self.assertEqual(result["debug"]["observability_status"], "failed")
        self.assertGreaterEqual(result["debug"]["latency_breakdown"]["server_total_ms"], 0)

    @patch("httpx.Client.send", side_effect=httpx.ReadTimeout("synthesis unavailable"))
    def test_agent_manager_preserves_synthesis_error_latency_when_model_request_fails(self, _send) -> None:
        settings = AppSettings(openai_api_key="test", tavily_api_key="test")
        model = ChatOpenAI(model="gpt-5-mini", api_key="test", max_retries=0)

        def synthesize(state: GraphState) -> dict:
            answer = model.invoke([HumanMessage(content=state["runtime"].user_input)])
            return {"messages": [answer]}

        graph = StateGraph(GraphState)
        graph.add_node("synthesis", _instrument_stage_node("synthesis", synthesize))
        graph.add_edge(START, "synthesis")
        graph.add_edge("synthesis", END)
        manager = AgentFlowManager(settings)
        manager.graph = graph.compile()
        self.addCleanup(manager.close)

        result = manager.run_agent_flow("question")

        self.assertEqual(result["debug"]["observability_status"], "failed")
        self.assertIn("timed out", result["message"].lower())
        latency = result["debug"]["latency_breakdown"]
        self.assertEqual(
            [(event["stage"], event["status"]) for event in latency["stage_attempts"]],
            [("synthesis", "error")],
        )
        self.assertGreaterEqual(latency["stage_attempts"][0]["latency_ms"], 0)
        self.assertGreaterEqual(latency["graph_total_ms"], 0)
        self.assertGreaterEqual(latency["server_total_ms"], 0)

    def test_debug_aggregates_llm_calls_into_debug_metadata(self) -> None:
        response = _response_with_llm_calls()
        result = _assemble_response(response)

        self.assertEqual(result["debug"]["token_usage"]["prompt_tokens"], 32)
        self.assertEqual(result["debug"]["token_usage"]["completion_tokens"], 8)
        self.assertEqual(result["debug"]["token_usage"]["total_tokens"], 40)
        self.assertEqual(result["debug"]["model_name"], "gpt-5-mini")
        self.assertEqual(result["debug"]["models_used"], ["gpt-5-nano", "gpt-5-mini"])
        self.assertEqual(len(result["debug"]["llm_calls"]), 2)
        self.assertEqual(
            [item["path"] for item in result["debug"]["llm_calls"]],
            ["structured", "structured"],
        )

    def test_debug_falls_back_to_current_turn_ai_message_metadata(self) -> None:
        response = _response_with_ai_metadata()
        result = _assemble_response(response)

        self.assertEqual(result["debug"]["token_usage"]["prompt_tokens"], 14)
        self.assertEqual(result["debug"]["token_usage"]["completion_tokens"], 6)
        self.assertEqual(result["debug"]["token_usage"]["total_tokens"], 20)
        self.assertEqual(result["debug"]["model_name"], "gpt-5-mini")
        self.assertEqual(result["debug"]["models_used"], ["gpt-5-mini"])
        self.assertEqual(len(result["debug"]["llm_calls"]), 1)
        self.assertEqual(result["debug"]["llm_calls"][0]["path"], "direct")

    @patch("openai.resources.embeddings.Embeddings.create", autospec=True)
    def test_upload_is_searchable_when_session_builds_with_configured_credentials(self, embed_request) -> None:
        def embed(client, *, input, **kwargs):
            if client._client.api_key != "test-key":
                raise ValueError("wrong embedding credentials")
            return {"data": [{"embedding": [1.0, 0.0, 0.0], "index": index} for index, _ in enumerate(input)]}

        # Only the external embedding request is replaced; files, Chroma and session state are real.
        embed_request.side_effect = embed
        settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
        with TemporaryDirectory() as root:
            upload = Path(root) / "uploads" / "evidence-pipeline" / "sample.py"
            upload.parent.mkdir(parents=True)
            upload.write_text("target_call(random_state=42)\n", encoding="utf-8")
            session = SessionContext()
            runner = ExecutionRunner(settings=settings, graph=None, session=session)
            try:
                state, build_ms = runner.prepare_graph_state("random_state", str(upload))
                self.assertIsNone(build_ms)
                result = build_tool_registry(settings).upload_search_tool.func(
                    query="random_state", retriever=state["runtime"].retriever,
                )
                self.assertEqual(result["diagnostics"]["status"], "success")
                self.assertEqual(result["evidence"][0]["kind"], "local")
                self.assertEqual(result["evidence"][0]["url_or_path"], str(upload))
                self.assertIn("random_state=42", result["evidence"][0]["snippet"])
                self.assertGreaterEqual(runner.finalize_pending_upload_retriever(), 0)
            finally:
                runner.cancel_pending_upload_retriever()
                session.close()

    def test_save_tool_message_does_not_override_final_answer(self) -> None:
        response = _response_with_save_receipt()
        result = _assemble_response(response)

        self.assertTrue(result["message"].startswith("final answer before save"))
        self.assertIn("저장 완료:", result["message"])
        self.assertTrue(result["filepath"].endswith("response_20260101_010101.txt"))
        self.assertTrue(result["response_payload"]["answer"].startswith("final answer before save"))
        self.assertEqual(result["response_payload"]["claims"], [])


    def test_docs_search_post_filters_cross_library_domains_for_hinted_queries(self) -> None:
        self.tavily_payload = {
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
