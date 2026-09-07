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
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence, parse_search_hits
from src.core.answer_schema import ActionReceipt, AnswerResponse, finalize_answer, export_answer_text, text_document
from src.core.contracts.graph_state import DebugState, PlannerState, ResponseState, RetrievalState
from src.runtime.agent_runtime import DebugCollector, ExecutionRunner, ResponseAssembler, SessionContext
from src.runtime.graph_builder import _instrument_stage_node
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.infra.tools.docs_search import infer_docs_query_hint
from src.infra.tools.docs_search.url_validation import validate_doc_url


def _hit(text="broadcasting", *, uri="https://numpy.org/doc/stable/", source_type="official", score=0.98):
    snapshot = build_snapshot(source_uri=uri, title="Source", media_type="text/plain", source_type=source_type,
                              content=text, parser="fixture", parser_version="1")
    element = DocumentElement(element_id="body", kind="paragraph", text=text)
    return SearchHit(evidence=build_evidence(snapshot=snapshot, element=element),
                     score=RetrievalScore(metric="provider_score", raw=score, normalized=score, direction="higher"), rank=1)


def _response_with_hits(hits):
    evidence = [hit.evidence for hit in hits]
    result = finalize_answer(text_document("final answer", basis="source" if evidence else "interaction",
                                           refs=[evidence[0].id] if evidence else []), evidence)
    return {
        "messages": [HumanMessage(content="question"),
                     ToolMessage(content=json.dumps({"hits": [hit.model_dump(mode="json") for hit in hits],
                                                      "diagnostics": {"tool": "tavily_search", "route": "docs", "status": "success"}}),
                                 name="tavily_search", tool_call_id="call-1"),
                     AIMessage(content="This later string must not replace the checked document.")],
        "retrieval": RetrievalState(hit_log=[hit.model_dump(mode="json") for hit in hits]),
        "response": ResponseState(result=result, evidence_packet=evidence),
        "planner": PlannerState(diagnostics={
            "status": "heuristic_fallback", "reason": "planner_failed_or_invalid", "fallback_routes": ["docs"],
            "intent_required": True, "required_routes": ["docs"], "override_applied": False, "override_reason": None,
        }),
        "debug": DebugState(retrieval_diagnostics=[{"tool": "tavily_search", "route": "docs", "status": "success", "query": "question", "attempt": 1}],
                            latency_trace=[
            {"kind": "stage", "stage": "planner", "attempt": 1, "latency_ms": 12, "status": "heuristic_fallback"},
            {"kind": "retrieval_route", "route": "docs", "tool": "tavily_search", "attempt": 1, "latency_ms": 48, "status": "success"},
            {"kind": "stage", "stage": "retrieval", "attempt": 1, "latency_ms": 50, "status": "success"},
            {"kind": "synthesis_attempt", "attempt": 1, "mode": "structured_only", "structured_ms": 22, "fallback_ms": None, "total_ms": 22},
            {"kind": "stage", "stage": "synthesis", "attempt": 1, "latency_ms": 22, "status": "structured_only"},
            {"kind": "stage", "stage": "validation", "attempt": 1, "latency_ms": 3, "status": "pass"},
        ]),
    }


def _response_with_llm_calls():
    response = _response_with_hits([])
    response["debug"] = DebugState(llm_calls=[
        {"stage": "planner", "attempt": 1, "path": "structured", "response_metadata": {"model_name": "gpt-5-nano"},
         "usage_metadata": {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}},
        {"stage": "synthesis", "attempt": 1, "path": "structured", "response_metadata": {"model_name": "gpt-5-mini"},
         "usage_metadata": {"input_tokens": 20, "output_tokens": 5, "total_tokens": 25}},
    ])
    return response


def _response_with_ai_metadata():
    return {"messages": [HumanMessage(content="question"), AIMessage(
        content="final answer", response_metadata={"model_name": "gpt-5-mini"},
        usage_metadata={"input_tokens": 14, "output_tokens": 6, "total_tokens": 20})],
        "response": ResponseState(result=finalize_answer(text_document("final answer"), []), synthesis_attempt=1)}


def _response_with_save_receipt():
    receipt = ActionReceipt(kind="save_text", status="success", file_path="output/save_text/response_20260101_010101.txt", message="Saved output")
    return {"messages": [HumanMessage(content="question"), AIMessage(content="final answer before save"),
                         ToolMessage(content=json.dumps({"message": "Saved output", "file_path": receipt.file_path}),
                                     name="save_text", tool_call_id="save-1")],
            "response": ResponseState(result=finalize_answer(text_document("final answer before save"), [], actions=[receipt]))}


def _assemble_response(response):
    debug = DebugCollector().build(response=response, updated_messages=response["messages"],
                                   graph_total_ms=100, upload_retriever_build_ms=None)
    return ResponseAssembler().assemble(response=response, debug_info=debug)


def _answer_text(result):
    return export_answer_text(AnswerResponse.model_validate(result["response"]))


def _indexed(text, *, start=0, end=None, cell_index=2, uri="uploads/session/sample_pipeline.ipynb"):
    snapshot = build_snapshot(source_uri=uri, title=Path(uri).name, media_type="text/plain", source_type="upload",
                              content=text, parser="fixture", parser_version="1")
    element = DocumentElement(element_id="source-cell", kind="code", text=text, language="python",
                              anchors=[SourceAnchor(kind="notebook", cell_id=f"native-{cell_index}", cell_index=cell_index)])
    ref = build_evidence(snapshot=snapshot, element=element, start=start, end=end)
    return Document(page_content=ref.excerpt, metadata={"source": uri, "evidence_ref": ref.model_dump_json(),
                                                     "document_chunk_count": 1, "document_char_count": len(text)})


class _FakeVectorStore:
    def similarity_search_with_score(self, query, k=4):
        return [(_indexed("setup value uploaded snippet", start=12), 0.87)]


class _FakeDedupVectorStore:
    def similarity_search_with_score(self, query, k=4):
        text = "first chunk\nsecond chunk"
        return [(_indexed(text, end=11, cell_index=0), 0.81), (_indexed(text, start=12, cell_index=0), 0.79)]


class _FakeNegativeScoreVectorStore:
    def similarity_search_with_score(self, query, k=4):
        return [(_indexed("negative score snippet", cell_index=1), -0.24)]


class _FakeSingleChunkLongVectorStore:
    def similarity_search_with_score(self, query, k=4):
        text = "setup = True " + ("x " * 320) + "target_call(random_state=42)"
        return [(_indexed(text, uri="uploads/session/sample_pipeline.py"), 0.12)]


class _FakeRetriever:
    def __init__(self, vectorstore=None):
        self.vectorstore = vectorstore or _FakeVectorStore()


class EvidencePipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tavily_payload = {"results": []}
        self.search_responder = None
        self.search_requests = []
        validate_doc_url.cache_clear()
        self.addCleanup(validate_doc_url.cache_clear)

        def respond(method, url, **kwargs):
            response = requests.Response()
            response.status_code = 200
            response.url = url
            if method.lower() == "post" and url == "https://api.tavily.com/search":
                self.search_requests.append(kwargs["json"])
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

    def test_extract_observed_hits_uses_tool_native_payloads(self) -> None:
        """Only retrieval messages contribute distinct observed hits; malformed payloads are reported."""
        docs = _hit().model_dump(mode="json")
        upload = _hit("local snippet", uri="uploads/s1/sample.ipynb", source_type="upload", score=0.71).model_dump(mode="json")
        messages = [
            ToolMessage(content=json.dumps({"hits": [docs, docs]}), name="tavily_search", tool_call_id="1"),
            ToolMessage(content=json.dumps({"hits": [upload]}), name="upload_search", tool_call_id="2"),
            ToolMessage(content="not-json", name="upload_search", tool_call_id="3"),
            ToolMessage(content=json.dumps({"hits": [upload]}), name="save_text", tool_call_id="4"),
        ]
        debug = DebugCollector().build(response={}, updated_messages=messages, graph_total_ms=0, upload_retriever_build_ms=None)
        self.assertEqual(debug["observed_hits"], [docs, upload])
        self.assertTrue(any("invalid JSON" in error for error in debug["errors"]))

    def test_response_keeps_canonical_document_and_excludes_unused_observed_hits(self) -> None:
        """Response citations derive from the checked content, independently of observed or later chat text."""
        adopted = _hit()
        unused = _hit("pandas", uri="https://pandas.pydata.org/docs/")
        response = _response_with_hits([adopted])
        response["messages"][1] = ToolMessage(content=json.dumps({"hits": [item.model_dump(mode="json") for item in (adopted, unused)]}),
                                               name="tavily_search", tool_call_id="call-1")
        result = _assemble_response(response)
        self.assertEqual(result["response"], response["response"].result.model_dump(mode="json"))
        self.assertEqual(len(result["debug"]["observed_hits"]), 2)
        self.assertEqual(_answer_text(result), "final answer [1]")
        self.assertEqual([citation["evidence"]["id"] for citation in result["response"]["citations"]], [adopted.evidence.id])

    def test_upload_search_returns_typed_source_range_and_handles_missing_retriever(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        unavailable = registry.upload_search_tool(query="uploaded info", k=3, retriever=None)
        self.assertEqual(unavailable["diagnostics"]["status"], "unavailable")
        self.assertEqual(unavailable["hits"], [])
        payload = registry.upload_search_tool(query="uploaded info", k=3, retriever=_FakeRetriever())
        self.assertEqual(payload["diagnostics"]["status"], "success")
        self.assertEqual(payload["diagnostics"]["metric"], "l2")
        self.assertEqual(payload["diagnostics"]["score_direction"], "lower_is_better")
        hits = parse_search_hits(payload)
        self.assertEqual(len(hits), 1)
        ref = hits[0].evidence
        self.assertEqual(ref.snapshot.source_type, "upload")
        self.assertEqual(ref.snapshot.source_uri, "uploads/session/sample_pipeline.ipynb")
        self.assertEqual(ref.element.anchors[0].cell_index, 2)
        self.assertEqual(ref.element.anchors[0].cell_id, "native-2")
        self.assertEqual((ref.selection.start, ref.selection.end), (12, 28))
        self.assertEqual(ref.excerpt, "uploaded snippet")
        self.assertEqual(ref.element.text[ref.selection.start:ref.selection.end], ref.excerpt)
        self.assertAlmostEqual(hits[0].score.normalized, 1.0 - (0.87 / math.sqrt(2.0)))

    def test_upload_search_keeps_multiple_source_ranges_from_same_snapshot(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.upload_search_tool(query="uploaded info", k=4, retriever=_FakeRetriever(vectorstore=_FakeDedupVectorStore()))
        hits = parse_search_hits(result)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].evidence.snapshot, hits[1].evidence.snapshot)
        self.assertNotEqual(hits[0].evidence.id, hits[1].evidence.id)
        self.assertEqual([hit.evidence.selection.start for hit in hits], [12, 0])

    def test_upload_search_clamps_normalized_distance_without_losing_raw_value(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.upload_search_tool(query="uploaded info", k=4, retriever=_FakeRetriever(vectorstore=_FakeNegativeScoreVectorStore()))
        hits = parse_search_hits(result)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].score.normalized, 1.0)
        self.assertEqual(hits[0].score.raw, -0.24)

    def test_upload_search_uses_exact_query_window_for_single_chunk_files(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        result = registry.upload_search_tool(query="random_state parameter", k=4, retriever=_FakeRetriever(vectorstore=_FakeSingleChunkLongVectorStore()))
        refs = [hit.evidence for hit in parse_search_hits(result)]
        self.assertEqual(len(refs), 1)
        self.assertLess(len(refs[0].excerpt), 500)
        self.assertIn("target_call(random_state=42)", refs[0].excerpt)
        self.assertEqual(refs[0].excerpt, refs[0].element.text[refs[0].selection.start:refs[0].selection.end])

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
        result = registry.tavily_search_tool(query="official docs")

        urls = [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]]
        self.assertIn("https://fastapi.tiangolo.com/ko/tutorial/response-model", urls)
        self.assertIn("https://huggingface.co/docs/transformers/index", urls)
        self.assertNotIn("https://huggingface.co/datasets/foo/bar", urls)

    def test_docs_search_returns_no_result_when_all_urls_are_filtered(self) -> None:
        """Bounded reformulation cannot promote non-document URLs into evidence."""
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
        result = registry.tavily_search_tool(query="official docs")

        self.assertEqual(result["diagnostics"]["status"], "no_result")
        self.assertEqual(result["diagnostics"]["answerability"], "missing")
        self.assertEqual(result["diagnostics"]["missing_requirements"], ["topic"])
        self.assertEqual(result["hits"], [])
        self.assertEqual(result["diagnostics"]["provider_result_count"], 2)
        self.assertEqual(result["diagnostics"]["filtered_path_prefix_count"], 2)
        self.assertEqual(result["diagnostics"]["final_evidence_count"], 0)
        self.assertEqual(len(self.search_requests), 2)
        for request in self.search_requests:
            self.assertIn("official docs", request["query"])
            self.assertEqual(request["include_domains"], self.search_requests[0]["include_domains"])
        self.assertEqual(result["diagnostics"]["attempted_queries"],
                         [request["query"] for request in self.search_requests])

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
        result = registry.tavily_search_tool(query="official docs")

        urls = [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]]
        self.assertEqual(urls, ["https://huggingface.co/docs/transformers/index"])

    def test_docs_search_recovers_evidence_with_bounded_subject_preserving_reformulation(self) -> None:
        """After an empty search, one reformulation preserves the subject and official domain."""
        cases = [
            ("train_test_split 공식 문법을", ["scikit-learn.org"], "sklearn.model_selection.train_test_split",
             "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
             "Split arrays into train and test subsets."),
            ("bare 공식 문서", ["docs.pears.com"], "Bare runtime API",
             "https://docs.pears.com/reference/bare/", "Bare provides a JavaScript runtime."),
            ("numpy", ["numpy.org"], "NumPy manual",
             "https://numpy.org/doc/stable/", "NumPy provides array operations and numerical routines."),
            ("pandas", ["pandas.pydata.org"], "pandas documentation",
             "https://pandas.pydata.org/docs/", "pandas provides labeled data structures."),
            ("fastapi", ["fastapi.tiangolo.com"], "FastAPI tutorial",
             "https://fastapi.tiangolo.com/tutorial/", "FastAPI declares API routes with Python type annotations."),
        ]
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))
        for query, domains, title, url, content in cases:
            with self.subTest(query=query):
                self.search_requests.clear()

                def respond(_payload):
                    if len(self.search_requests) == 1:
                        return {"results": []}
                    return {"results": [{"url": url, "title": title, "content": content, "score": 0.92}]}
                self.search_responder = respond

                result = registry.tavily_search_tool(query=query)

                self.assertEqual(result["diagnostics"]["status"], "success")
                self.assertEqual(result["diagnostics"]["answerability"], "covered")
                self.assertEqual(result["diagnostics"]["missing_requirements"], [])
                hits = parse_search_hits(result)
                self.assertEqual([(hit.evidence.snapshot.source_uri, hit.evidence.snapshot.title,
                                   hit.evidence.excerpt) for hit in hits], [(url, title, content)])
                self.assertEqual(result["diagnostics"]["provider_result_count"], 1)
                self.assertEqual(len(self.search_requests), 2)
                for request in self.search_requests:
                    self.assertIn(query, request["query"].replace('"', ""))
                    self.assertEqual(request["include_domains"], domains)
                self.assertNotEqual(self.search_requests[0]["query"], self.search_requests[1]["query"])
                self.assertEqual(result["diagnostics"]["attempted_queries"],
                                 [request["query"] for request in self.search_requests])


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
        result = registry.tavily_search_tool(query="numpy 공식 문서")

        urls = [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]]
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
        result = _assemble_response(_response_with_hits([_hit()]))
        self.assertEqual(result["debug"]["retrieval_diagnostics"][0]["status"], "success")
        self.assertEqual(result["debug"]["planner_diagnostics"]["status"], "heuristic_fallback")
        self.assertTrue(result["debug"]["planner_diagnostics"]["intent_required"])
        self.assertEqual(result["debug"]["planner_diagnostics"]["required_routes"], ["docs"])

    def test_debug_exposes_latency_breakdown(self) -> None:
        response = _response_with_hits([])
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

        self.assertEqual(_answer_text(result), "query must not be blank")
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
        self.assertIn("timed out", _answer_text(result).lower())
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
                result = build_tool_registry(settings).upload_search_tool(
                    query="random_state", retriever=state["runtime"].retriever,
                )
                self.assertEqual(result["diagnostics"]["status"], "success")
                self.assertEqual(result["hits"][0]["evidence"]["snapshot"]["source_type"], "upload")
                self.assertEqual(result["hits"][0]["evidence"]["snapshot"]["source_uri"], str(upload))
                self.assertIn("random_state=42", parse_search_hits(result)[0].evidence.excerpt)
                self.assertGreaterEqual(runner.finalize_pending_upload_retriever(), 0)
            finally:
                runner.cancel_pending_upload_retriever()
                session.close()

    def test_save_receipt_is_separate_from_canonical_answer(self) -> None:
        response = _response_with_save_receipt()
        result = _assemble_response(response)
        self.assertEqual(_answer_text(result), "final answer before save")
        self.assertEqual(result["response"], response["response"].result.model_dump(mode="json"))
        self.assertEqual(result["response"]["actions"][0]["status"], "success")
        self.assertTrue(result["response"]["actions"][0]["file_path"].endswith("response_20260101_010101.txt"))


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
        result = registry.tavily_search_tool(query="pandas official docs")

        self.assertEqual(
            [item["evidence"]["snapshot"]["source_uri"] for item in result["hits"]],
            ["https://pandas.pydata.org/docs/reference/api/pandas.concat.html"],
        )

    def test_docs_search_word_match_hints_do_not_match_substrings_for_library_names(self) -> None:
        self.assertIsNone(infer_docs_query_hint("numpydoc official docs"))
        self.assertIsNone(infer_docs_query_hint("fastapiusers official docs"))
        self.assertIsNone(infer_docs_query_hint("pandasai official docs"))


if __name__ == "__main__":
    unittest.main()
