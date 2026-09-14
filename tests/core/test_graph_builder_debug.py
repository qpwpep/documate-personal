import json
import unittest

import requests
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text
from src.core.request_contracts import RequestContract
from src.core.contracts.boundary.debug import get_debug_state
from src.infra.chunking import chunk_python_text
from src.core.contracts.boundary.graph import build_graph_state_input
from src.runtime.agent_runtime.debug_collector import DebugCollector
from src.runtime.graph_builder import _instrument_stage_node, build_agent_graph
from src.infra.settings import AppSettings

from .helpers import _CaptureStructuredSynthesizeLLM
from src.infra.tools.docs_search.url_validation import validate_doc_url


def _http_response(url: str, payload: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response._content = json.dumps(payload or {}).encode()
    response._content_consumed = True
    return response


class _UploadVectorStore:
    def similarity_search_with_score(self, query: str, k: int = 4):
        documents = chunk_python_text(
            path="uploads/demo/sample_pipeline.py",
            text="X = np.concatenate([train, test], axis=0)",
            chunk_size=800, chunk_overlap=120,
        )
        return [(documents.hydrate(document), 0.2) for document in documents.chunks]


class _EvidenceAwareSynthesisLLM(_CaptureStructuredSynthesizeLLM):
    def invoke(self, messages):
        packet_text = next(message.content for message in messages if str(message.content).startswith("[Evidence Packet]"))
        packet = json.loads(packet_text.split("\n", 2)[2])
        docs = next(item for item in packet if item["source_type"] == "official")
        upload = next(item for item in packet if item["source_type"] == "upload")
        self.payload = {"blocks": [
            {"type": "paragraph", "content": [
                {"text": "NumPy concatenate joins arrays along an existing axis.", "basis": "source", "refs": [docs["id"]]},
                {"text": "The uploaded code combines train and test using axis=0.", "basis": "inference", "refs": [docs["id"], upload["id"]]},
            ]},
            {"type": "code", "language": "python", "content": {"text": upload["excerpt"], "basis": "excerpt", "refs": [upload["id"]]}},
        ]}
        return super().invoke(messages)


class GraphBuilderDebugTest(unittest.TestCase):
    def test_partial_debug_patch_preserves_existing_debug_fields(self) -> None:
        wrapped = _instrument_stage_node(
            "post_synthesis_validation",
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
        self.assertTrue(
            any(item.get("stage") == "post_synthesis_validation" for item in debug.latency_trace)
        )

    def test_debug_collector_keeps_validation_events_out_of_runtime_errors(self) -> None:
        debug = DebugCollector().build(
            response=build_graph_state_input(
                user_input="question",
                messages=[],
                debug={
                    "retrieval_errors": ["tavily_search: failed (timeout)"],
                    "validation_errors": ["validate_evidence: retry_reason=unresolved_references"],
                    "validation_events": ["validate_evidence: retry_reason=unresolved_references"],
                },
            ),
            updated_messages=[HumanMessage(content="question")],
            graph_total_ms=10,
            upload_retriever_build_ms=None,
        )

        self.assertEqual(debug["errors"], ["tavily_search: failed (timeout)"])
        self.assertEqual(
            debug["validation_events"],
            ["validate_evidence: retry_reason=unresolved_references"],
        )

    @patch("requests.head")
    @patch("requests.post")
    @patch("src.infra.llm.ChatOpenAI")
    def test_debug_survives_validation_and_action_postprocess(
        self,
        provider_model,
        http_post,
        http_head,
    ) -> None:
        validate_doc_url.cache_clear()
        self.addCleanup(validate_doc_url.cache_clear)
        http_head.side_effect = lambda url, **kwargs: _http_response(url)
        http_post.side_effect = lambda url, **kwargs: _http_response(
            url,
            {"results": [{
                "url": "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                "title": "NumPy concatenate",
                "content": "NumPy concatenate joins a sequence of arrays along an existing axis.",
                "score": 0.94,
            }]},
        )
        settings = AppSettings(openai_api_key="test", tavily_api_key="test")
        def provider(**kwargs):
            if kwargs.get("model") == settings.planner_model:
                return _CaptureStructuredSynthesizeLLM(payload={
                    "request_contract": RequestContract().to_wire().model_dump(mode="json"),
                    "use_retrieval": True,
                    "tasks": [
                        {"route": "docs", "query": "numpy concatenate official docs", "k": 3},
                        {"route": "upload", "query": "numpy concatenate uploaded example", "k": 3},
                    ],
                }, include_raw=True)
            return _EvidenceAwareSynthesisLLM(include_raw=True)

        provider_model.side_effect = provider

        graph = build_agent_graph(settings)
        result = graph.invoke(
            build_graph_state_input(
                user_input="Explain NumPy concatenate from official docs and compare it with the uploaded file example.",
                messages=[],
                retriever=SimpleNamespace(vectorstore=_UploadVectorStore()),
            )
        )

        debug = get_debug_state(result)
        self.assertEqual(len(debug.retrieval_diagnostics), 2)
        self.assertEqual([item.route for item in debug.retrieval_diagnostics], ["docs", "upload"])
        self.assertEqual([item.stage for item in debug.llm_calls], ["planner", "synthesis"])
        self.assertTrue(all(item.usage_metadata["total_tokens"] == 14 for item in debug.llm_calls))
        stage_events = [
            item for item in debug.latency_trace if isinstance(item, dict) and item.get("kind") == "stage"
        ]
        self.assertTrue(any(item.get("stage") == "pre_synthesis_validation" for item in stage_events))
        self.assertTrue(any(item.get("stage") == "post_synthesis_validation" for item in stage_events))
        self.assertTrue(any(item.get("stage") == "action_postprocess" for item in stage_events))
        self.assertTrue(
            any(
                item.get("source") == "planner" and item.get("decision") == "retrieve"
                for item in debug.edge_decisions
            )
        )
        self.assertEqual(debug.planner_errors, [])
        self.assertEqual(
            {citation.evidence.snapshot.source_type for citation in result["response"].result.citations},
            {"official", "upload"},
        )
        self.assertTrue(export_answer_text(result["response"].result))
        self.assertEqual(
            [message.content for message in result["messages"] if isinstance(message, AIMessage)],
            [export_answer_text(result["response"].result)],
        )
        self.assertTrue(all(check.reference_status != "missing" for check in result["response"].result.checks))
        self.assertIn("exact_match", {check.support_status for check in result["response"].result.checks})



if __name__ == "__main__":
    unittest.main()
