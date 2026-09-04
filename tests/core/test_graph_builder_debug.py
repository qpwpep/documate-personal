import json
import unittest

import requests
from langchain_core.documents import Document
from types import SimpleNamespace
from unittest.mock import patch

from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.debug import get_debug_state
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
        return [(
            Document(
                page_content="X = np.concatenate([train, test], axis=0)",
                metadata={
                    "source": "uploads/demo/sample_pipeline.ipynb",
                    "cell_id": 1,
                    "chunk_id": 0,
                    "start_offset": 0,
                    "end_offset": 64,
                },
            ),
            0.2,
        )]


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
                    "validation_errors": ["validate_evidence: retry_reason=unsupported_claims"],
                    "validation_events": ["validate_evidence: retry_reason=unsupported_claims"],
                },
            ),
            updated_messages=[HumanMessage(content="question")],
            graph_total_ms=10,
            upload_retriever_build_ms=None,
        )

        self.assertEqual(debug["errors"], ["tavily_search: failed (timeout)"])
        self.assertEqual(
            debug["validation_events"],
            ["validate_evidence: retry_reason=unsupported_claims"],
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
        provider_model.side_effect = lambda **kwargs: _CaptureStructuredSynthesizeLLM(
            payload={
                "use_retrieval": True,
                "tasks": [
                    {"route": "docs", "query": "numpy concatenate official docs", "k": 3},
                    {"route": "upload", "query": "numpy concatenate uploaded example", "k": 3},
                ],
            } if kwargs.get("model") == settings.planner_model else {
                "answer": "공식 설명과 업로드 비교를 정리했습니다.",
                "claims": [
                    {
                        "text": "NumPy concatenate는 기존 축을 따라 배열 시퀀스를 결합한다.",
                        "evidence_ids": ["url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html"],
                        "confidence": 0.94,
                    },
                    {
                        "text": "업로드 파일은 axis=0으로 train/test를 이어 붙이는 예시를 사용한다.",
                        "evidence_ids": ["path:uploads/demo/sample_pipeline.ipynb#cell=1;chunk=0;start=0;end=64"],
                        "confidence": 0.88,
                    },
                ],
                "confidence": 0.91,
            },
        )

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
            {source_id for claim in result["response"].payload.claims for source_id in claim.evidence_ids},
            {
                "url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                "path:uploads/demo/sample_pipeline.ipynb#cell=1;chunk=0;start=0;end=64",
            },
        )
        self.assertTrue(result["response"].final_answer)


if __name__ == "__main__":
    unittest.main()
