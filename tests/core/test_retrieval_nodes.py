import json
import copy
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import requests
from langchain_core.documents import Document

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence, parse_search_hits
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.settings import AppSettings
from src.infra.tools import build_tool_registry
from src.infra.tools.docs_search.url_validation import validate_doc_url
from src.infra.tools.local_rag import build_upload_search_tool
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node


def _http_response(url: str, payload: dict | None = None) -> requests.Response:
    response = requests.Response()
    response.status_code = 200
    response.url = url
    response._content = json.dumps(payload or {}).encode()
    response._content_consumed = True
    return response


def _upload_document(text: str = "train_test_split(X, y, test_size=0.2, random_state=42)", cell_index: int = 2) -> Document:
    snapshot = build_snapshot(source_uri="uploads/demo/sample_pipeline.ipynb", title="sample_pipeline.ipynb",
                              media_type="application/x-ipynb+json", source_type="upload", content=text,
                              parser="test-notebook", parser_version="1")
    element = DocumentElement(element_id=f"cell-{cell_index}", kind="code", text=text, language="python",
                              anchors=[SourceAnchor(kind="notebook", cell_id=f"native-{cell_index}", cell_index=cell_index)])
    return Document(
        page_content=text,
        metadata={
            "source": "uploads/demo/sample_pipeline.ipynb",
            "evidence_ref": build_evidence(snapshot=snapshot, element=element).model_dump_json(),
        },
    )


class _VectorStore:
    def __init__(self, *, completed: Event | None = None, unavailable: bool = False):
        self.completed = completed
        self.unavailable = unavailable

    def similarity_search_with_score(self, query: str, k: int = 4):
        if self.unavailable:
            raise AssertionError("Preserved upload results must avoid another database read")
        if self.completed is not None:
            self.completed.set()
        return [(_upload_document(), 0.2)]


class RetrievalNodeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AppSettings(openai_api_key="test-key", tavily_api_key="test-key")
        self.registry = build_tool_registry(self.settings)
        validate_doc_url.cache_clear()
        self.addCleanup(validate_doc_url.cache_clear)
        head = patch("requests.head", side_effect=lambda url, **kwargs: _http_response(url))
        head.start()
        self.addCleanup(head.stop)
        post = patch("requests.post", side_effect=self._docs_response)
        self.http_post = post.start()
        self.addCleanup(post.stop)

    @staticmethod
    def _docs_response(url: str, **kwargs) -> requests.Response:
        return _http_response(
            url,
            {"results": [{
                "url": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                "title": "train_test_split",
                "content": "train_test_split splits arrays or matrices into random train and test subsets.",
                "score": 0.99,
            }]},
        )

    def _dispatch(self):
        return make_retrieve_dispatch_node(
            self.registry.tavily_search_tool,
            self.registry.upload_search_tool,
            verbose=False,
        )

    @staticmethod
    def _state(*, retry: dict | None = None, vectorstore=None, routes=("docs", "upload")):
        return build_graph_state_input(
            user_input="Compare official train_test_split docs with my uploaded notebook.",
            messages=[],
            retriever=SimpleNamespace(vectorstore=vectorstore or _VectorStore()),
            planner={"output": PlannerOutput(
                use_retrieval=True,
                tasks=[RetrievalTask(route=route, query="train_test_split", k=3) for route in routes],
            )},
            retry=retry or {},
        )

    def test_upload_search_reranks_parameter_query_toward_usage_cell(self) -> None:
        class _UsageVectorStore:
            def similarity_search_with_score(self, query: str, k: int = 4):
                usage = _upload_document()
                imported = _upload_document("from sklearn.model_selection import train_test_split", cell_index=1)
                return [(imported, 0.30), (usage, 0.28)]

        payload = build_upload_search_tool()(
            query="업로드 노트북에서 train_test_split 파라미터를 찾아줘",
            k=4,
            retriever=SimpleNamespace(vectorstore=_UsageVectorStore()),
        )

        hit = parse_search_hits(payload)[0]
        self.assertEqual(hit.evidence.element.anchors[0].cell_index, 2)
        self.assertIn("test_size=0.2", hit.evidence.excerpt)

    def test_retrieve_dispatch_merges_docs_and_upload_evidence_with_tool_messages(self) -> None:
        updates = self._dispatch()(self._state())

        self.assertEqual(
            [(hit.evidence.route, hit.evidence.snapshot.source_type) for hit in parse_search_hits(updates["retrieval"].hit_log)],
            [("docs", "official"), ("upload", "upload")],
        )
        self.assertEqual([message.name for message in updates["messages"]], ["tavily_search", "upload_search"])
        self.assertEqual(
            [(item.route, item.status) for item in updates["debug"].retrieval_diagnostics],
            [("docs", "success"), ("upload", "success")],
        )
        self.assertEqual(parse_search_hits(updates["retrieval"].hit_log)[1].evidence.element.anchors[0].cell_index, 2)

    def test_retrieve_dispatch_records_error_diagnostics_when_provider_fails(self) -> None:
        self.http_post.side_effect = requests.ConnectionError("boom")

        updates = self._dispatch()(self._state(routes=("docs",)))

        payload = json.loads(updates["messages"][0].content)
        self.assertEqual(payload["diagnostics"]["status"], "error")
        self.assertEqual(updates["debug"].retrieval_diagnostics[0].status, "error")
        self.assertEqual(updates["retrieval"].hit_log, [])

    def test_invalid_hit_is_removed_from_state_tool_message_and_diagnostics(self) -> None:
        """Malformed source references cannot survive through another retrieval representation."""
        payload = build_upload_search_tool()(query="train_test_split", retriever=SimpleNamespace(vectorstore=_VectorStore()))
        invalid = copy.deepcopy(payload["hits"][0])
        invalid["evidence"]["id"] = "ref:tampered"
        payload["hits"].append(invalid)
        payload["diagnostics"]["evidence_count"] = 2
        payload["diagnostics"]["final_evidence_count"] = 2
        dispatch = make_retrieve_dispatch_node(lambda **kwargs: {}, lambda **kwargs: payload, verbose=False)
        updates = dispatch(self._state(routes=("upload",)))
        hits = updates["retrieval"].hit_log
        self.assertEqual(len(hits), 1)
        self.assertEqual(json.loads(updates["messages"][0].content)["hits"], hits)
        self.assertEqual(updates["debug"].retrieval_diagnostics[0].evidence_count, 1)
        self.assertEqual(updates["debug"].retrieval_diagnostics[0].final_evidence_count, 1)
        self.assertTrue(updates["debug"].retrieval_errors)

    def test_retrieve_dispatch_preserves_planner_task_order_when_upload_finishes_first(self) -> None:
        upload_completed = Event()

        def _wait_for_upload(url: str, **kwargs) -> requests.Response:
            self.assertTrue(upload_completed.wait(timeout=2), "Retrieval routes must run concurrently")
            return self._docs_response(url, **kwargs)

        self.http_post.side_effect = _wait_for_upload
        updates = self._dispatch()(self._state(vectorstore=_VectorStore(completed=upload_completed)))

        self.assertEqual(
            [(item.route, item.status) for item in updates["debug"].retrieval_diagnostics],
            [("docs", "success"), ("upload", "success")],
        )
        self.assertEqual(
            [event["route"] for event in updates["debug"].latency_trace if event.get("kind") == "retrieval_route"],
            ["docs", "upload"],
        )

    def test_retrieve_dispatch_reuses_preserved_upload_results_on_docs_retry(self) -> None:
        initial = self._dispatch()(self._state(routes=("upload",)))
        upload_hits = initial["retrieval"].hit_log
        diagnostic = initial["debug"].retrieval_diagnostics[0].model_dump()
        updates = self._dispatch()(self._state(
            vectorstore=_VectorStore(unavailable=True),
            retry={
                "attempt": 1,
                "failed_routes": ["docs"],
                "preserved_hits": upload_hits,
                "preserved_retrieval_diagnostics": [diagnostic],
            },
        ))

        self.assertEqual(updates["retrieval"].hit_log[1:], upload_hits)
        self.assertEqual(
            [(item.route, item.status) for item in updates["debug"].retrieval_diagnostics],
            [("docs", "success"), ("upload", "success")],
        )
        self.assertEqual(updates["debug"].retrieval_errors, [])
