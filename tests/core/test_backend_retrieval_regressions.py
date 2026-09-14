import unittest

from hypothesis import given, strategies as st

from src.core.contracts import RetrievalDiagnostic
from src.core.contracts.debug import RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence, parse_search_hits
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.planner.query_sanitizer import (
    sanitize_planner_output_queries,
    sanitize_retrieval_query,
)
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from src.infra.tools.local_rag import build_upload_search_tool
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag.serialization import build_query_focused_snippet


def _hit(text: str, *, source_type: str, score: float) -> dict:
    uri = "https://numpy.org/doc/stable/" if source_type == "official" else "uploads/demo/sample.py"
    snapshot = build_snapshot(source_uri=uri, title="Test source", media_type="text/plain", source_type=source_type,
                              content=text, parser="test", parser_version="1")
    element = DocumentElement(element_id="source", kind="paragraph", text=text)
    return SearchHit(evidence=build_evidence(snapshot=snapshot, element=element),
                     score=RetrievalScore(metric="test-relevance", raw=score, normalized=score, direction="higher"), rank=1).model_dump(mode="json")


class BackendRetrievalRegressionTest(unittest.TestCase):
    def test_query_preserves_topic_when_identifiers_and_korean_terms_are_mixed(self) -> None:
        queries = (
            "Python 결제 API 멱등성 키 설계 공식 문서 기술 레퍼런스",
            "PyMuPDF 공식 문서 PDF 글꼴 추출 및 포함 글꼴 확인",
            "Python json.loads 공식 문서 null 빈 문자열 예제",
        )
        for query in queries:
            for route in ("docs", "upload"):
                for retry_context in (None, RetryState(retry_reason="no_evidence")):
                    with self.subTest(query=query, route=route, retry=retry_context):
                        self.assertEqual(
                            sanitize_retrieval_query(
                                route=route, query=query, retry_context=retry_context,
                            ),
                            query,
                        )

    @given(
        query=st.text(min_size=1, max_size=500).filter(
            lambda text: any(not char.isspace() for char in text)
        ),
        route=st.sampled_from(("docs", "upload")),
        retry=st.booleans(),
    )
    def test_query_preserves_non_whitespace_content_when_normalized_repeatedly(
        self, query: str, route: str, retry: bool,
    ) -> None:
        retry_context = RetryState(retry_reason="no_evidence") if retry else None
        sanitized = sanitize_retrieval_query(
            route=route, query=query, retry_context=retry_context,
        )
        self.assertEqual(
            [char for char in sanitized if not char.isspace()],
            [char for char in query if not char.isspace()],
        )
        self.assertFalse(any(char.isspace() and char != " " for char in sanitized))
        self.assertNotIn("  ", sanitized)
        self.assertEqual(sanitized, sanitized.strip())
        self.assertEqual(
            sanitize_retrieval_query(
                route=route, query=sanitized, retry_context=retry_context,
            ),
            sanitized,
        )

    def test_plan_preserves_route_queries_and_limits_when_whitespace_is_normalized(self) -> None:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(
                    route="docs", query="  Python\tjson.loads: null\n빈 문자열?  ", k=4,
                ),
                RetrievalTask(
                    route="upload", query="  .py / .ipynb\n결제 API: 멱등성 키;  ", k=7,
                ),
            ],
        )
        self.assertEqual(
            sanitize_planner_output_queries(
                planner_output,
                user_input="공식 문서와 파일을 비교해줘.",
                retry_context=RetryState(retry_reason="no_evidence"),
            ),
            PlannerOutput(
                use_retrieval=True,
                tasks=[
                    RetrievalTask(
                        route="docs", query="Python json.loads: null 빈 문자열?", k=4,
                    ),
                    RetrievalTask(
                        route="upload", query=".py / .ipynb 결제 API: 멱등성 키;", k=7,
                    ),
                ],
            ),
        )

    def test_query_rejects_unsupported_retrieval_route(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported retrieval route: archive"):
            sanitize_retrieval_query(route="archive", query="Python json.loads")

    def test_docs_query_keeps_identifiers_before_korean_particles(self) -> None:
        pydantic_query = sanitize_retrieval_query(
            route="docs",
            query="Pydantic v2 Field와 validation 방식을 설명해줘.",
        )
        pytorch_query = sanitize_retrieval_query(
            route="docs",
            query="PyTorch Dataset과 DataLoader 차이를 공식 문서 기준으로 설명해줘.",
        )

        self.assertIn("Field", pydantic_query)
        self.assertIn("validation", pydantic_query)
        self.assertIn("Dataset", pytorch_query)
        self.assertIn("DataLoader", pytorch_query)

    def test_query_excerpt_is_contiguous_source_text_in_long_local_files(self) -> None:
        text = (
            "import pandas as pd\n"
            + ("x = 1\n" * 120)
            + 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()\n'
            + 'sales_with_profile = all_sales.merge(profiles, on="user_id", how="left")\n'
        )
        truncated = build_query_focused_snippet(text, query="groupby merge", max_length=220)
        self.assertIn(truncated, text)
        self.assertIn("groupby", truncated)
        self.assertIn("merge", truncated)

    def test_pre_synthesis_validation_keeps_existing_upload_evidence_at_low_scores(self) -> None:
        validate_node = make_pre_synthesis_validation_node(verbose=False)
        for score, upload_query in (
            (0.0, "uploaded notebook example"),
            (0.15, "uploaded notebook example"),
            (0.05, "train_test_split random_state"),
            (0.0, "groupby usage"),
        ):
            with self.subTest(score=score, query=upload_query):
                state = build_graph_state_input(
                    user_input="Compare official documentation with uploaded code.",
                    planner={"output": PlannerOutput(use_retrieval=True, tasks=[
                        RetrievalTask(route="docs", query="official docs", k=3),
                        RetrievalTask(route="upload", query=upload_query, k=3),
                    ])},
                    retrieval={"hit_log": [
                        _hit("Official documentation describes the operation.", source_type="official", score=0.9),
                        _hit("train_test_split(X, y, test_size=0.2, random_state=42)", source_type="upload", score=score),
                    ]},
                    debug={"retrieval_diagnostics": [RetrievalDiagnostic(
                        route="upload", tool="upload_search", status="success",
                        query=upload_query, normalized_score=score,
                    )]},
                )

                updates = validate_node(state)

                self.assertFalse(updates["retry"].needs_retry)
                self.assertIsNone(updates["retry"].retry_reason)
                self.assertEqual(updates["retry"].attempt, 0)
                self.assertNotIn("response", updates)

    def test_upload_search_normalizes_raw_l2_scores_without_warning(self) -> None:
        upload_tool = build_upload_search_tool()

        def document(text):
            indexed = chunk_python_text(path="uploads/demo/sample.py", text=text, chunk_size=800, chunk_overlap=120)
            return indexed.hydrate(indexed.chunks[0])

        class _VectorStore:
            def similarity_search_with_score(self, query, k=4):
                _ = (query, k)
                return [
                    (document("from sklearn.model_selection import train_test_split"), 1.8),
                    (document("train_test_split(X, y, test_size=0.2, random_state=42)"), 1.2),
                ]

        retriever = type("Retriever", (), {"vectorstore": _VectorStore()})()
        payload = upload_tool(
            query="업로드 노트북에서 train_test_split 파라미터를 찾아줘",
            k=4,
            retriever=retriever,
        )

        self.assertEqual(payload["diagnostics"]["warnings"], [])
        self.assertEqual(payload["diagnostics"]["metric"], "l2")
        self.assertEqual(payload["diagnostics"]["score_direction"], "lower_is_better")
        scores = [hit.score.normalized for hit in parse_search_hits(payload)]
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(max(scores), 0.1514718625761431)
        self.assertEqual(min(scores), 0.0)


if __name__ == "__main__":
    unittest.main()
