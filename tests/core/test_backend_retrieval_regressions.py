import unittest

from hypothesis import given, strategies as st

from src.core.contracts import RetrievalDiagnostic
from src.core.contracts.debug import RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import truncate_snippet
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.planner.query_sanitizer import (
    sanitize_planner_output_queries,
    sanitize_retrieval_query,
)
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from src.infra.tools.local_rag import build_upload_search_tool


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

    def test_upload_query_keeps_identifier_for_hybrid_requests(self) -> None:
        sanitized = sanitize_retrieval_query(
            route="upload",
            query="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘.",
        )
        self.assertIn("train_test_split", sanitized)

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

    def test_truncate_snippet_preserves_tail_for_long_local_files(self) -> None:
        text = (
            "import pandas as pd\n"
            + ("x = 1\n" * 120)
            + 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()\n'
            + 'sales_with_profile = all_sales.merge(profiles, on="user_id", how="left")\n'
        )
        truncated = truncate_snippet(text, max_length=220)
        self.assertIsNotNone(truncated)
        self.assertIn("import pandas as pd", truncated)
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
                    retrieval={"evidence_log": [
                        {
                            "kind": "official", "tool": "tavily_search",
                            "source_id": "url:https://numpy.org/doc/stable/",
                            "url_or_path": "https://numpy.org/doc/stable/",
                            "snippet": "Official documentation describes the operation.", "score": 0.9,
                        },
                        {
                            "kind": "local", "tool": "upload_search",
                            "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=96",
                            "document_id": "path:uploads/demo/sample.py",
                            "url_or_path": "uploads/demo/sample.py",
                            "snippet": "train_test_split(X, y, test_size=0.2, random_state=42)",
                            "score": score, "chunk_id": 0, "start_offset": 0, "end_offset": 96,
                        },
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
                self.assertAlmostEqual(updates["retry"].score_avg, (0.9 + score) / 2)
                self.assertNotIn("response", updates)

    def test_upload_search_normalizes_raw_l2_scores_without_warning(self) -> None:
        upload_tool = build_upload_search_tool()

        class _Doc:
            def __init__(self, text, cell_id):
                self.page_content = text
                self.metadata = {
                    "source": "uploads/demo/sample_pipeline.ipynb",
                    "cell_id": cell_id,
                    "chunk_id": 0,
                    "start_offset": 0,
                    "end_offset": len(text),
                }

        class _VectorStore:
            def similarity_search_with_score(self, query, k=4):
                _ = (query, k)
                return [
                    (_Doc("from sklearn.model_selection import train_test_split", 1), 1.8),
                    (_Doc("train_test_split(X, y, test_size=0.2, random_state=42)", 2), 1.2),
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
        scores = [item["score"] for item in payload["evidence"]]
        self.assertTrue(all(0.0 <= score <= 1.0 for score in scores))
        self.assertEqual(len(scores), 2)
        self.assertAlmostEqual(max(scores), 0.1514718625761431)
        self.assertEqual(min(scores), 0.0)


if __name__ == "__main__":
    unittest.main()
