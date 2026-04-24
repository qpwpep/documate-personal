import unittest

from src.core.contracts import RetrievalDiagnostic
from src.core.evidence import truncate_snippet
from src.runtime.nodes.planner.query_sanitizer import sanitize_retrieval_query
from src.runtime.nodes.validation.evidence_validator import route_passes_validation
from src.infra.tools.local_rag import build_local_rag_tools


class BackendRetrievalRegressionTest(unittest.TestCase):
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

    def test_upload_validation_salvages_clamped_score_with_identifier_hit(self) -> None:
        items = [
            {
                "kind": "local",
                "tool": "upload_search",
                "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=96",
                "document_id": "path:uploads/demo/sample.py",
                "url_or_path": "uploads/demo/sample.py",
                "snippet": 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()',
                "score": 0.0,
                "chunk_id": 0,
                "start_offset": 0,
                "end_offset": 96,
            }
        ]
        from src.core.evidence import EvidenceItem

        evidence_items = [EvidenceItem.model_validate(item) for item in items]
        self.assertTrue(
            route_passes_validation(
                "upload",
                "업로드한 파일에서 groupby를 어떻게 쓰는지 찾아서 설명해줘",
                evidence_items,
            )
        )

    def test_hybrid_upload_validation_rejects_generic_zero_score_match(self) -> None:
        from src.core.evidence import EvidenceItem

        evidence_items = [
            EvidenceItem.model_validate(
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=96",
                    "document_id": "path:uploads/demo/sample.py",
                    "url_or_path": "uploads/demo/sample.py",
                    "snippet": 'X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)',
                    "score": 0.0,
                    "chunk_id": 0,
                    "start_offset": 0,
                    "end_offset": 96,
                }
            )
        ]

        self.assertFalse(
            route_passes_validation(
                "upload",
                "uploaded notebook example",
                evidence_items,
                required_routes=["docs", "upload"],
                diagnostics=[
                    RetrievalDiagnostic(
                        route="upload",
                        tool="upload_search",
                        status="success",
                        query="uploaded notebook example",
                        normalized_score=0.0,
                    )
                ],
                user_input="Compare official train_test_split parameters with the uploaded notebook example.",
            )
        )

    def test_hybrid_upload_validation_accepts_minimum_normalized_score(self) -> None:
        from src.core.evidence import EvidenceItem

        evidence_items = [
            EvidenceItem.model_validate(
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=96",
                    "document_id": "path:uploads/demo/sample.py",
                    "url_or_path": "uploads/demo/sample.py",
                    "snippet": "uploaded notebook result",
                    "score": 0.15,
                    "chunk_id": 0,
                    "start_offset": 0,
                    "end_offset": 96,
                }
            )
        ]

        self.assertTrue(
            route_passes_validation(
                "upload",
                "uploaded notebook example",
                evidence_items,
                required_routes=["docs", "upload"],
                diagnostics=[
                    RetrievalDiagnostic(
                        route="upload",
                        tool="upload_search",
                        status="success",
                        query="uploaded notebook example",
                        normalized_score=0.15,
                    )
                ],
                user_input="Compare official docs with the uploaded notebook example.",
            )
        )

    def test_hybrid_upload_validation_accepts_identifier_plus_keyword(self) -> None:
        from src.core.evidence import EvidenceItem

        evidence_items = [
            EvidenceItem.model_validate(
                {
                    "kind": "local",
                    "tool": "upload_search",
                    "source_id": "path:uploads/demo/sample.py#chunk=0;start=0;end=96",
                    "document_id": "path:uploads/demo/sample.py",
                    "url_or_path": "uploads/demo/sample.py",
                    "snippet": 'X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)',
                    "score": 0.05,
                    "chunk_id": 0,
                    "start_offset": 0,
                    "end_offset": 96,
                }
            )
        ]

        self.assertTrue(
            route_passes_validation(
                "upload",
                "uploaded notebook example",
                evidence_items,
                required_routes=["docs", "upload"],
                diagnostics=[
                    RetrievalDiagnostic(
                        route="upload",
                        tool="upload_search",
                        status="success",
                        query="uploaded notebook example",
                        normalized_score=0.05,
                    )
                ],
                user_input="Compare official train_test_split random_state guidance with the uploaded notebook example.",
            )
        )

    def test_upload_search_normalizes_raw_l2_scores_without_warning(self) -> None:
        settings = type("Settings", (), {"openai_api_key": "test-key"})()
        _rag_tool, upload_tool = build_local_rag_tools(settings)

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
        payload = upload_tool.func(
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
