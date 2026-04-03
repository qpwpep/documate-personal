import unittest

from src.evidence import truncate_snippet
from src.nodes.planner.query_sanitizer import sanitize_retrieval_query
from src.nodes.validation.evidence_validator import route_passes_validation


class BackendRetrievalRegressionTest(unittest.TestCase):
    def test_upload_query_keeps_identifier_for_hybrid_requests(self) -> None:
        sanitized = sanitize_retrieval_query(
            route="upload",
            query="train_test_split 공식 문법을 설명하고 업로드 노트북의 실제 사용 예를 찾아줘.",
        )
        self.assertIn("train_test_split", sanitized)

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
        from src.evidence import EvidenceItem

        evidence_items = [EvidenceItem.model_validate(item) for item in items]
        self.assertTrue(
            route_passes_validation(
                "upload",
                "업로드한 파일에서 groupby를 어떻게 쓰는지 찾아서 설명해줘",
                evidence_items,
            )
        )


if __name__ == "__main__":
    unittest.main()
