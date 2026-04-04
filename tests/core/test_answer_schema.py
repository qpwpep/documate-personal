import unittest

from src.answer_schema import build_deterministic_grounded_payload, clean_grounded_text
from src.evidence import EvidenceItem


class AnswerSchemaTest(unittest.TestCase):
    def test_clean_grounded_text_removes_markdown_and_navigation_lines(self) -> None:
        cleaned = clean_grounded_text(
            "\n".join(
                [
                    "# NumPy broadcasting",
                    "Home > Docs > API",
                    "Skip to content",
                    "Table of contents",
                    "- Broadcasting expands compatible array shapes.",
                    "Previous: Indexing routines",
                ]
            )
        )

        self.assertEqual(cleaned, "Broadcasting expands compatible array shapes.")

    def test_clean_grounded_text_drops_doc_title_signature_and_section_headings(self) -> None:
        cleaned = clean_grounded_text(
            "\n".join(
                [
                    "train_test_split",
                    "train_test_split(*arrays, test_size=None, random_state=None)",
                    "Parameters",
                    "Split arrays or matrices into random train and test subsets.",
                ]
            )
        )

        self.assertEqual(cleaned, "Split arrays or matrices into random train and test subsets.")

    def test_clean_grounded_text_drops_doc_titles_toc_fragments_and_broken_signatures(self) -> None:
        cleaned = clean_grounded_text(
            "\n".join(
                [
                    "Skip to content",
                    "train_test_split - scikit-learn 1.8.0 documentation",
                    "Parameters Returns Examples Notes",
                    r"train\_test\_split#. sklearn.model\_selection.train\_test\_split(*arrays, test_size=None)",
                    "Split arrays or matrices into random train and test subsets.",
                ]
            )
        )

        self.assertEqual(cleaned, "Split arrays or matrices into random train and test subsets.")

    def test_build_deterministic_grounded_payload_uses_cleaned_fallback_claim_text(self) -> None:
        payload = build_deterministic_grounded_payload(
            evidence_items=[
                EvidenceItem(
                    kind="official",
                    tool="tavily_search",
                    source_id="url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    document_id="url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    url_or_path="https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                    title="numpy.concatenate",
                    snippet="\n".join(
                        [
                            "On this page",
                            "- Use [`numpy.concatenate`](https://numpy.org) to join arrays along an existing axis.",
                            "Next: numpy.stack",
                        ]
                    ),
                    score=0.91,
                )
            ]
        )

        self.assertEqual(len(payload.claims), 1)
        self.assertEqual(
            payload.claims[0].text,
            "Use numpy.concatenate to join arrays along an existing axis.",
        )

    def test_build_deterministic_grounded_payload_builds_explanatory_hybrid_fallback(self) -> None:
        payload = build_deterministic_grounded_payload(
            evidence_items=[
                EvidenceItem(
                    kind="official",
                    tool="tavily_search",
                    source_id="url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                    document_id="url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                    url_or_path="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                    title="train_test_split",
                    snippet="Split arrays or matrices into random train and test subsets.",
                    score=0.92,
                ),
                EvidenceItem(
                    kind="local",
                    tool="upload_search",
                    source_id="path:uploads/demo/sample_pipeline.ipynb#cell=2;chunk=0;start=0;end=96",
                    document_id="path:uploads/demo/sample_pipeline.ipynb",
                    url_or_path="uploads/demo/sample_pipeline.ipynb",
                    title=None,
                    snippet="X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)",
                    score=0.81,
                    cell_id=2,
                    chunk_id=0,
                    start_offset=0,
                    end_offset=96,
                ),
            ]
        )

        self.assertEqual(len(payload.claims), 2)
        self.assertIn("공식 문서 기준으로는", payload.claims[0].text)
        self.assertIn("업로드 파일에서는", payload.claims[1].text)
        self.assertIn("근거는 공식 문서 1건과 업로드 파일 1건만 반영했습니다.", payload.answer)


if __name__ == "__main__":
    unittest.main()
