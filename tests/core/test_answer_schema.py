import unittest

import src.core.answer_schema as answer_schema
from src.core.answer_schema import AnswerSection, ClaimItem
from src.core.answer_schema.fallbacks import build_deterministic_grounded_payload
from src.core.answer_schema.rendering import build_empty_response_payload, render_payload_from_claims
from src.core.answer_schema.text_cleaning import clean_grounded_text
from src.core.evidence import EvidenceItem


class AnswerSchemaTest(unittest.TestCase):
    def test_answer_schema_barrel_reexports_split_modules(self) -> None:
        self.assertIs(answer_schema.build_deterministic_grounded_payload, build_deterministic_grounded_payload)
        self.assertIs(answer_schema.clean_grounded_text, clean_grounded_text)

    def test_placeholder_reference_sections_do_not_override_grounded_claims(self) -> None:
        evidence = EvidenceItem(
            kind="official",
            tool="tavily_search",
            source_id="url:https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree",
            document_id="url:https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree",
            url_or_path="https://www.crummy.com/software/BeautifulSoup/bs4/doc/#searching-the-tree",
            title="Beautiful Soup Documentation",
            snippet="The find_all method looks through a tag's descendants and retrieves matching tags.",
            score=0.93,
        )

        payload = render_payload_from_claims(
            claims=[
                ClaimItem(
                    text="BeautifulSoup에서는 find_all()로 특정 태그를 찾을 수 있습니다.",
                    evidence_ids=[evidence.source_id],
                )
            ],
            evidence_items=[evidence],
            confidence=0.93,
            sections=[
                AnswerSection(
                    kind="code",
                    heading="특정 태그 찾기 예제",
                    body="위 코드 참고",
                )
            ],
        )

        self.assertEqual(
            payload.answer,
            "BeautifulSoup에서는 find_all()로 특정 태그를 찾을 수 있습니다. [1]",
        )
        self.assertEqual(payload.sections, [])

    def test_placeholder_reference_answer_is_treated_as_empty(self) -> None:
        payload = build_empty_response_payload(answer="특정 태그 찾기 예제\n위 코드 참고")

        self.assertEqual(payload.answer, "")

    def test_reference_section_with_actual_code_is_kept(self) -> None:
        payload = build_empty_response_payload(
            sections=[
                AnswerSection(
                    kind="summary",
                    heading="예제",
                    body="아래 코드 참고:\n```python\nsoup.find_all('a')\n```",
                )
            ]
        )

        self.assertIn("soup.find_all('a')", payload.answer)

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
