import unittest

from src.core.request_contracts import infer_answer_contract


class RequestContractsTest(unittest.TestCase):
    def test_code_example_markers_require_code_example_section(self) -> None:
        examples = [
            "BeautifulSoup으로 특정 태그 찾는 예제를 보여줘",
            "BeautifulSoup 샘플 코드 보여줘",
            "Show a BeautifulSoup code example.",
            "Show sample code for BeautifulSoup.",
        ]

        for query in examples:
            with self.subTest(query=query):
                contract = infer_answer_contract(query, ["docs"])

                self.assertEqual(contract.required_sections, ["code_example"])

    def test_hybrid_contract_requires_upload_section_for_explicit_comparison(self) -> None:
        contract = infer_answer_contract(
            "Compare official docs with the uploaded code.",
            ["docs", "upload"],
        )

        self.assertEqual(
            contract.required_sections,
            ["official_docs", "upload_code", "comparison"],
        )

    def test_hybrid_contract_omits_upload_section_when_comparison_is_implicit(self) -> None:
        contract = infer_answer_contract(
            "Explain official docs using the uploaded code context.",
            ["docs", "upload"],
        )

        self.assertEqual(
            contract.required_sections,
            ["official_docs", "comparison"],
        )

    def test_hybrid_contract_preserves_source_sections_with_requested_presentation_sections(self) -> None:
        contract = infer_answer_contract(
            "Compare official docs with the uploaded code as a summary and checklist.",
            ["docs", "upload"],
        )

        self.assertEqual(
            contract.required_sections,
            ["summary", "checklist", "official_docs", "upload_code", "comparison"],
        )
        self.assertTrue(contract.split_by_source)


if __name__ == "__main__":
    unittest.main()
