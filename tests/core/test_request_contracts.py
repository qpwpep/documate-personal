import unittest

from src.core.request_contracts import infer_answer_contract


class RequestContractsTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
