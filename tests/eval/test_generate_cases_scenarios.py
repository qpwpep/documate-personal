import unittest
from collections import Counter
from pathlib import Path

from src.eval.generate_cases import build_generated_cases, generate_cases_file
from src.eval.schemas import BenchmarkCase


def _seed_case(case_id: str, category: str) -> BenchmarkCase:
    upload = None
    expected_tools = []
    require_official = False
    require_local = False
    if category == "docs_only":
        expected_tools = ["tavily_search"]
        require_official = True
    elif category == "rag_only":
        expected_tools = ["upload_search"]
        require_local = True
        upload = "sample_pipeline.ipynb"
    elif category == "hybrid":
        expected_tools = ["tavily_search", "upload_search"]
        require_official = True
        require_local = True
        upload = "sample_pipeline.ipynb"
    elif category == "tool_action":
        expected_tools = ["save_text"]

    return BenchmarkCase(
        case_id=case_id,
        category=category,
        query=f"{category} query",
        upload_fixture=upload,
        expected_tools=expected_tools,
        require_official_citation=require_official,
        require_local_citation=require_local,
    )


class GenerateCasesScenarioTest(unittest.TestCase):
    def test_target_120_balances_category_and_scenario(self) -> None:
        seed_cases = [
            _seed_case("seed_docs", "docs_only"),
            _seed_case("seed_rag", "rag_only"),
            _seed_case("seed_hybrid", "hybrid"),
            _seed_case("seed_tool", "tool_action"),
        ]
        regression_cases = [
            _seed_case("reg_docs", "docs_only"),
            _seed_case("reg_rag", "rag_only"),
            _seed_case("reg_hybrid", "hybrid"),
            _seed_case("reg_tool", "tool_action"),
        ]

        generated = build_generated_cases(
            seed_cases=seed_cases,
            regression_seed_cases=regression_cases,
            target=120,
            random_seed=42,
        )
        self.assertEqual(len(generated), 120)

        category_counts = Counter(case.category for case in generated)
        scenario_counts = Counter(case.scenario for case in generated)
        cell_counts = Counter((case.category, case.scenario) for case in generated)

        for category in ["docs_only", "rag_only", "hybrid", "tool_action"]:
            self.assertEqual(category_counts[category], 30)
        for scenario in ["seed_mutation", "adversarial", "regression", "ambiguity"]:
            self.assertEqual(scenario_counts[scenario], 30)
        self.assertEqual(min(cell_counts.values()), 7)
        self.assertEqual(max(cell_counts.values()), 8)

    def test_missing_regression_seed_file_fails_fast(self) -> None:
        with self.assertRaises(FileNotFoundError):
            generate_cases_file(
                seed_path=Path("data/benchmarks/fixtures/cases.seed.jsonl"),
                regression_seed_path=Path("data/benchmarks/fixtures/missing.regression.seed.jsonl"),
                out_path=Path("data/benchmarks/fixtures/_tmp.generated.jsonl"),
                target=8,
                random_seed=42,
            )


if __name__ == "__main__":
    unittest.main()
