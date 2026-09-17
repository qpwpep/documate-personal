import unittest
from collections import Counter
from pathlib import Path

from src.eval.config_models import BenchmarkCase
from src.eval.generate_cases import build_generated_cases, generate_cases_file


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
    def test_generated_scenarios_preserve_preparation_and_attachments(self) -> None:
        seed_cases = [
            _seed_case("seed_docs", "docs_only"),
            _seed_case("seed_rag", "rag_only"),
            _seed_case("seed_hybrid", "hybrid"),
            BenchmarkCase(
                case_id="seed_tool", category="tool_action", query="방금 답변을 저장해줘.",
                setup_turns=["첫 파일을 설명해줘.", "두 번째 파일과 비교해줘."],
                upload_fixtures=["first.py", "second.py"], expected_tools=["save_text"],
                save_expectation={"outcome": "required_success",
                                  "target": {"kind": "setup_answer", "setup_turn_index": 1}},
            ),
        ]
        generated = build_generated_cases(
            seed_cases=seed_cases, regression_seed_cases=seed_cases,
            target=120, random_seed=42,
        )
        action_cases = [case for case in generated if case.category == "tool_action"]
        self.assertEqual({case.scenario for case in action_cases}, {
            "seed_mutation", "adversarial", "regression", "ambiguity",
        })
        for case in action_cases:
            self.assertEqual(case.setup_turns, seed_cases[-1].setup_turns)
            self.assertEqual(case.resolved_upload_fixtures, ["first.py", "second.py"])
            self.assertEqual(case.save_expectation, seed_cases[-1].save_expectation)
            for conflicting_instruction in ("추정", "출처 표시 없이", "가능한 해석", "단계별", "실무 관점"):
                self.assertNotIn(conflicting_instruction, case.query)
        self.assertGreater(len({case.query for case in action_cases}), 1)

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
