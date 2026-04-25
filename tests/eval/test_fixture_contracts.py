import unittest
from collections import Counter
from pathlib import Path

from src.eval.io import load_cases_jsonl


class FixtureContractsTest(unittest.TestCase):
    def test_upload_fixture_cases_expect_upload_search(self) -> None:
        paths = [
            Path("data/benchmarks/fixtures/cases.seed.jsonl"),
            Path("data/benchmarks/fixtures/cases.regression.seed.jsonl"),
            Path("data/benchmarks/fixtures/cases.generated.jsonl"),
        ]

        for path in paths:
            for case in load_cases_jsonl(path):
                if not case.upload_fixture:
                    continue
                self.assertIn("upload_search", case.expected_tools, msg=f"{path}: {case.case_id}")
                self.assertNotIn("rag_search", case.expected_tools, msg=f"{path}: {case.case_id}")

    def test_slack_cases_include_destination_hint(self) -> None:
        paths = [
            Path("data/benchmarks/fixtures/cases.seed.jsonl"),
            Path("data/benchmarks/fixtures/cases.regression.seed.jsonl"),
            Path("data/benchmarks/fixtures/cases.generated.jsonl"),
        ]

        for path in paths:
            for case in load_cases_jsonl(path):
                if "slack_notify" not in case.expected_tools:
                    continue
                destination_missing_is_expected = "SLACK_DESTINATION_MISSING" in case.expected_error_codes
                step_destination_exists = any(
                    step.slack_channel_id or step.slack_user_id or step.slack_email
                    for step in case.steps
                )
                self.assertTrue(
                    bool(case.slack_channel_id or case.slack_user_id or case.slack_email)
                    or step_destination_exists
                    or destination_missing_is_expected,
                    msg=f"{path}: {case.case_id}",
                )

    def test_v3_seed_fixture_has_schema_v2_and_20_cases_per_category(self) -> None:
        cases = load_cases_jsonl(Path("data/benchmarks/fixtures/cases.seed.jsonl"))

        self.assertEqual(len(cases), 80)
        self.assertEqual(Counter(case.category for case in cases), {
            "docs_only": 20,
            "rag_only": 20,
            "hybrid": 20,
            "tool_action": 20,
        })
        self.assertTrue(all(case.schema_version == 2 for case in cases))
        self.assertTrue(all(case.benchmark_fixture_schema_version == 2 for case in cases))
        self.assertTrue(all(case.golden_criteria.required_facts for case in cases))

    def test_generated_fixture_is_release_balanced_320(self) -> None:
        cases = load_cases_jsonl(Path("data/benchmarks/fixtures/cases.generated.jsonl"))
        cell_counts = Counter((case.category, case.scenario) for case in cases)

        self.assertEqual(len(cases), 320)
        self.assertEqual(set(cell_counts.values()), {20})


if __name__ == "__main__":
    unittest.main()
