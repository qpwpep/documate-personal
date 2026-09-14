import unittest
from pathlib import Path

from pydantic import ValidationError

from src.eval.config_models import BenchmarkCase
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
                if not case.resolved_upload_fixtures:
                    continue
                self.assertIn("upload_search", case.expected_tools, msg=f"{path}: {case.case_id}")

    def test_previous_answer_action_fixtures_include_preparation_turns(self) -> None:
        for path in Path("data/benchmarks/fixtures").glob("cases.*.jsonl"):
            for case in load_cases_jsonl(path):
                if case.category != "tool_action":
                    continue
                self.assertTrue(case.setup_turns, msg=f"{path}: {case.case_id}")

    def test_fixture_expectations_use_current_upload_tool_name(self) -> None:
        for path in Path("data/benchmarks/fixtures").glob("cases.*.jsonl"):
            for case in load_cases_jsonl(path):
                self.assertNotIn(
                    "rag_search", case.expected_tools + case.forbidden_tools,
                    msg=f"{path}: {case.case_id}",
                )

    def test_legacy_upload_fixture_resolves_to_one_attachment(self) -> None:
        case = BenchmarkCase(
            case_id="legacy", category="rag_only", query="첨부 파일을 요약해줘.",
            upload_fixture="sample_pipeline.ipynb",
        )
        self.assertEqual(case.resolved_upload_fixtures, ["sample_pipeline.ipynb"])

    def test_scenario_round_trip_preserves_turn_and_attachment_order(self) -> None:
        payload = {
            "case_id": "scenario", "category": "rag_only", "query": "두 결과를 비교해줘.",
            "setup_turns": ["첫 파일을 요약해줘.", "두 번째 파일을 요약해줘."],
            "upload_fixtures": ["first.py", "second.ipynb"],
        }
        case = BenchmarkCase.model_validate(payload)
        restored = BenchmarkCase.model_validate_json(case.model_dump_json())
        self.assertEqual(restored.setup_turns, payload["setup_turns"])
        self.assertEqual(restored.resolved_upload_fixtures, payload["upload_fixtures"])

    def test_conflicting_upload_declarations_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValidationError, "upload_fixture"):
            BenchmarkCase(
                case_id="ambiguous", category="rag_only", query="첨부를 요약해줘.",
                upload_fixture="legacy.py", upload_fixtures=["other.py"],
            )

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
                self.assertTrue(
                    bool(case.slack_channel_id or case.slack_user_id or case.slack_email),
                    msg=f"{path}: {case.case_id}",
                )


if __name__ == "__main__":
    unittest.main()
