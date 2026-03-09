import unittest
from pathlib import Path

from src.eval.schemas import load_cases_jsonl


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
                self.assertTrue(
                    bool(case.slack_channel_id or case.slack_user_id or case.slack_email),
                    msg=f"{path}: {case.case_id}",
                )


if __name__ == "__main__":
    unittest.main()
