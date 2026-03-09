import json
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from src.eval.runner_online import _run_single_case
from src.eval.schemas import BenchmarkCase, BenchmarkConfig


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


class _DummyJudge:
    def __init__(self, result: tuple[float | None, str | None, str | None]):
        self._result = result

    def score_case(self, case: BenchmarkCase, response_text: str, tool_calls: list[str]):
        _ = (case, response_text, tool_calls)
        return self._result


class RunnerErrorBucketsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = BenchmarkCase(case_id="docs_seed_001", category="docs_only", query="test query")
        self.config = BenchmarkConfig()
        self.fixtures_path = Path("data/benchmarks/fixtures/cases.generated.jsonl")

    @patch("src.eval.runner_online.requests.post", side_effect=requests.Timeout)
    def test_timeout_goes_to_runtime_errors(self, _mock_post) -> None:
        result = _run_single_case(
            run_id="run-timeout",
            endpoint="http://localhost:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=1,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )
        self.assertTrue(any("request timeout" in msg for msg in result.runtime_errors))
        self.assertEqual(result.response_errors, [])
        self.assertEqual(result.judge_errors, [])

    @patch("src.eval.runner_online.requests.post")
    def test_contract_error_goes_to_response_errors(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": "legacy string response",
                "trace": "x",
                "file_path": "",
                "debug": {"tool_calls": [], "token_usage": {}, "observed_evidence": []},
            },
        )
        result = _run_single_case(
            run_id="run-contract",
            endpoint="http://localhost:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )
        self.assertEqual(result.runtime_errors, [])
        self.assertTrue(any("response payload must be an object" in msg for msg in result.response_errors))

    @patch("src.eval.runner_online.requests.post")
    def test_judge_error_goes_to_judge_errors(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "ok", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "tool_calls": ["tavily_search"],
                    "token_usage": {},
                    "observed_evidence": [],
                },
            },
        )
        result = _run_single_case(
            run_id="run-judge",
            endpoint="http://localhost:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, "judge parse fail")),
            config=self.config,
        )
        self.assertEqual(result.runtime_errors, [])
        self.assertEqual(result.response_errors, [])
        self.assertIn("judge parse fail", result.judge_errors)

    @patch("src.eval.runner_online.requests.post")
    def test_runner_preserves_retrieval_diagnostic_statuses(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "ok", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "tool_calls": ["tavily_search", "upload_search", "rag_search"],
                    "token_usage": {},
                    "observed_evidence": [],
                    "retrieval_diagnostics": [
                        {
                            "tool": "tavily_search",
                            "route": "docs",
                            "status": "success",
                            "message": "",
                            "query": "docs query",
                            "attempt": 1,
                        },
                        {
                            "tool": "upload_search",
                            "route": "upload",
                            "status": "no_result",
                            "message": "no uploaded evidence found",
                            "query": "upload query",
                            "attempt": 1,
                        },
                        {
                            "tool": "rag_search",
                            "route": "local",
                            "status": "error",
                            "message": "local search failed",
                            "query": "local query",
                            "attempt": 1,
                        },
                        {
                            "tool": "upload_search",
                            "route": "upload",
                            "status": "unavailable",
                            "message": "upload retriever unavailable",
                            "query": "upload query",
                            "attempt": 2,
                        },
                    ],
                    "planner_diagnostics": {
                        "status": "heuristic_fallback",
                        "reason": "planner_failed_or_invalid",
                        "fallback_routes": ["docs", "upload"],
                    },
                },
            },
        )
        result = _run_single_case(
            run_id="run-diagnostics",
            endpoint="http://localhost:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )
        self.assertEqual(
            [item.status for item in result.retrieval_diagnostics],
            ["success", "no_result", "error", "unavailable"],
        )
        self.assertIsNotNone(result.planner_diagnostics)
        self.assertEqual(result.planner_diagnostics.status, "heuristic_fallback")

    @patch("src.eval.runner_online.requests.post")
    def test_runner_parses_validator_reason_from_retry_context(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "need more evidence", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "tool_calls": ["tavily_search"],
                    "token_usage": {},
                    "observed_evidence": [],
                    "retry_context": {
                        "attempt": 1,
                        "max_retries": 1,
                        "retry_reason": "no_evidence",
                        "retrieval_feedback": "low evidence confidence; broaden query or switch route.",
                    },
                },
            },
        )

        result = _run_single_case(
            run_id="run-validator",
            endpoint="http://localhost:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )

        self.assertEqual(result.validator_reason, "no_evidence")
        self.assertEqual(
            result.validator_feedback,
            "low evidence confidence; broaden query or switch route.",
        )


if __name__ == "__main__":
    unittest.main()
