import json
import unittest
from pathlib import Path
from unittest.mock import patch

import requests

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.online_runner import _run_single_case
from src.eval.reporting.histograms import build_analysis


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


class _DummyJudge:
    def __init__(self, result):
        self._result = result

    def score_case(self, case: BenchmarkCase, response_text: str, tool_calls: list[str]):
        _ = (case, response_text, tool_calls)
        return self._result


class RunnerErrorBucketsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = BenchmarkCase(case_id="docs_seed_001", category="docs_only", query="test query")
        self.config = BenchmarkConfig()
        self.fixtures_path = Path("data/benchmarks/fixtures/cases.generated.jsonl")

    @patch("src.eval.online_runner.case_runner.requests.post", side_effect=requests.Timeout)
    def test_timeout_goes_to_runtime_errors(self, _mock_post) -> None:
        result = _run_single_case(
            run_id="run-timeout",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=1,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )
        self.assertTrue(any("request timeout" in msg for msg in result.runtime_errors))
        self.assertEqual(result.response_errors, [])
        self.assertEqual(result.judge_errors, [])

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_contract_error_goes_to_response_errors(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": "legacy string response",
                "trace": "x",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": [],
                    "tool_call_count": 0,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )
        result = _run_single_case(
            run_id="run-contract",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )
        self.assertEqual(result.runtime_errors, [])
        self.assertTrue(any("response payload must be an object" in msg for msg in result.response_errors))

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_judge_error_goes_to_judge_errors(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "ok", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )
        result = _run_single_case(
            run_id="run-judge",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, "judge parse fail")),
            config=self.config,
        )
        self.assertEqual(result.runtime_errors, [])
        self.assertEqual(result.response_errors, [])
        self.assertIn("judge parse fail", result.judge_errors)

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_runner_preserves_retrieval_diagnostic_statuses(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "ok", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search", "upload_search", "rag_search"],
                    "tool_call_count": 3,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
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
                    "retry_context": None,
                    "latency_breakdown": None,
                },
            },
        )
        result = _run_single_case(
            run_id="run-diagnostics",
            endpoint="http://127.0.0.1:8000",
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

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_runner_parses_validator_reason_from_retry_context(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "need more evidence", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_evidence": [],
                    "retry_context": {
                        "attempt": 1,
                        "max_retries": 1,
                        "retry_reason": "no_evidence",
                        "retrieval_feedback": "low evidence confidence; broaden query or switch route.",
                    },
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )

        result = _run_single_case(
            run_id="run-validator",
            endpoint="http://127.0.0.1:8000",
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

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_runner_marks_missing_critical_debug_fields_as_response_error(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "ok", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "tool_calls": ["tavily_search"],
                    "observed_evidence": [],
                },
            },
        )

        result = _run_single_case(
            run_id="run-debug-contract",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )

        self.assertTrue(any("critical debug fields missing:" in msg for msg in result.response_errors))
        self.assertFalse(result.passed)

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_runner_applies_docs_judge_min_score_gate(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "ok", "evidence": []},
                "trace": "x",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )

        result = _run_single_case(
            run_id="run-judge-gate",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge(
                (
                    0.4,
                    "answer stayed too generic",
                    None,
                    {
                        "answer_quality": 0.4,
                        "groundedness": 0.5,
                        "citation_traceability": 0.5,
                        "tool_choice": 1.0,
                        "format_language": 0.7,
                    },
                )
            ),
            config=self.config,
        )

        self.assertIsNotNone(result.judge_subscores)
        self.assertEqual(result.judge_score_total, 0.4)
        self.assertTrue(any("judge_min_score audit failed:" in msg for msg in result.judge_errors))
        self.assertFalse(result.product_pass)
        self.assertFalse(result.release_pass)
        self.assertFalse(result.passed)

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_runner_parses_standard_error_codes_and_output_shape_metrics(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {
                    "answer": "ok",
                    "claims": [],
                    "sections": [
                        {"kind": "summary", "heading": "Summary", "body": "ok"},
                        {"kind": "comparison", "heading": "Compare", "body": "same"},
                    ],
                    "evidence": [],
                },
                "trace": "x",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "token_usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 21,
                        "total_tokens": 31,
                    },
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "error_codes": [
                        "RETRIEVAL_DOCS_TIMEOUT",
                        "LOCAL_RAG_FAILED",
                        "VALIDATION_UNSUPPORTED_CLAIMS",
                    ],
                    "validation_events": ["validate_evidence: retry_reason=unsupported_claims"],
                    "edge_decisions": [
                        {
                            "source": "planner",
                            "decision": "retrieve",
                            "reason": "retrieval_required:1_task(s)",
                        }
                    ],
                    "planner_errors": [],
                    "observed_evidence": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [
                        {
                            "tool": "tavily_search",
                            "route": "docs",
                            "status": "error",
                            "message": "invoke failed",
                            "error_code": "RETRIEVAL_DOCS_TIMEOUT",
                            "query": "docs query",
                            "attempt": 1,
                        }
                    ],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )

        result = _run_single_case(
            run_id="run-error-codes",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=self.fixtures_path,
            case=self.case,
            timeout_seconds=5,
            judge=_DummyJudge((None, None, None)),
            config=self.config,
        )

        self.assertEqual(
            result.error_codes,
            [
                "RETRIEVAL_DOCS_TIMEOUT",
                "LOCAL_RAG_FAILED",
                "VALIDATION_UNSUPPORTED_CLAIMS",
            ],
        )
        self.assertEqual(
            result.validation_events,
            ["validate_evidence: retry_reason=unsupported_claims"],
        )
        self.assertEqual(result.edge_decisions[0]["decision"], "retrieve")
        self.assertEqual(result.retrieval_diagnostics[0].error_code, "RETRIEVAL_DOCS_TIMEOUT")
        self.assertEqual(result.output_tokens, 21)
        self.assertEqual(result.section_count, 2)
        analysis = build_analysis(case_map={self.case.case_id: self.case}, results=[result])
        self.assertEqual(
            {item.error_code for item in analysis.error_code_histogram},
            {
                "RETRIEVAL_DOCS_TIMEOUT",
                "LOCAL_RAG_FAILED",
                "VALIDATION_UNSUPPORTED_CLAIMS",
            },
        )


if __name__ == "__main__":
    unittest.main()
