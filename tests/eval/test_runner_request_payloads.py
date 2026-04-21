import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.online_runner import _run_single_case
from src.eval.schemas import BenchmarkCase, BenchmarkConfig


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


class _DummyJudge:
    def score_case(self, case, response_text, tool_calls):
        _ = (case, response_text, tool_calls)
        return (None, None, None)


class _CaptureJudge:
    def __init__(self) -> None:
        self.kwargs = None

    def score_case(self, case, response_text, tool_calls, **kwargs):
        _ = (case, response_text, tool_calls)
        self.kwargs = kwargs
        return (0.8, "ok", None, None)


class RunnerRequestPayloadTest(unittest.TestCase):
    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_slack_destination_fields_are_forwarded(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "shared", "evidence": []},
                "trace": "trace-id",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["slack_notify"],
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
        case = BenchmarkCase(
            case_id="tool_seed_999",
            category="tool_action",
            query="share this to slack",
            expected_tools=["slack_notify"],
            slack_channel_id="C123BENCH",
            slack_user_id="U123BENCH",
            slack_email="bench@example.com",
        )

        _run_single_case(
            run_id="run-slack-payload",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=case,
            timeout_seconds=5,
            judge=_DummyJudge(),
            config=BenchmarkConfig(),
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["slack_channel_id"], "C123BENCH")
        self.assertEqual(payload["slack_user_id"], "U123BENCH")
        self.assertEqual(payload["slack_email"], "bench@example.com")

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_planner_errors_are_parsed_from_debug_payload(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "shared", "evidence": []},
                "trace": "trace-id",
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
                    "observed_evidence": [],
                    "planner_errors": ["planner: structured output invocation failed (boom)"],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )

        result = _run_single_case(
            run_id="run-planner-errors",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=BenchmarkCase(
                case_id="docs_only_seed_mutation_001",
                category="docs_only",
                query="numpy docs",
                expected_tools=["tavily_search"],
            ),
            timeout_seconds=5,
            judge=_DummyJudge(),
            config=BenchmarkConfig(),
        )

        self.assertEqual(
            result.planner_errors,
            ["planner: structured output invocation failed (boom)"],
        )

    @patch("src.eval.online_runner.case_runner.requests.post")
    def test_judge_payload_includes_structured_fields(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {
                    "answer": "공식 설명 [1] 업로드 비교 [2]",
                    "claims": [
                        {"text": "공식 설명", "evidence_ids": ["url:https://numpy.org/doc/stable/"]},
                        {"text": "업로드 비교", "evidence_ids": ["path:uploads/demo/sample.ipynb#cell=0;chunk=0;start=0;end=12"]},
                    ],
                    "sections": [
                        {"kind": "official_docs", "heading": "공식 문서", "body": "공식 설명"},
                        {"kind": "comparison", "heading": "비교", "body": "업로드 비교"},
                    ],
                    "evidence": [
                        {
                            "kind": "official",
                            "tool": "tavily_search",
                            "source_id": "url:https://numpy.org/doc/stable/",
                            "document_id": "url:https://numpy.org/doc/stable/",
                            "url_or_path": "https://numpy.org/doc/stable/",
                            "title": "NumPy Docs",
                            "snippet": "official snippet",
                            "score": 0.9,
                        },
                        {
                            "kind": "local",
                            "tool": "upload_search",
                            "source_id": "path:uploads/demo/sample.ipynb#cell=0;chunk=0;start=0;end=12",
                            "document_id": "path:uploads/demo/sample.ipynb",
                            "url_or_path": "uploads/demo/sample.ipynb",
                            "title": "Notebook",
                            "snippet": "local snippet",
                            "score": 0.8,
                            "cell_id": 0,
                            "chunk_id": 0,
                            "start_offset": 0,
                            "end_offset": 12,
                        },
                    ],
                },
                "trace": "Session ID: abc, Request ID: req123, Agent ID: 1",
                "file_path": "",
                "debug": {
                    "schema_version": 3,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search", "upload_search"],
                    "tool_call_count": 2,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "planner_errors": [],
                    "observed_evidence": [
                        {
                            "kind": "official",
                            "tool": "tavily_search",
                            "source_id": "url:https://numpy.org/doc/stable/",
                            "document_id": "url:https://numpy.org/doc/stable/",
                            "url_or_path": "https://numpy.org/doc/stable/",
                            "title": "NumPy Docs",
                            "snippet": "official snippet",
                            "score": 0.9,
                        }
                    ],
                    "retry_context": {"retry_reason": "low_score", "retrieval_feedback": "compare more explicitly"},
                    "retrieval_diagnostics": [
                        {"tool": "tavily_search", "route": "docs", "status": "success", "message": "", "query": "numpy docs", "attempt": 1},
                        {"tool": "upload_search", "route": "upload", "status": "success", "message": "", "query": "numpy docs", "attempt": 1},
                    ],
                    "planner_diagnostics": {
                        "status": "heuristic_fallback",
                        "reason": "planner_failed_or_invalid",
                        "fallback_routes": ["docs", "upload"],
                        "intent_required": True,
                        "required_routes": ["docs", "upload"],
                        "override_applied": False,
                        "override_reason": None,
                    },
                    "latency_breakdown": {
                        "server_total_ms": 100,
                        "graph_total_ms": 90,
                        "upload_retriever_build_ms": 10,
                        "stage_totals_ms": {
                            "summarize_ms": 0,
                            "planner_ms": 5,
                            "retrieval_total_ms": 40,
                            "synthesis_total_ms": 20,
                            "validation_ms": 15,
                            "action_postprocess_ms": 10,
                        },
                        "stage_attempts": [],
                        "retrieval_routes": [],
                        "synthesis_attempts": [
                            {
                                "attempt": 1,
                                "mode": "structured_only",
                                "structured_ms": 20,
                                "fallback_ms": None,
                                "total_ms": 20,
                            }
                        ],
                    },
                },
            },
        )
        judge = _CaptureJudge()

        result = _run_single_case(
            run_id="run-judge-payload",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=BenchmarkCase(
                case_id="hybrid_seed_001",
                category="hybrid",
                query="numpy docs와 업로드 비교",
                expected_tools=["tavily_search", "upload_search"],
            ),
            timeout_seconds=5,
            judge=judge,
            config=BenchmarkConfig(),
        )

        self.assertIsNotNone(judge.kwargs)
        self.assertEqual(len(judge.kwargs["claims"]), 2)
        self.assertEqual(len(judge.kwargs["response_evidence"]), 2)
        self.assertEqual(len(judge.kwargs["sections"]), 2)
        self.assertEqual(judge.kwargs["sections"][0].kind, "official_docs")
        self.assertEqual(len(judge.kwargs["observed_evidence"]), 1)
        self.assertEqual(len(judge.kwargs["retrieval_diagnostics"]), 2)
        self.assertEqual(judge.kwargs["validator_reason"], "low_score")
        self.assertEqual(judge.kwargs["synthesis_mode"], "structured_only")
        self.assertTrue(result.judge_input_complete)
        self.assertEqual(result.request_id, "req123")


if __name__ == "__main__":
    unittest.main()
