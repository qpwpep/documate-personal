from tests.eval.response_fixtures import answer_provenance, sse_http_response
from src.core.contracts.debug import DEBUG_SCHEMA_VERSION
from tests.eval.response_fixtures import source_hit
from tests.eval.response_fixtures import comparison_response, plain_response
import json
import unittest

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from src.eval.config_models import BenchmarkCase, BenchmarkConfig, BenchmarkLiveSlackConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner import _run_single_case, run_online_benchmark


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")


class _JudgeModelBoundary:
    def __init__(self) -> None:
        self.payloads = []

    def invoke(self, messages):
        self.payloads.append(json.loads(messages[-1].content))
        return SimpleNamespace(content=json.dumps({
            "score": 0.8,
            "reason": "ok",
            "subscores": {key: 0.8 for key in (
                "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language"
            )},
        }))


class RunnerRequestPayloadTest(unittest.TestCase):
    @patch("src.app.client.requests.post")
    def test_slack_destination_fields_are_forwarded(self, mock_post) -> None:
        mock_post.return_value = sse_http_response(
            200,
            {
                "response": plain_response('shared'),
                "trace": "trace-id",
                "debug": {
                    "answer_provenance": answer_provenance(plain_response('shared')),
                    "schema_version": DEBUG_SCHEMA_VERSION,
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
                    "observed_hits": [],
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
            judge=LLMJudge(model_name="test-model", enabled=False),
            config=BenchmarkConfig(),
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["slack_channel_id"], "C123BENCH")
        self.assertEqual(payload["slack_user_id"], "U123BENCH")
        self.assertEqual(payload["slack_email"], "bench@example.com")

    @patch("src.app.client.requests.post")
    def test_live_slack_channel_uses_configured_destination_and_records_delivery(self, mock_post) -> None:
        response = {
            **plain_response("shared"),
            "actions": [{"kind": "slack_notify", "status": "success", "target": "C999LIVE"}],
        }
        mock_post.return_value = sse_http_response(
            200,
            {
                "response": response,
                "trace": "trace-id",
                "debug": {
                    "answer_provenance": answer_provenance(response),
                    "schema_version": DEBUG_SCHEMA_VERSION,
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
                    "observed_hits": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )
        case = BenchmarkCase(
            case_id="tool_seed_channel_live",
            category="tool_action",
            query="share this to slack",
            expected_tools=["slack_notify"],
            slack_channel_id="C123BENCH",
        )

        result = _run_single_case(
            run_id="run-slack-live-channel",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=case,
            timeout_seconds=5,
            judge=LLMJudge(model_name="test-model", enabled=False),
            config=BenchmarkConfig(),
            live_slack=BenchmarkLiveSlackConfig(enabled=True, channel_id="C999LIVE"),
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["slack_channel_id"], "C999LIVE")
        self.assertNotIn("slack_user_id", payload)
        self.assertNotIn("slack_email", payload)
        self.assertEqual(len(result.actions), 1)
        self.assertEqual(result.actions[0].status, "success")
        self.assertEqual(result.actions[0].target, "C999LIVE")
        self.assertTrue(result.slack_delivery_required)
        self.assertEqual(result.slack_delivery_status, "success")

    @patch("src.app.client.requests.post")
    def test_live_slack_dm_cases_use_live_dm_destination(self, mock_post) -> None:
        mock_post.return_value = sse_http_response(
            200,
            {
                "response": plain_response('shared'),
                "trace": "trace-id",
                "debug": {
                    "answer_provenance": answer_provenance(plain_response('shared')),
                    "schema_version": DEBUG_SCHEMA_VERSION,
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
                    "observed_hits": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,
                },
            },
        )
        case = BenchmarkCase(
            case_id="tool_seed_dm_live",
            category="tool_action",
            query="share this to slack dm",
            expected_tools=["slack_notify"],
            slack_user_id="U123BENCH",
        )

        _run_single_case(
            run_id="run-slack-live-dm",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=case,
            timeout_seconds=5,
            judge=LLMJudge(model_name="test-model", enabled=False),
            config=BenchmarkConfig(),
            live_slack=BenchmarkLiveSlackConfig(enabled=True, user_id="U999LIVE"),
        )

        _, kwargs = mock_post.call_args
        payload = kwargs["json"]
        self.assertEqual(payload["slack_user_id"], "U999LIVE")
        self.assertNotIn("slack_channel_id", payload)
        self.assertNotIn("slack_email", payload)

    def test_live_slack_requires_destination_before_requests(self) -> None:
        case = BenchmarkCase(
            case_id="tool_seed_live_missing",
            category="tool_action",
            query="share this to slack",
            expected_tools=["slack_notify"],
            slack_channel_id="C123BENCH",
        )

        with TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "cases.jsonl"
            fixture.write_bytes((case.model_dump_json() + "\n").encode("utf-8"))
            with self.assertRaisesRegex(ValueError, "tool_seed_live_missing"):
                run_online_benchmark(
                    fixtures_path=fixture,
                    endpoint="http://127.0.0.1:8000",
                    config=BenchmarkConfig(),
                    config_path=Path("data/benchmarks/config.toml"),
                    output_root=Path(temp_dir),
                    track="release",
                    live_slack=BenchmarkLiveSlackConfig(enabled=True),
                )

    @patch("src.app.client.requests.post")
    def test_planner_errors_are_parsed_from_debug_payload(self, mock_post) -> None:
        mock_post.return_value = sse_http_response(
            200,
            {
                "response": plain_response('shared'),
                "trace": "trace-id",
                "debug": {
                    "answer_provenance": answer_provenance(plain_response('shared')),
                    "schema_version": DEBUG_SCHEMA_VERSION,
                    "observability_status": "ok",
                    "missing_required_debug_fields": [],
                    "tool_calls": ["tavily_search"],
                    "tool_call_count": 1,
                    "token_usage": {},
                    "model_name": None,
                    "models_used": [],
                    "llm_calls": [],
                    "errors": [],
                    "observed_hits": [],
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
            judge=LLMJudge(model_name="test-model", enabled=False),
            config=BenchmarkConfig(),
        )

        self.assertEqual(
            result.planner_errors,
            ["planner: structured output invocation failed (boom)"],
        )

    @patch("src.app.client.requests.post")
    def test_judge_payload_includes_structured_fields(self, mock_post) -> None:
        mock_post.return_value = sse_http_response(
            200,
            {
                "response": comparison_response(),
                "trace": "Session ID: abc, Request ID: req123, Agent ID: 1",
                "debug": {
                    "answer_provenance": answer_provenance(comparison_response()),
                    "schema_version": DEBUG_SCHEMA_VERSION,
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
                    "observed_hits": [
                        source_hit().model_dump(mode="json"),
                        source_hit(official=False, text="업로드 비교").model_dump(mode="json"),
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
        judge = LLMJudge(model_name="test-model", enabled=False)
        judge.enabled = True
        judge.client = model = _JudgeModelBoundary()

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

        self.assertEqual(len(model.payloads), 1)
        payload = model.payloads[0]
        self.assertEqual(payload["response"], comparison_response())
        self.assertEqual(len(payload["observed_hits"]), 2)
        self.assertEqual(len(payload["retrieval_diagnostics"]), 2)
        self.assertEqual(payload["validator_reason"], "low_score")
        self.assertEqual(payload["synthesis_mode"], "structured_only")
        self.assertEqual(result.judge_errors, [])
        self.assertEqual(result.llm_judge_score, 0.8)
        self.assertTrue(result.judge_input_complete)
        self.assertEqual(result.request_id, "req123")

    @patch("src.app.client.requests.post")
    def test_judge_payload_includes_public_actions_for_live_slack_cases(self, mock_post) -> None:
        mock_post.return_value = sse_http_response(
            200,
            {
                "response": {**plain_response('shared'), "actions": [{'kind': 'slack_notify', 'status': 'error', 'target': 'C999LIVE', 'message': None, 'error': 'channel_not_found'}]},
                "trace": "trace-id",
                "debug": {
                    "answer_provenance": answer_provenance({**plain_response('shared'), "actions": [{'kind': 'slack_notify', 'status': 'error', 'target': 'C999LIVE', 'message': None, 'error': 'channel_not_found'}]}),
                    "schema_version": DEBUG_SCHEMA_VERSION,
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
                    "observed_hits": [],
                    "retry_context": None,
                    "retrieval_diagnostics": [],
                    "planner_diagnostics": None,
                    "latency_breakdown": None,

                },
            },
        )
        judge = LLMJudge(model_name="test-model", enabled=False)
        judge.enabled = True
        judge.client = model = _JudgeModelBoundary()

        result = _run_single_case(
            run_id="run-live-judge-payload",
            endpoint="http://127.0.0.1:8000",
            fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
            case=BenchmarkCase(
                case_id="tool_live_judge",
                category="tool_action",
                query="share this to slack",
                expected_tools=["slack_notify"],
                slack_channel_id="C123BENCH",
            ),
            timeout_seconds=5,
            judge=judge,
            config=BenchmarkConfig(),
            live_slack=BenchmarkLiveSlackConfig(enabled=True, channel_id="C999LIVE"),
        )

        self.assertEqual(len(model.payloads), 1)
        payload = model.payloads[0]
        self.assertTrue(payload["slack_delivery_required"])
        self.assertEqual(payload["response"]["actions"][0]["status"], "error")
        self.assertEqual(payload["response"]["actions"][0]["error"], "channel_not_found")
        self.assertEqual(result.judge_errors, [])
        self.assertEqual(result.slack_delivery_status, "failed")


if __name__ == "__main__":
    unittest.main()
