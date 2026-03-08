import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.eval.reporting import build_markdown_report
from src.eval.runner_online import _run_single_case
from src.eval.schemas import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    GateResult,
    RunSummary,
    SummaryStats,
)


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


class LatencyBreakdownReportingTest(unittest.TestCase):
    @patch("src.eval.runner_online.requests.post")
    def test_run_single_case_parses_latency_breakdown(self, mock_post) -> None:
        mock_post.return_value = _FakeResponse(
            200,
            {
                "response": {"answer": "done", "claims": [], "evidence": [], "confidence": None},
                "trace": "trace-id",
                "file_path": "",
                "debug": {
                    "tool_calls": ["tavily_search"],
                    "token_usage": {},
                    "observed_evidence": [],
                    "latency_breakdown": {
                        "server_total_ms": 1500,
                        "graph_total_ms": 1400,
                        "upload_retriever_build_ms": None,
                        "stage_totals_ms": {
                            "summarize_ms": 0,
                            "planner_ms": 30,
                            "retrieval_total_ms": 900,
                            "synthesis_total_ms": 350,
                            "validation_ms": 60,
                            "action_postprocess_ms": 20,
                        },
                        "stage_attempts": [],
                        "retrieval_routes": [],
                        "synthesis_attempts": [],
                    },
                },
            },
        )

        result = _run_single_case(
            run_id="run-latency",
            endpoint="http://localhost:8000",
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

        self.assertIsNotNone(result.latency_breakdown)
        self.assertEqual(result.latency_breakdown.graph_total_ms, 1400)
        self.assertEqual(result.latency_breakdown.stage_totals_ms.retrieval_total_ms, 900)

    def test_build_markdown_report_includes_latency_breakdown_section(self) -> None:
        summary = RunSummary(
            run_id="20260308_000000",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            generated_at_utc="2026-03-08T00:00:00+00:00",
            metrics=SummaryStats(
                total_cases=1,
                scored_cases=1,
                passed_cases=1,
                pass_rate=1.0,
                tool_precision=1.0,
                tool_recall=1.0,
                citation_compliance=1.0,
                p50_latency_ms=1000.0,
                p95_latency_ms=1000.0,
                avg_cost_per_case_usd=0.0001,
                failures=[],
            ),
            gates=[
                GateResult(name="pass_rate", threshold=0.82, actual=1.0, passed=True),
                GateResult(name="tool_precision", threshold=0.9, actual=1.0, passed=True),
                GateResult(name="tool_recall", threshold=0.85, actual=1.0, passed=True),
                GateResult(name="citation_compliance", threshold=0.88, actual=1.0, passed=True),
                GateResult(name="p95_latency_ms", threshold=20000, actual=1000.0, passed=True),
                GateResult(name="avg_cost_per_case_usd", threshold=0.035, actual=0.0001, passed=True),
            ],
            overall_passed=True,
            weights={
                "tool_match": 0.3,
                "content_constraints": 0.25,
                "citation_compliance": 0.2,
                "safety_format": 0.05,
                "llm_judge": 0.2,
            },
            hard_gates={
                "pass_rate": 0.82,
                "tool_precision": 0.9,
                "tool_recall": 0.85,
                "citation_compliance": 0.88,
                "p95_latency_ms": 20000,
                "avg_cost_per_case_usd": 0.035,
            },
            pricing={"prompt_per_1k_usd": 0.00015, "completion_per_1k_usd": 0.0006},
            judge_enabled=True,
            judge_model="gpt-5-mini",
        )
        results = [
            CaseResult.model_validate(
                {
                    "run_id": "20260308_000000",
                    "case_id": "docs_only_seed_mutation_001",
                    "category": "docs_only",
                    "scenario": "seed_mutation",
                    "query": "numpy docs",
                    "session_id": "session-1",
                    "endpoint": "http://localhost:8000/agent",
                    "request_payload": {"query": "numpy docs"},
                    "http_status": 200,
                    "response_text": "done",
                    "latency_ms_e2e": 1000,
                    "latency_ms_server": 950,
                    "tool_calls": ["tavily_search"],
                    "runtime_errors": [],
                    "response_errors": [],
                    "judge_errors": [],
                    "effective_weights": {},
                    "rule_scores": {},
                    "created_at_utc": "2026-03-08T00:00:00+00:00",
                    "latency_breakdown": {
                        "server_total_ms": 950,
                        "graph_total_ms": 930,
                        "upload_retriever_build_ms": None,
                        "stage_totals_ms": {
                            "summarize_ms": 0,
                            "planner_ms": 20,
                            "retrieval_total_ms": 600,
                            "synthesis_total_ms": 250,
                            "validation_ms": 40,
                            "action_postprocess_ms": 20,
                        },
                        "stage_attempts": [],
                        "retrieval_routes": [],
                        "synthesis_attempts": [],
                    },
                }
            )
        ]

        report = build_markdown_report(summary, results)

        self.assertIn("## Latency Breakdown", report)
        self.assertIn("| retrieval_total_ms |", report)
        self.assertIn("### Slow Cases (Top 10)", report)
        self.assertIn("docs_only_seed_mutation_001", report)


if __name__ == "__main__":
    unittest.main()
