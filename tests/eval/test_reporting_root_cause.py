import unittest

from src.eval.reporting import build_summary
from src.eval.schemas import BenchmarkCase, BenchmarkConfig, CaseResult


def _make_result(
    *,
    case: BenchmarkCase,
    tool_calls: list[str],
    retrieval_diagnostics: list[dict] | None = None,
    planner_diagnostics: dict | None = None,
    validator_reason: str | None = None,
    validator_feedback: str | None = None,
    llm_judge_reason: str | None = None,
    passed: bool,
    latency_breakdown: dict | None = None,
    final_score: float = 0.4,
) -> CaseResult:
    return CaseResult.model_validate(
        {
            "run_id": "run-root-cause",
            "case_id": case.case_id,
            "category": case.category,
            "scenario": case.scenario,
            "query": case.query,
            "session_id": f"session-{case.case_id}",
            "endpoint": "http://localhost:8000/agent",
            "request_payload": {"query": case.query},
            "http_status": 200,
            "response_text": "response body",
            "response_payload": {"answer": "response body", "evidence": []},
            "evidence": [],
            "observed_evidence": [],
            "retrieval_diagnostics": retrieval_diagnostics or [],
            "planner_diagnostics": planner_diagnostics,
            "latency_ms_e2e": 1200,
            "latency_ms_server": 900,
            "latency_breakdown": latency_breakdown,
            "tool_calls": tool_calls,
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "runtime_errors": [],
            "response_errors": [],
            "judge_errors": [],
            "validator_reason": validator_reason,
            "validator_feedback": validator_feedback,
            "effective_weights": {
                "tool_match": 0.3,
                "content_constraints": 0.25,
                "citation_compliance": 0.2,
                "safety_format": 0.05,
                "llm_judge": 0.2,
            },
            "rule_scores": {
                "tool_match": 1.0,
                "content_constraints": 0.5,
                "citation_compliance": 1.0,
                "safety_format": 1.0,
            },
            "rule_score_total": 0.675,
            "llm_judge_score": 0.0,
            "llm_judge_reason": llm_judge_reason,
            "final_score": final_score,
            "passed": passed,
            "cost_usd": 0.0002,
            "created_at_utc": "2026-03-08T00:00:00+00:00",
        }
    )


class ReportingRootCauseTest(unittest.TestCase):
    def test_build_summary_adds_root_cause_analysis(self) -> None:
        docs_validator_case = BenchmarkCase(
            case_id="docs_validator_case",
            category="docs_only",
            query="fastapi response_model",
            expected_tools=["tavily_search"],
        )
        docs_confused_case = BenchmarkCase(
            case_id="docs_confused_case",
            category="docs_only",
            query="pandas merge",
            expected_tools=["tavily_search"],
            forbidden_tools=["rag_search"],
        )
        hybrid_case = BenchmarkCase(
            case_id="hybrid_case",
            category="hybrid",
            query="compare uploaded notebook with docs",
            expected_tools=["tavily_search", "upload_search"],
        )
        tool_case = BenchmarkCase(
            case_id="tool_case",
            category="tool_action",
            query="save the answer",
            expected_tools=["save_text"],
        )

        results = [
            _make_result(
                case=docs_validator_case,
                tool_calls=["tavily_search"],
                retrieval_diagnostics=[
                    {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "success",
                        "message": "",
                        "query": docs_validator_case.query,
                        "attempt": 1,
                    }
                ],
                planner_diagnostics={
                    "status": "heuristic_fallback",
                    "reason": "planner_failed_or_invalid",
                    "fallback_routes": ["docs"],
                    "intent_required": True,
                    "required_routes": ["docs"],
                    "override_applied": False,
                    "override_reason": None,
                },
                validator_reason="no_evidence",
                validator_feedback="low evidence confidence",
                llm_judge_reason="Assistant failed to answer even though official docs evidence was present.",
                passed=False,
                final_score=0.675,
            ),
            _make_result(
                case=docs_confused_case,
                tool_calls=["rag_search", "save_text"],
                llm_judge_reason=(
                    "Assistant used local retrieval and skipped the official docs path, "
                    "so the answer did not satisfy the request."
                ),
                passed=False,
                final_score=0.375,
            ),
            _make_result(
                case=hybrid_case,
                tool_calls=["tavily_search", "upload_search"],
                retrieval_diagnostics=[
                    {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "success",
                        "message": "",
                        "query": hybrid_case.query,
                        "attempt": 1,
                    },
                    {
                        "tool": "upload_search",
                        "route": "upload",
                        "status": "success",
                        "message": "",
                        "query": hybrid_case.query,
                        "attempt": 1,
                    },
                ],
                planner_diagnostics={
                    "status": "heuristic_fallback",
                    "reason": "planner_failed_or_invalid",
                    "fallback_routes": ["docs", "upload"],
                    "intent_required": True,
                    "required_routes": ["docs", "upload"],
                    "override_applied": False,
                    "override_reason": None,
                },
                validator_reason="low_score",
                llm_judge_reason="Answer stayed too generic and did not compare the uploaded context to docs.",
                passed=False,
                latency_breakdown={
                    "server_total_ms": 900,
                    "graph_total_ms": 850,
                    "upload_retriever_build_ms": 35,
                    "stage_totals_ms": {
                        "summarize_ms": 10,
                        "planner_ms": 15,
                        "retrieval_total_ms": 400,
                        "synthesis_total_ms": 220,
                        "validation_ms": 60,
                        "action_postprocess_ms": 20,
                    },
                    "stage_attempts": [],
                    "retrieval_routes": [],
                    "synthesis_attempts": [],
                },
                final_score=0.595,
            ),
            _make_result(
                case=tool_case,
                tool_calls=["save_text"],
                passed=True,
                final_score=0.9,
            ),
        ]

        summary = build_summary(
            run_id="run-root-cause",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            config=BenchmarkConfig(),
            cases=[docs_validator_case, docs_confused_case, hybrid_case, tool_case],
            results=results,
        )

        self.assertIsNotNone(summary.analysis)

        category_rates = {row.category: row for row in summary.analysis.category_pass_rates}
        self.assertEqual(category_rates["docs_only"].passed_cases, 0)
        self.assertEqual(category_rates["docs_only"].total_cases, 2)
        self.assertEqual(category_rates["docs_only"].pass_rate, 0.0)
        self.assertEqual(category_rates["tool_action"].pass_rate, 1.0)

        planner_buckets = {
            (row.category, row.status, row.reason, row.override_reason): row.count
            for row in summary.analysis.planner_diagnostics_histogram
        }
        self.assertEqual(
            planner_buckets[("docs_only", "heuristic_fallback", "planner_failed_or_invalid", None)],
            1,
        )
        self.assertEqual(
            planner_buckets[("docs_only", "missing", "diagnostics_unavailable", None)],
            1,
        )

        retrieval_buckets = {
            (row.category, row.route, row.status): row.count
            for row in summary.analysis.retrieval_route_status_histogram
        }
        self.assertEqual(retrieval_buckets[("docs_only", "docs", "success")], 1)
        self.assertEqual(retrieval_buckets[("hybrid", "upload", "success")], 1)

        self.assertEqual(len(summary.analysis.route_confusion), 1)
        route_confusion = summary.analysis.route_confusion[0]
        self.assertEqual(route_confusion.category, "docs_only")
        self.assertEqual(route_confusion.expected_routes, ["docs"])
        self.assertEqual(route_confusion.observed_routes, ["local"])
        self.assertEqual(route_confusion.missing_expected_routes, ["docs"])
        self.assertEqual(route_confusion.unexpected_routes, [])
        self.assertEqual(route_confusion.forbidden_routes, ["local"])
        self.assertEqual(route_confusion.count, 1)

        validator_buckets = {
            (row.category, row.reason): (row.count, row.share)
            for row in summary.analysis.validator_reason_histogram
        }
        self.assertEqual(validator_buckets[("docs_only", "no_evidence")], (1, 0.5))
        self.assertEqual(validator_buckets[("docs_only", "missing")], (1, 0.5))
        self.assertEqual(validator_buckets[("hybrid", "low_score")], (1, 1.0))

        self.assertIsNotNone(summary.analysis.latency_breakdown_coverage)
        self.assertEqual(summary.analysis.latency_breakdown_coverage.available_cases, 1)
        self.assertEqual(summary.analysis.latency_breakdown_coverage.total_cases, 4)
        self.assertEqual(summary.analysis.latency_breakdown_coverage.coverage_rate, 0.25)

        stage_rows = {row.stage: row for row in summary.analysis.stage_latency_percentiles}
        self.assertEqual(stage_rows["upload_retriever_build_ms"].sample_count, 1)
        self.assertEqual(stage_rows["upload_retriever_build_ms"].p50_latency_ms, 35.0)
        self.assertEqual(stage_rows["retrieval_total_ms"].p95_latency_ms, 400.0)

        failure_reasons = {item["case_id"]: item["reason"] for item in summary.metrics.failures}
        self.assertEqual(failure_reasons["docs_validator_case"], "validator:no_evidence")
        self.assertIn(
            "Assistant used local retrieval and skipped the official docs path",
            failure_reasons["docs_confused_case"],
        )
        self.assertNotIn("save_text", str(summary.analysis.route_confusion))


if __name__ == "__main__":
    unittest.main()
