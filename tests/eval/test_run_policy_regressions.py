import unittest

from src.eval.reporting import build_summary
from src.eval.schemas import BenchmarkCase, BenchmarkConfig, CaseResult


def _case(case_id: str, category: str) -> BenchmarkCase:
    expected_tools = {
        "docs_only": ["tavily_search"],
        "hybrid": ["tavily_search", "upload_search"],
        "rag_only": ["rag_search"],
        "tool_action": ["save_text"],
    }[category]
    return BenchmarkCase(case_id=case_id, category=category, query=f"{case_id} query", expected_tools=expected_tools)


def _result(
    *,
    case: BenchmarkCase,
    release_pass: bool,
    product_pass: bool | None = None,
    judge_pass: bool | None = True,
    composite_quality_score: float = 0.85,
    rule_score_total: float = 0.85,
    llm_judge_score: float | None = 0.85,
    synthesis_mode: str | None = "structured_only",
    judge_input_complete: bool | None = True,
) -> CaseResult:
    product_value = product_pass if product_pass is not None else release_pass
    return CaseResult.model_validate(
        {
            "run_id": "run-policy",
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "session_id": f"session-{case.case_id}",
            "endpoint": "http://localhost:8000/agent",
            "request_payload": {"query": case.query},
            "request_id": f"req-{case.case_id}",
            "http_status": 200,
            "response_text": "응답",
            "response_payload": {"answer": "응답", "claims": [], "evidence": []},
            "tool_calls": list(case.expected_tools),
            "tool_call_count": len(case.expected_tools),
            "effective_weights": {
                "answer_quality": 0.2,
                "groundedness": 0.2,
                "citation_traceability": 0.2,
                "tool_choice": 0.15,
                "format_language": 0.05,
                "llm_judge": 0.2,
            },
            "rule_scores": {
                "answer_quality": 1.0,
                "groundedness": 1.0,
                "citation_traceability": 1.0,
                "tool_choice": 1.0,
                "format_language": 1.0,
            },
            "rule_score_total": rule_score_total,
            "llm_judge_score": llm_judge_score,
            "judge_input_complete": judge_input_complete,
            "judge_gate_passed": judge_pass,
            "judge_pass": judge_pass,
            "product_pass": product_value,
            "release_pass": release_pass,
            "composite_quality_score": composite_quality_score,
            "synthesis_mode": synthesis_mode,
            "cost_usd": 0.0002,
            "llm_calls": [{"stage": "synthesis", "attempt": 1, "path": "structured", "response_metadata": {}, "usage_metadata": {}}],
            "created_at_utc": "2026-04-02T00:00:00+00:00",
        }
    )


class RunPolicyRegressionTest(unittest.TestCase):
    def test_docs_only_pass_floor_is_visible_in_category_rates(self) -> None:
        docs_a = _case("docs-a", "docs_only")
        docs_b = _case("docs-b", "docs_only")
        summary = build_summary(
            run_id="run-policy",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            config=BenchmarkConfig(),
            cases=[docs_a, docs_b],
            results=[
                _result(case=docs_a, release_pass=True),
                _result(case=docs_b, release_pass=False, composite_quality_score=0.6, rule_score_total=0.6, llm_judge_score=0.6),
            ],
        )
        category_rates = {row.category: row.pass_rate for row in summary.analysis.category_pass_rates}
        self.assertEqual(category_rates["docs_only"], 0.5)

    def test_hybrid_pass_floor_is_visible_in_category_rates(self) -> None:
        hybrid_a = _case("hybrid-a", "hybrid")
        hybrid_b = _case("hybrid-b", "hybrid")
        hybrid_c = _case("hybrid-c", "hybrid")
        summary = build_summary(
            run_id="run-policy",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            config=BenchmarkConfig(),
            cases=[hybrid_a, hybrid_b, hybrid_c],
            results=[
                _result(case=hybrid_a, release_pass=True),
                _result(case=hybrid_b, release_pass=True),
                _result(case=hybrid_c, release_pass=False, composite_quality_score=0.65, rule_score_total=0.65, llm_judge_score=0.65),
            ],
        )
        category_rates = {row.category: row.pass_rate for row in summary.analysis.category_pass_rates}
        self.assertEqual(category_rates["hybrid"], 0.6667)

    def test_high_rule_low_judge_divergence_audit_has_run_level_ceiling(self) -> None:
        case_a = _case("docs-a", "docs_only")
        case_b = _case("docs-b", "docs_only")
        summary = build_summary(
            run_id="run-policy",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            config=BenchmarkConfig(),
            cases=[case_a, case_b],
            results=[
                _result(case=case_a, release_pass=True, rule_score_total=0.95, llm_judge_score=0.4),
                _result(case=case_b, release_pass=True, rule_score_total=0.9, llm_judge_score=0.85),
            ],
        )
        divergence_gate = next(gate for gate in summary.gates if gate.name == "high_rule_low_judge_divergence_rate")
        self.assertEqual(divergence_gate.actual, 0.5)
        self.assertFalse(divergence_gate.passed)

    def test_deterministic_direct_usage_audit_tracks_run_level_rate(self) -> None:
        case_a = _case("rag-a", "rag_only")
        case_b = _case("rag-b", "rag_only")
        summary = build_summary(
            run_id="run-policy",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            config=BenchmarkConfig(),
            cases=[case_a, case_b],
            results=[
                _result(case=case_a, release_pass=True, synthesis_mode="deterministic_grounded_direct"),
                _result(case=case_b, release_pass=True, synthesis_mode="structured_only"),
            ],
        )
        usage_gate = next(gate for gate in summary.gates if gate.name == "deterministic_direct_usage_rate")
        self.assertEqual(usage_gate.actual, 0.5)
        self.assertFalse(usage_gate.passed)

    def test_judge_input_completeness_audit_tracks_missing_case_inputs(self) -> None:
        case_a = _case("docs-a", "docs_only")
        case_b = _case("docs-b", "docs_only")
        summary = build_summary(
            run_id="run-policy",
            endpoint="http://localhost:8000",
            fixtures_path="data/benchmarks/fixtures/cases.generated.jsonl",
            config_path="data/benchmarks/config.toml",
            config=BenchmarkConfig(),
            cases=[case_a, case_b],
            results=[
                _result(case=case_a, release_pass=True, judge_input_complete=True),
                _result(case=case_b, release_pass=True, judge_input_complete=False, judge_pass=False),
            ],
        )
        completeness_gate = next(gate for gate in summary.gates if gate.name == "judge_input_completeness_rate")
        self.assertEqual(completeness_gate.actual, 0.5)
        self.assertFalse(completeness_gate.passed)


if __name__ == "__main__":
    unittest.main()
