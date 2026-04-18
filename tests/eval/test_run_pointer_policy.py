import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.eval.main import resolve_run_track, validate_history_targets
from src.eval.online_runner import latest_run_pointer_path, run_online_benchmark
from src.eval.schemas import BenchmarkCase, BenchmarkConfig, CaseResult, dump_jsonl


def _case(case_id: str) -> BenchmarkCase:
    return BenchmarkCase(
        case_id=case_id,
        category="docs_only",
        query=f"{case_id} query",
        expected_tools=["tavily_search"],
    )


def _result(run_id: str, case: BenchmarkCase) -> CaseResult:
    return CaseResult.model_validate(
        {
            "run_id": run_id,
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "session_id": f"session-{case.case_id}",
            "endpoint": "http://localhost:8000/agent",
            "request_payload": {"query": case.query},
            "http_status": 200,
            "response_text": "ok",
            "response_payload": {"answer": "ok", "claims": [], "evidence": []},
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
            "rule_score_total": 1.0,
            "judge_gate_passed": None,
            "judge_pass": None,
            "product_pass": True,
            "release_pass": True,
            "composite_quality_score": 1.0,
            "synthesis_mode": "structured_only",
            "cost_usd": 0.0002,
            "llm_calls": [{"stage": "synthesis", "attempt": 1, "path": "structured", "response_metadata": {}, "usage_metadata": {}}],
            "latency_ms_e2e": 1000,
            "created_at_utc": "2026-04-02T00:00:00+00:00",
        }
    )


class RunPointerPolicyTest(unittest.TestCase):
    def test_resolve_run_track_defaults_limit_runs_to_smoke(self) -> None:
        self.assertEqual(resolve_run_track(None, 1), "smoke")
        self.assertEqual(resolve_run_track(None, None), "release")
        self.assertEqual(resolve_run_track("release", 1), "release")

    def test_validate_history_targets_rejects_smoke_on_release_outputs(self) -> None:
        with self.assertRaises(ValueError):
            validate_history_targets("smoke", Path("README.md"), Path("docs/assets/benchmark_history.svg"))

    @patch("src.eval.online_runner.case_runner._run_single_case")
    def test_limit_run_updates_smoke_pointer_only(self, mock_run_single_case) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixtures_path = root / "cases.jsonl"
            output_root = root / "output" / "benchmarks"
            output_root.mkdir(parents=True)
            release_pointer = latest_run_pointer_path(output_root, "release")
            release_pointer.write_text("release-prev\n", encoding="utf-8")

            case = _case("docs_only_001")
            dump_jsonl(fixtures_path, [case])
            mock_run_single_case.return_value = _result("unused-run-id", case)

            run_dir, _, summary = run_online_benchmark(
                fixtures_path=fixtures_path,
                endpoint="http://localhost:8000",
                config=BenchmarkConfig(judge_enabled=False),
                config_path=Path("data/benchmarks/config.toml"),
                output_root=output_root,
                track="smoke",
                limit=1,
            )

            smoke_pointer = latest_run_pointer_path(output_root, "smoke")
            payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary.track, "smoke")
            self.assertEqual(summary.requested_limit, 1)
            self.assertTrue(smoke_pointer.exists())
            self.assertEqual(smoke_pointer.read_text(encoding="utf-8").strip(), summary.run_id)
            self.assertEqual(release_pointer.read_text(encoding="utf-8").strip(), "release-prev")
            self.assertEqual(payload["track"], "smoke")
            self.assertEqual(payload["requested_limit"], 1)

    @patch("src.eval.online_runner.case_runner._run_single_case")
    def test_release_run_updates_release_pointer(self, mock_run_single_case) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fixtures_path = root / "cases.jsonl"
            output_root = root / "output" / "benchmarks"
            output_root.mkdir(parents=True)

            case = _case("docs_only_001")
            dump_jsonl(fixtures_path, [case])
            mock_run_single_case.return_value = _result("unused-run-id", case)

            run_dir, _, summary = run_online_benchmark(
                fixtures_path=fixtures_path,
                endpoint="http://localhost:8000",
                config=BenchmarkConfig(judge_enabled=False),
                config_path=Path("data/benchmarks/config.toml"),
                output_root=output_root,
                track="release",
                limit=None,
            )

            release_pointer = latest_run_pointer_path(output_root, "release")
            payload = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))

            self.assertEqual(summary.track, "release")
            self.assertIsNone(summary.requested_limit)
            self.assertTrue(release_pointer.exists())
            self.assertEqual(release_pointer.read_text(encoding="utf-8").strip(), summary.run_id)
            self.assertEqual(payload["track"], "release")
            self.assertIsNone(payload["requested_limit"])


if __name__ == "__main__":
    unittest.main()
