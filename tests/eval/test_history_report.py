import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.eval.history_loader import load_history_runs, select_comparable_runs
from src.eval.history_report import refresh_history_report


def _summary_payload(
    *,
    run_id: str,
    generated_at_utc: str,
    fixtures_path: str = "data\\benchmarks\\fixtures\\cases.generated.jsonl",
    total_cases: int = 120,
    track: str = "release",
    requested_limit: int | None = None,
    pass_rate: float,
    tool_precision: float,
    tool_recall: float,
    citation_compliance: float,
    p50_latency_ms: float,
    p95_latency_ms: float,
    avg_cost_per_case_usd: float,
) -> dict:
    return {
        "run_id": run_id,
        "endpoint": "http://127.0.0.1:8000",
        "fixtures_path": fixtures_path,
        "config_path": "data\\benchmarks\\config.toml",
        "generated_at_utc": generated_at_utc,
        "mode": "online",
        "track": track,
        "requested_limit": requested_limit,
        "metrics": {
            "total_cases": total_cases,
            "scored_cases": total_cases,
            "passed_cases": max(1, int(total_cases * pass_rate)),
            "pass_rate": pass_rate,
            "tool_precision": tool_precision,
            "tool_recall": tool_recall,
            "citation_compliance": citation_compliance,
            "p50_latency_ms": p50_latency_ms,
            "p95_latency_ms": p95_latency_ms,
            "avg_cost_per_case_usd": avg_cost_per_case_usd,
            "failures": [],
        },
        "gates": [
            {
                "name": "pass_rate",
                "threshold": 0.82,
                "actual": pass_rate,
                "passed": pass_rate >= 0.82,
            },
            {
                "name": "tool_precision",
                "threshold": 0.90,
                "actual": tool_precision,
                "passed": tool_precision >= 0.90,
            },
            {
                "name": "tool_recall",
                "threshold": 0.85,
                "actual": tool_recall,
                "passed": tool_recall >= 0.85,
            },
            {
                "name": "citation_compliance",
                "threshold": 0.88,
                "actual": citation_compliance,
                "passed": citation_compliance >= 0.88,
            },
            {
                "name": "p95_latency_ms",
                "threshold": 20000,
                "actual": p95_latency_ms,
                "passed": p95_latency_ms <= 20000,
            },
            {
                "name": "avg_cost_per_case_usd",
            "threshold": 0.01,
                "actual": avg_cost_per_case_usd,
            "passed": avg_cost_per_case_usd <= 0.01,
            },
        ],
        "overall_passed": False,
        "weights": {
            "tool_match": 0.3,
            "content_constraints": 0.25,
            "citation_compliance": 0.2,
            "safety_format": 0.05,
            "llm_judge": 0.2,
        },
        "hard_gates": {
            "pass_rate": 0.82,
            "tool_precision": 0.9,
            "tool_recall": 0.85,
            "citation_compliance": 0.88,
            "p95_latency_ms": 20000,
            "avg_cost_per_case_usd": 0.01,
        },
        "pricing": {
            "prompt_per_1k_usd": 0.00015,
            "completion_per_1k_usd": 0.0006,
        },
        "judge_enabled": True,
        "judge_model": "gpt-5.4-mini",
    }


class HistoryReportTest(unittest.TestCase):
    def test_select_comparable_runs_uses_release_pointer_and_excludes_smoke(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output" / "benchmarks"
            output_root.mkdir(parents=True)

            for payload in [
                _summary_payload(
                    run_id="20260306_163931",
                    generated_at_utc="2026-03-06T17:48:12.325082+00:00",
                    pass_rate=0.3583,
                    tool_precision=0.9062,
                    tool_recall=0.9667,
                    citation_compliance=0.85,
                    p50_latency_ms=29327.0,
                    p95_latency_ms=59439.2,
                    avg_cost_per_case_usd=0.00085588,
                ),
                _summary_payload(
                    run_id="20260307_101108",
                    generated_at_utc="2026-03-07T11:08:17.965871+00:00",
                    pass_rate=0.3667,
                    tool_precision=0.9091,
                    tool_recall=1.0,
                    citation_compliance=0.8167,
                    p50_latency_ms=24374.5,
                    p95_latency_ms=46977.55,
                    avg_cost_per_case_usd=0.00081372,
                ),
                _summary_payload(
                    run_id="20260307_120000",
                    generated_at_utc="2026-03-07T12:00:00+00:00",
                    track="smoke",
                    requested_limit=10,
                    total_cases=10,
                    pass_rate=0.8,
                    tool_precision=0.9,
                    tool_recall=0.9,
                    citation_compliance=0.9,
                    p50_latency_ms=1000.0,
                    p95_latency_ms=2000.0,
                    avg_cost_per_case_usd=0.0001,
                ),
            ]:
                run_dir = output_root / payload["run_id"]
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

            (output_root / "latest_release_run.txt").write_text("20260307_101108\n", encoding="utf-8")
            (output_root / "latest_smoke_run.txt").write_text("20260307_120000\n", encoding="utf-8")

            runs = load_history_runs(output_root)
            latest, comparable = select_comparable_runs(runs, track="release", latest_run_id="20260307_101108")

            self.assertEqual(latest.run_id, "20260307_101108")
            self.assertEqual([run.run_id for run in comparable], ["20260306_163931", "20260307_101108"])

    def test_select_comparable_runs_uses_smoke_pointer(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output" / "benchmarks"
            output_root.mkdir(parents=True)

            for payload in [
                _summary_payload(
                    run_id="20260307_120000",
                    generated_at_utc="2026-03-07T12:00:00+00:00",
                    track="smoke",
                    requested_limit=10,
                    total_cases=10,
                    pass_rate=0.8,
                    tool_precision=0.9,
                    tool_recall=0.9,
                    citation_compliance=0.9,
                    p50_latency_ms=1000.0,
                    p95_latency_ms=2000.0,
                    avg_cost_per_case_usd=0.0001,
                ),
                _summary_payload(
                    run_id="20260307_130000",
                    generated_at_utc="2026-03-07T13:00:00+00:00",
                    track="smoke",
                    requested_limit=10,
                    total_cases=10,
                    pass_rate=0.9,
                    tool_precision=0.95,
                    tool_recall=1.0,
                    citation_compliance=0.95,
                    p50_latency_ms=900.0,
                    p95_latency_ms=1800.0,
                    avg_cost_per_case_usd=0.00009,
                ),
            ]:
                run_dir = output_root / payload["run_id"]
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

            (output_root / "latest_smoke_run.txt").write_text("20260307_130000\n", encoding="utf-8")

            runs = load_history_runs(output_root)
            latest, comparable = select_comparable_runs(runs, track="smoke", latest_run_id="20260307_130000")

            self.assertEqual(latest.run_id, "20260307_130000")
            self.assertEqual([run.run_id for run in comparable], ["20260307_120000", "20260307_130000"])

    def test_refresh_history_report_updates_release_readme_and_svg(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_root = root / "output" / "benchmarks"
            output_root.mkdir(parents=True)
            svg_path = root / "docs" / "assets" / "benchmark_history.svg"
            readme_path = root / "README.md"
            readme_path.write_text(
                "# Demo\n\n## 검증 결과\n\n"
                "| 테스트 | `123 passed, 4 subtests passed` |\n\n"
                "old\n\n## 문서\n\nkeep\n",
                encoding="utf-8",
            )

            for payload in [
                _summary_payload(
                    run_id="20260303_134325",
                    generated_at_utc="2026-03-03T15:24:50.806715+00:00",
                    pass_rate=0.3833,
                    tool_precision=0.8049,
                    tool_recall=0.44,
                    citation_compliance=0.3056,
                    p50_latency_ms=49835.5,
                    p95_latency_ms=62063.0,
                    avg_cost_per_case_usd=0.00219042,
                ),
                _summary_payload(
                    run_id="20260307_101108",
                    generated_at_utc="2026-03-07T11:08:17.965871+00:00",
                    pass_rate=0.3667,
                    tool_precision=0.9091,
                    tool_recall=1.0,
                    citation_compliance=0.8167,
                    p50_latency_ms=24374.5,
                    p95_latency_ms=46977.55,
                    avg_cost_per_case_usd=0.00081372,
                ),
                _summary_payload(
                    run_id="20260307_120000",
                    generated_at_utc="2026-03-07T12:00:00+00:00",
                    track="smoke",
                    requested_limit=10,
                    total_cases=10,
                    pass_rate=0.8,
                    tool_precision=0.9,
                    tool_recall=0.9,
                    citation_compliance=0.9,
                    p50_latency_ms=1000.0,
                    p95_latency_ms=2000.0,
                    avg_cost_per_case_usd=0.0001,
                ),
            ]:
                run_dir = output_root / payload["run_id"]
                run_dir.mkdir()
                (run_dir / "summary.json").write_text(json.dumps(payload), encoding="utf-8")

            (output_root / "latest_release_run.txt").write_text("20260307_101108\n", encoding="utf-8")
            (output_root / "latest_smoke_run.txt").write_text("20260307_120000\n", encoding="utf-8")

            latest, comparable = refresh_history_report(
                output_root=output_root,
                readme_path=readme_path,
                svg_path=svg_path,
                track="release",
            )

            readme_text = readme_path.read_text(encoding="utf-8")
            svg_text = svg_path.read_text(encoding="utf-8")

            self.assertEqual(latest.run_id, "20260307_101108")
            self.assertEqual(len(comparable), 2)
            self.assertIn("`123 passed, 4 subtests passed`", readme_text)
            self.assertIn("`20260307_101108`", readme_text)
            self.assertIn(
                "로컬 benchmark 실행은 `output/benchmarks/latest_release_run.txt`를",
                readme_text,
            )
            self.assertIn("비교 가능한 2개 generated-suite 런 기준", readme_text)
            self.assertIn("[docs/assets/benchmark_history.svg](docs/assets/benchmark_history.svg)", readme_text)
            self.assertIn("`output/benchmarks/<run_id>/summary.json`", readme_text)
            self.assertIn("`output/benchmarks/<run_id>/report.md`", readme_text)
            self.assertNotIn("](output/benchmarks/", readme_text)
            self.assertIn("20260307_101108", svg_text)
            self.assertIn("20260303_134325", svg_text)
            self.assertNotIn("20260307_120000", readme_text)


if __name__ == "__main__":
    unittest.main()
