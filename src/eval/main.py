from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .generate_cases import generate_cases_file
from .reporting import build_markdown_report
from .runner_online import run_online_benchmark
from .schemas import BenchmarkConfig, CaseResult, RunSummary, load_config


DEFAULT_CONFIG_PATH = Path("data/benchmarks/config.toml")
DEFAULT_FIXTURES_PATH = Path("data/benchmarks/fixtures/cases.generated.jsonl")
DEFAULT_OUTPUT_ROOT = Path("output/benchmarks")


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _load_config_with_env_overrides(config_path: Path) -> BenchmarkConfig:
    config = load_config(config_path)

    env_judge_model = os.getenv("JUDGE_MODEL")
    if env_judge_model:
        config.judge_model = env_judge_model

    env_judge_enabled = os.getenv("BENCHMARK_JUDGE_ENABLED")
    config.judge_enabled = _str_to_bool(env_judge_enabled, config.judge_enabled)
    return config


def command_generate(args: argparse.Namespace) -> int:
    generated = generate_cases_file(
        seed_path=args.seed,
        out_path=args.out,
        target=args.target,
        regression_seed_path=args.regression_seed,
        random_seed=args.random_seed,
    )
    print(f"Generated {len(generated)} cases at {args.out}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    if args.mode != "online":
        raise ValueError("Only online mode is supported.")

    endpoint = args.endpoint or os.getenv("BENCHMARK_ENDPOINT", "http://localhost:8000")
    config = _load_config_with_env_overrides(args.config)

    run_dir, _, summary = run_online_benchmark(
        fixtures_path=args.fixtures,
        endpoint=endpoint,
        config=config,
        config_path=args.config,
        output_root=args.output_root,
        limit=args.limit,
    )

    print(f"Run directory: {run_dir}")
    print(f"Overall: {'PASS' if summary.overall_passed else 'FAIL'}")
    return 0


def command_report(args: argparse.Namespace) -> int:
    run_path = args.run.resolve()
    summary_path = run_path / "summary.json"
    raw_path = run_path / "raw_results.jsonl"
    report_path = run_path / "report.md"

    if not summary_path.exists():
        raise FileNotFoundError(f"summary.json not found: {summary_path}")
    if not raw_path.exists():
        raise FileNotFoundError(f"raw_results.jsonl not found: {raw_path}")

    summary = RunSummary(**json.loads(summary_path.read_text(encoding="utf-8")))

    # Validate raw result lines for report regeneration safety.
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        record = line.strip()
        if not record:
            continue
        CaseResult.model_validate_json(record)

    report_path.write_text(build_markdown_report(summary), encoding="utf-8")
    print(f"Regenerated report: {report_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DocuMate benchmark CLI (online only)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_generate = subparsers.add_parser("generate", help="Generate benchmark fixture cases")
    parser_generate.add_argument("--seed", type=Path, required=True, help="Seed JSONL file path")
    parser_generate.add_argument("--out", type=Path, required=True, help="Output JSONL file path")
    parser_generate.add_argument("--target", type=int, required=True, help="Target number of cases")
    parser_generate.add_argument(
        "--regression-seed",
        type=Path,
        default=Path("data/benchmarks/fixtures/cases.regression.seed.jsonl"),
        help="Regression seed JSONL file path",
    )
    parser_generate.add_argument("--random-seed", type=int, default=42, help="Random seed")
    parser_generate.set_defaults(func=command_generate)

    parser_run = subparsers.add_parser("run", help="Run online benchmark against FastAPI /agent")
    parser_run.add_argument("--mode", choices=["online"], required=True, help="Execution mode")
    parser_run.add_argument(
        "--fixtures",
        type=Path,
        default=DEFAULT_FIXTURES_PATH,
        help=f"Benchmark fixtures JSONL path (default: {DEFAULT_FIXTURES_PATH})",
    )
    parser_run.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help=f"Benchmark config TOML path (default: {DEFAULT_CONFIG_PATH})",
    )
    parser_run.add_argument(
        "--endpoint",
        type=str,
        default=None,
        help="FastAPI base URL. If omitted, use BENCHMARK_ENDPOINT or http://localhost:8000",
    )
    parser_run.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Benchmark output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser_run.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs")
    parser_run.set_defaults(func=command_run)

    parser_report = subparsers.add_parser("report", help="Regenerate markdown report from an existing run")
    parser_report.add_argument("--run", type=Path, required=True, help="Run directory path")
    parser_report.set_defaults(func=command_report)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
