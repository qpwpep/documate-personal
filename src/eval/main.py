from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.infra.runtime_paths import (
    get_benchmark_config_path,
    get_benchmark_history_svg_path,
    get_benchmark_output_dir,
    get_generated_cases_fixture_path,
    get_readme_path,
    get_regression_seed_cases_path,
)
from .generate_cases import generate_cases_file
from .history_report import refresh_history_report
from .io import load_config
from .online_runner import run_online_benchmark
from .reporting import build_markdown_report
from .config_models import BenchmarkConfig
from .result_models import CaseResult
from .summary_models import RunSummary, RunTrack
from src.infra.settings import load_benchmark_cli_env_settings


DEFAULT_CONFIG_PATH = get_benchmark_config_path()
DEFAULT_FIXTURES_PATH = get_generated_cases_fixture_path()
DEFAULT_OUTPUT_ROOT = get_benchmark_output_dir()
DEFAULT_HISTORY_README = get_readme_path()
DEFAULT_HISTORY_SVG = get_benchmark_history_svg_path()
DEFAULT_REGRESSION_SEED_PATH = get_regression_seed_cases_path()

def _load_config_with_env_overrides(
    config_path: Path,
    *,
    benchmark_env: object | None = None,
) -> BenchmarkConfig:
    config = load_config(config_path)
    if benchmark_env is not None:
        config.judge_model = str(getattr(benchmark_env, "judge_model", config.judge_model) or config.judge_model)
        config.judge_enabled = bool(getattr(benchmark_env, "judge_enabled", config.judge_enabled))
    return config


def resolve_run_track(track: str | None, limit: int | None) -> RunTrack:
    if track in {"release", "smoke"}:
        return track
    return "smoke" if limit is not None and limit > 0 else "release"


def _paths_reference_same_target(candidate: Path, expected: Path) -> bool:
    try:
        return candidate.samefile(expected)
    except OSError:
        return candidate.resolve(strict=False) == expected.resolve(strict=False)


def validate_history_targets(track: RunTrack, readme_path: Path, svg_path: Path) -> None:
    if track == "smoke" and (
        _paths_reference_same_target(readme_path, DEFAULT_HISTORY_README)
        or _paths_reference_same_target(svg_path, DEFAULT_HISTORY_SVG)
    ):
        raise ValueError("Smoke history requires explicit --readme and --svg paths to avoid overwriting release artifacts.")


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

    benchmark_env = load_benchmark_cli_env_settings(args.config)
    endpoint = args.endpoint or benchmark_env.endpoint
    config = _load_config_with_env_overrides(args.config, benchmark_env=benchmark_env)
    track = resolve_run_track(args.track, args.limit)

    run_dir, _, summary = run_online_benchmark(
        fixtures_path=args.fixtures,
        endpoint=endpoint,
        config=config,
        config_path=args.config,
        output_root=args.output_root,
        track=track,
        limit=args.limit,
    )

    print(f"Run directory: {run_dir}")
    print(f"Track: {summary.track}")
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
    results: list[CaseResult] = []
    for line in raw_path.read_text(encoding="utf-8").splitlines():
        record = line.strip()
        if not record:
            continue
        results.append(CaseResult.model_validate_json(record))

    report_path.write_text(build_markdown_report(summary, results), encoding="utf-8")
    print(f"Regenerated report: {report_path}")
    return 0


def command_history(args: argparse.Namespace) -> int:
    validate_history_targets(args.track, args.readme, args.svg)
    latest, comparable_runs = refresh_history_report(
        output_root=args.output_root,
        readme_path=args.readme,
        svg_path=args.svg,
        track=args.track,
    )
    print(f"Updated benchmark history for {len(comparable_runs)} comparable runs.")
    print(f"Track: {args.track}")
    print(f"Latest run: {latest.run_id}")
    print(f"README: {args.readme}")
    print(f"SVG: {args.svg}")
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
        default=DEFAULT_REGRESSION_SEED_PATH,
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
        help="FastAPI base URL. If omitted, use BENCHMARK_ENDPOINT or http://127.0.0.1:8000",
    )
    parser_run.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Benchmark output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser_run.add_argument(
        "--track",
        choices=["release", "smoke"],
        default=None,
        help="Run track. Defaults to smoke when --limit is set, otherwise release.",
    )
    parser_run.add_argument("--limit", type=int, default=None, help="Optional case limit for smoke runs")
    parser_run.set_defaults(func=command_run)

    parser_report = subparsers.add_parser("report", help="Regenerate markdown report from an existing run")
    parser_report.add_argument("--run", type=Path, required=True, help="Run directory path")
    parser_report.set_defaults(func=command_report)

    parser_history = subparsers.add_parser(
        "history",
        help="Refresh benchmark history sections in README and regenerate the trend SVG",
    )
    parser_history.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=f"Benchmark output root directory (default: {DEFAULT_OUTPUT_ROOT})",
    )
    parser_history.add_argument(
        "--readme",
        type=Path,
        default=DEFAULT_HISTORY_README,
        help="README file to refresh",
    )
    parser_history.add_argument(
        "--svg",
        type=Path,
        default=DEFAULT_HISTORY_SVG,
        help="SVG output path for the benchmark trend chart",
    )
    parser_history.add_argument(
        "--track",
        choices=["release", "smoke"],
        default="release",
        help="History track to refresh. Smoke runs require explicit --readme and --svg targets.",
    )
    parser_history.set_defaults(func=command_history)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
