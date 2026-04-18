from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..judge_llm import LLMJudge
from ..reporting.summary import build_summary
from ..reporting.writer import write_run_outputs
from ..schemas import BenchmarkCase, BenchmarkConfig, CaseResult, RunSummary, RunTrack, load_cases_jsonl
from .request_builder import build_request_context, cleanup_session_upload_dir
from .response_parser import ParsedResponseData, parse_agent_response
from .result_builder import build_case_result


def _run_single_case(
    *,
    run_id: str,
    endpoint: str,
    fixtures_path: Path,
    case: BenchmarkCase,
    timeout_seconds: int,
    judge: LLMJudge,
    config: BenchmarkConfig,
) -> CaseResult:
    request_context = build_request_context(fixtures_path=fixtures_path, case=case)
    endpoint_url = endpoint.rstrip("/") + "/agent"

    latency_ms_e2e: int | None = None
    parsed_response = ParsedResponseData()
    if not request_context.runtime_errors:
        started = time.monotonic()
        try:
            response = requests.post(endpoint_url, json=request_context.request_payload, timeout=timeout_seconds)
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            parsed_response = parse_agent_response(response)
        except requests.Timeout:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            parsed_response.runtime_errors.append("request timeout")
        except requests.RequestException as exc:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            parsed_response.runtime_errors.append(f"request failed: {exc}")
        except Exception as exc:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            parsed_response.runtime_errors.append(f"unexpected error: {exc}")
    else:
        parsed_response.runtime_errors.extend(request_context.runtime_errors)

    result = build_case_result(
        run_id=run_id,
        endpoint_url=endpoint_url,
        case=case,
        judge=judge,
        config=config,
        session_id=request_context.session_id,
        created_at=request_context.created_at,
        request_payload=request_context.request_payload,
        latency_ms_e2e=latency_ms_e2e,
        parsed_response=parsed_response,
    )
    cleanup_session_upload_dir(request_context.session_id)
    return result


def _normalize_limit(limit: int | None) -> int | None:
    return limit if limit is not None and limit > 0 else None


def latest_run_pointer_path(output_root: Path, track: RunTrack) -> Path:
    return output_root / f"latest_{track}_run.txt"


def run_online_benchmark(
    *,
    fixtures_path: Path,
    endpoint: str,
    config: BenchmarkConfig,
    config_path: Path,
    output_root: Path,
    track: RunTrack,
    limit: int | None = None,
) -> tuple[Path, list[CaseResult], RunSummary]:
    cases = load_cases_jsonl(fixtures_path)
    requested_limit = _normalize_limit(limit)
    if requested_limit is not None:
        cases = cases[:requested_limit]
    if not cases:
        raise ValueError("No benchmark cases found.")

    run_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    judge = LLMJudge(model_name=config.judge_model, enabled=config.judge_enabled)

    results: list[CaseResult] = []
    for index, case in enumerate(cases, 1):
        result = _run_single_case(
            run_id=run_id,
            endpoint=endpoint,
            fixtures_path=fixtures_path,
            case=case,
            timeout_seconds=config.request_timeout_seconds,
            judge=judge,
            config=config,
        )
        results.append(result)
        print(
            f"[{index}/{len(cases)}] {case.case_id} score={float(result.composite_quality_score or 0.0):.3f} "
            f"status={result.http_status} latency={result.latency_ms_e2e}ms"
        )

    summary = build_summary(
        run_id=run_id,
        endpoint=endpoint,
        fixtures_path=str(fixtures_path),
        config_path=str(config_path),
        track=track,
        requested_limit=requested_limit,
        config=config,
        cases=cases,
        results=results,
    )

    run_dir = output_root / run_id
    write_run_outputs(output_dir=run_dir, results=results, summary=summary)
    latest_run_pointer_path(output_root, track).write_text(run_id + "\n", encoding="utf-8")
    return run_dir, results, summary


__all__ = [
    "_run_single_case",
    "latest_run_pointer_path",
    "run_online_benchmark",
]
