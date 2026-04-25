from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from ..judge_llm import LLMJudge
from ..config_models import BenchmarkCase, BenchmarkConfig, BenchmarkLiveSlackConfig, BenchmarkStep
from ..io import load_cases_jsonl
from ..reporting.summary import build_summary
from ..reporting.writer import write_run_outputs
from ..result_models import CaseResult
from ..summary_models import RunSummary, RunTrack
from .request_builder import build_request_context, cleanup_session_upload_dir
from .response_parser import ParsedResponseData, parse_agent_response
from .result_builder import build_case_result


def _post_case_request(
    *,
    run_id: str,
    endpoint_url: str,
    fixtures_path: Path,
    case: BenchmarkCase,
    timeout_seconds: int,
    judge: LLMJudge,
    config: BenchmarkConfig,
    live_slack: BenchmarkLiveSlackConfig | None = None,
    session_id: str | None = None,
    clear_uploads: bool = False,
) -> CaseResult:
    request_context = build_request_context(
        fixtures_path=fixtures_path,
        case=case,
        live_slack=live_slack,
        session_id=session_id,
        clear_uploads=clear_uploads,
    )

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
        slack_delivery_required=request_context.slack_delivery_required,
    )
    return result


def _case_for_step(case: BenchmarkCase, step: BenchmarkStep, index: int) -> tuple[BenchmarkCase, bool]:
    payload = case.model_dump()
    step_id = step.step_id or f"step_{index:02d}"
    payload["case_id"] = f"{case.case_id}::{step_id}"
    payload["query"] = step.query

    if step.upload_fixture is not None:
        payload["upload_fixture"] = step.upload_fixture
        payload["upload_fixtures"] = [step.upload_fixture]
    elif step.upload_fixtures is not None:
        payload["upload_fixture"] = step.upload_fixtures[0] if step.upload_fixtures else None
        payload["upload_fixtures"] = list(step.upload_fixtures)
    else:
        payload["upload_fixture"] = None
        payload["upload_fixtures"] = []

    for field_name in (
        "slack_channel_id",
        "slack_user_id",
        "slack_email",
        "expected_tools",
        "forbidden_tools",
        "must_include",
        "must_not_include",
        "require_official_citation",
        "require_local_citation",
        "golden_criteria",
        "expected_behavior",
        "planner_mode",
        "faults",
    ):
        value = getattr(step, field_name)
        if value is not None:
            payload[field_name] = value

    payload["reset_slack_destination"] = bool(step.reset_slack_destination)
    if step.expected_error_codes:
        payload["expected_error_codes"] = list(step.expected_error_codes)
    return BenchmarkCase.model_validate(payload), bool(step.clear_uploads or step.upload_fixtures == [])


def _summarize_step_result(result: CaseResult, step: BenchmarkStep, index: int) -> dict:
    return {
        "step_index": index,
        "step_id": step.step_id or f"step_{index:02d}",
        "case_id": result.case_id,
        "query": result.query,
        "request_payload": result.request_payload,
        "http_status": result.http_status,
        "response_text": result.response_text,
        "tool_calls": result.tool_calls,
        "error_codes": result.error_codes,
        "runtime_errors": result.runtime_errors,
        "response_errors": result.response_errors,
        "rule_scores": result.rule_scores,
        "composite_quality_score": result.composite_quality_score,
        "release_pass": result.release_pass,
    }


def _aggregate_journey_result(case: BenchmarkCase, step_results: list[CaseResult], steps: list[BenchmarkStep]) -> CaseResult:
    last_result = step_results[-1]
    scored = [
        float(result.composite_quality_score)
        for result in step_results
        if result.composite_quality_score is not None
    ]
    aggregate_score = sum(scored) / len(scored) if scored else last_result.composite_quality_score
    latency_values = [result.latency_ms_e2e for result in step_results if result.latency_ms_e2e is not None]
    combined_tools: list[str] = []
    for result in step_results:
        for tool_name in result.tool_calls:
            if tool_name not in combined_tools:
                combined_tools.append(tool_name)
    return last_result.model_copy(
        update={
            "case_id": case.case_id,
            "query": case.query,
            "upload_fixture": case.upload_fixture,
            "upload_fixtures": case.upload_fixtures,
            "expected_behavior": case.expected_behavior,
            "expected_error_codes": case.expected_error_codes,
            "request_payload": {
                "steps": [result.request_payload for result in step_results],
            },
            "latency_ms_e2e": sum(latency_values) if latency_values else None,
            "tool_calls": combined_tools,
            "tool_call_count": len(combined_tools),
            "step_results": [
                _summarize_step_result(result, step, index)
                for index, (step, result) in enumerate(zip(steps, step_results), start=1)
            ],
            "composite_quality_score": aggregate_score,
            "final_score": aggregate_score,
            "product_pass": all(result.product_pass for result in step_results),
            "judge_pass": all(result.judge_pass is not False for result in step_results),
            "release_pass": all(bool(result.release_pass) for result in step_results),
            "passed": all(bool(result.release_pass) for result in step_results),
        }
    )


def _run_single_case(
    *,
    run_id: str,
    endpoint: str,
    fixtures_path: Path,
    case: BenchmarkCase,
    timeout_seconds: int,
    judge: LLMJudge,
    config: BenchmarkConfig,
    live_slack: BenchmarkLiveSlackConfig | None = None,
) -> CaseResult:
    endpoint_url = endpoint.rstrip("/") + "/agent"
    session_id: str | None = None
    try:
        if not case.steps:
            result = _post_case_request(
                run_id=run_id,
                endpoint_url=endpoint_url,
                fixtures_path=fixtures_path,
                case=case,
                timeout_seconds=timeout_seconds,
                judge=judge,
                config=config,
                live_slack=live_slack,
            )
            session_id = result.session_id
            return result

        session_id = None
        step_results: list[CaseResult] = []
        for index, step in enumerate(case.steps, start=1):
            step_case, clear_uploads = _case_for_step(case, step, index)
            result = _post_case_request(
                run_id=run_id,
                endpoint_url=endpoint_url,
                fixtures_path=fixtures_path,
                case=step_case,
                timeout_seconds=timeout_seconds,
                judge=judge,
                config=config,
                live_slack=live_slack,
                session_id=session_id,
                clear_uploads=clear_uploads,
            )
            session_id = result.session_id
            step_results.append(result)
        return _aggregate_journey_result(case, step_results, case.steps)
    finally:
        if session_id:
            cleanup_session_upload_dir(session_id)


def _normalize_limit(limit: int | None) -> int | None:
    return limit if limit is not None and limit > 0 else None


def latest_run_pointer_path(output_root: Path, track: RunTrack) -> Path:
    return output_root / f"latest_{track}_run.txt"


def _validate_live_slack_targets(
    *,
    cases: list[BenchmarkCase],
    live_slack: BenchmarkLiveSlackConfig | None,
) -> None:
    resolved_live_slack = live_slack or BenchmarkLiveSlackConfig()
    if not resolved_live_slack.enabled:
        return

    applicable_cases = [case for case in cases if resolved_live_slack.applies_to_case(case)]
    if not applicable_cases:
        return

    missing_channel_case = next(
        (
            case
            for case in applicable_cases
            if resolved_live_slack.requires_channel_destination(case)
        ),
        None,
    )
    if missing_channel_case and not resolved_live_slack.has_channel_destination():
        raise ValueError(
            "Live Slack channel destination is required for benchmark case "
            f"{missing_channel_case.case_id}. Provide --live-slack-channel-id or BENCHMARK_SLACK_CHANNEL_ID."
        )

    missing_dm_case = next(
        (
            case
            for case in applicable_cases
            if resolved_live_slack.requires_dm_destination(case)
        ),
        None,
    )
    if missing_dm_case and not resolved_live_slack.has_dm_destination():
        raise ValueError(
            "Live Slack DM destination is required for benchmark case "
            f"{missing_dm_case.case_id}. Provide --live-slack-user-id, --live-slack-email, "
            "BENCHMARK_SLACK_USER_ID, BENCHMARK_SLACK_EMAIL, or app-level DM defaults."
        )


def run_online_benchmark(
    *,
    fixtures_path: Path,
    endpoint: str,
    config: BenchmarkConfig,
    config_path: Path,
    output_root: Path,
    track: RunTrack,
    limit: int | None = None,
    live_slack: BenchmarkLiveSlackConfig | None = None,
) -> tuple[Path, list[CaseResult], RunSummary]:
    cases = load_cases_jsonl(fixtures_path)
    requested_limit = _normalize_limit(limit)
    if requested_limit is not None:
        cases = cases[:requested_limit]
    if not cases:
        raise ValueError("No benchmark cases found.")
    _validate_live_slack_targets(cases=cases, live_slack=live_slack)

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
            live_slack=live_slack,
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
        slack_live_enabled=bool((live_slack or BenchmarkLiveSlackConfig()).enabled),
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
