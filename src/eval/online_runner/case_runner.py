from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from urllib3.exceptions import ReadTimeoutError

from src.infra.sse import iter_sse_events

from ..judge_llm import LLMJudge
from ..config_models import BenchmarkCase, BenchmarkConfig, BenchmarkLiveSlackConfig
from ..io import load_cases_jsonl
from ..reporting.summary import build_summary
from ..reporting.writer import write_run_outputs
from ..result_models import CaseResult
from ..summary_models import RunSummary, RunTrack
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
    live_slack: BenchmarkLiveSlackConfig | None = None,
) -> CaseResult:
    request_context = build_request_context(
        fixtures_path=fixtures_path,
        case=case,
        live_slack=live_slack,
    )
    endpoint_url = endpoint.rstrip("/") + "/agent/stream"

    latency_ms_e2e: int | None = None
    parsed_response = ParsedResponseData()
    if not request_context.runtime_errors:
        started = time.monotonic()
        stream_opened = False
        try:
            with requests.post(
                endpoint_url,
                json=request_context.request_payload,
                timeout=timeout_seconds,
                headers={"Accept": "text/event-stream"},
                stream=True,
                allow_redirects=False,
            ) as response:
                parsed_response.http_status = response.status_code
                parsed_response.request_id = str(response.headers.get("x-request-id") or "").strip() or None
                if response.status_code != 200:
                    body = response.text.strip()
                    if len(body) > 300:
                        body = body[:300] + " ..."
                    parsed_response.runtime_errors.append(f"HTTP {response.status_code}: {body}")
                else:
                    content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
                    if content_type != "text/event-stream":
                        raise ValueError(f"expected text/event-stream, received {content_type or 'missing Content-Type'}")
                    stream_opened = True
                    received_event = False
                    final_received = False
                    done_received = False
                    for event in iter_sse_events(response.iter_content(chunk_size=None)):
                        received_event = True
                        if event.event == "error":
                            message = str(event.data.get("message") or "server reported an error")
                            parsed_response.runtime_errors.append(f"SSE error: {message}")
                        elif event.event == "final_response":
                            latency_ms_e2e = int((time.monotonic() - started) * 1000)
                            final = parse_agent_response(
                                event.data,
                                http_status=response.status_code,
                                request_id=parsed_response.request_id,
                            )
                            final.runtime_errors.extend(parsed_response.runtime_errors)
                            parsed_response = final
                            final_received = True
                            break
                        elif event.event == "done":
                            done_received = True
                            break
                    if not final_received:
                        reason = "done received" if done_received else ("stream ended" if received_event else "empty stream")
                        parsed_response.response_errors.append(f"SSE final_response missing ({reason})")
        except requests.Timeout:
            message = "SSE stream timeout before final_response" if stream_opened else "request timeout"
            parsed_response.runtime_errors.append(message)
        except requests.RequestException as exc:
            if stream_opened and any(isinstance(reason, ReadTimeoutError) for reason in exc.args):
                parsed_response.runtime_errors.append("SSE stream timeout before final_response")
            else:
                prefix = "SSE stream disconnected before final_response" if stream_opened else "request failed"
                parsed_response.runtime_errors.append(f"{prefix}: {exc}")
        except ValueError as exc:
            parsed_response.response_errors.append(f"SSE protocol error: {exc}")
        except Exception as exc:
            parsed_response.runtime_errors.append(f"unexpected error: {exc}")
        finally:
            if latency_ms_e2e is None:
                latency_ms_e2e = int((time.monotonic() - started) * 1000)
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
    cleanup_session_upload_dir(request_context.session_id)
    return result


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
