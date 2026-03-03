from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from .judge_llm import LLMJudge
from .reporting import build_summary, write_run_outputs
from .scoring_rules import (
    compute_cost_usd,
    compute_final_score,
    compute_rule_scores,
    compute_rule_weighted_score,
)
from .schemas import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    RunSummary,
    TokenUsage,
    load_cases_jsonl,
)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_upload_path(
    *,
    case: BenchmarkCase,
    fixtures_path: Path,
    session_id: str,
) -> str | None:
    if not case.upload_fixture:
        return None

    source = (fixtures_path.parent / "uploads" / case.upload_fixture).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"upload fixture not found: {source}")

    session_dir = (Path("uploads") / session_id).resolve()
    session_dir.mkdir(parents=True, exist_ok=True)
    target = session_dir / source.name
    shutil.copy2(source, target)
    return target.as_posix()


def _cleanup_session_upload_dir(session_id: str) -> None:
    session_dir = Path("uploads") / session_id
    if session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)


def _build_error_message_from_response(response: requests.Response) -> str:
    body = response.text.strip()
    if len(body) > 300:
        body = body[:300] + " ..."
    return f"HTTP {response.status_code}: {body}"


def _parse_token_usage(raw_debug: dict[str, Any] | None) -> TokenUsage | None:
    if not raw_debug:
        return None
    raw_usage = raw_debug.get("token_usage")
    if not isinstance(raw_usage, dict):
        return None
    try:
        return TokenUsage(
            prompt_tokens=int(raw_usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(raw_usage.get("completion_tokens", 0) or 0),
            total_tokens=int(raw_usage.get("total_tokens", 0) or 0),
        )
    except (TypeError, ValueError):
        return None


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
    session_id = str(uuid.uuid4())
    created_at = _utc_now_iso()
    endpoint_url = endpoint.rstrip("/") + "/agent"
    errors: list[str] = []
    request_payload: dict[str, Any] = {
        "query": case.query,
        "session_id": session_id,
        "include_debug": True,
    }

    upload_path: str | None = None
    try:
        upload_path = _build_upload_path(case=case, fixtures_path=fixtures_path, session_id=session_id)
        if upload_path:
            request_payload["upload_file_path"] = upload_path
    except Exception as exc:
        errors.append(str(exc))

    http_status = 0
    response_text = ""
    response_trace: str | None = None
    response_file_path: str | None = None
    debug_payload: dict[str, Any] | None = None
    latency_ms_e2e: int | None = None
    latency_ms_server: int | None = None
    model_name: str | None = None
    tool_calls: list[str] = []
    token_usage: TokenUsage | None = None

    if not errors:
        started = time.monotonic()
        try:
            response = requests.post(endpoint_url, json=request_payload, timeout=timeout_seconds)
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            http_status = response.status_code

            if response.status_code != 200:
                errors.append(_build_error_message_from_response(response))
            else:
                body = response.json()
                response_text = str(body.get("response", ""))
                response_trace = body.get("trace")
                response_file_path = body.get("file_path")
                debug_payload = body.get("debug") if isinstance(body.get("debug"), dict) else None

                if debug_payload:
                    tool_calls = [str(name) for name in (debug_payload.get("tool_calls") or []) if name]
                    latency_ms_server = (
                        int(debug_payload.get("latency_ms_server"))
                        if debug_payload.get("latency_ms_server") is not None
                        else None
                    )
                    model_name = (
                        str(debug_payload.get("model_name"))
                        if debug_payload.get("model_name")
                        else None
                    )
                    token_usage = _parse_token_usage(debug_payload)
                else:
                    errors.append("debug payload is missing (include_debug=true expected)")
        except requests.Timeout:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            errors.append("request timeout")
        except requests.RequestException as exc:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            errors.append(f"request failed: {exc}")
        except json.JSONDecodeError:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            errors.append("response is not valid JSON")
        except Exception as exc:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            errors.append(f"unexpected error: {exc}")

    rule_scores = compute_rule_scores(
        case=case,
        response_text=response_text,
        called_tools=tool_calls,
        errors=errors,
    )
    rule_weighted = compute_rule_weighted_score(rule_scores, config.weights)

    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    if response_text.strip():
        llm_judge_score, llm_judge_reason, judge_error = judge.score_case(
            case=case,
            response_text=response_text,
            tool_calls=tool_calls,
        )
        if judge_error:
            errors.append(judge_error)

    final_score = compute_final_score(
        rule_weighted_score=rule_weighted,
        llm_judge_score=llm_judge_score,
        weights=config.weights,
    )
    passed = final_score >= 0.75
    cost = compute_cost_usd(token_usage=token_usage, pricing=config.pricing)

    result = CaseResult(
        run_id=run_id,
        case_id=case.case_id,
        category=case.category,
        query=case.query,
        session_id=session_id,
        endpoint=endpoint_url,
        upload_fixture=case.upload_fixture,
        request_payload=request_payload,
        http_status=http_status,
        response_text=response_text,
        file_path=response_file_path,
        trace=response_trace,
        latency_ms_e2e=latency_ms_e2e,
        latency_ms_server=latency_ms_server,
        tool_calls=tool_calls,
        token_usage=token_usage,
        model_name=model_name,
        errors=errors,
        rule_scores=rule_scores,
        rule_score_total=rule_weighted,
        llm_judge_score=llm_judge_score,
        llm_judge_reason=llm_judge_reason,
        final_score=final_score,
        passed=passed,
        cost_usd=cost,
        created_at_utc=created_at,
    )

    _cleanup_session_upload_dir(session_id)
    return result


def run_online_benchmark(
    *,
    fixtures_path: Path,
    endpoint: str,
    config: BenchmarkConfig,
    config_path: Path,
    output_root: Path,
    limit: int | None = None,
) -> tuple[Path, list[CaseResult], RunSummary]:
    cases = load_cases_jsonl(fixtures_path)
    if limit is not None and limit > 0:
        cases = cases[:limit]
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
            f"[{index}/{len(cases)}] {case.case_id} score={result.final_score:.3f} "
            f"status={result.http_status} latency={result.latency_ms_e2e}ms"
        )

    summary = build_summary(
        run_id=run_id,
        endpoint=endpoint,
        fixtures_path=str(fixtures_path),
        config_path=str(config_path),
        config=config,
        cases=cases,
        results=results,
    )

    run_dir = output_root / run_id
    write_run_outputs(output_dir=run_dir, results=results, summary=summary)
    (output_root / "latest_run.txt").write_text(run_id + "\n", encoding="utf-8")
    return run_dir, results, summary
