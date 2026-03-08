from __future__ import annotations

import json
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..latency import LatencyBreakdownModel
from .judge_llm import LLMJudge
from .reporting import build_summary, write_run_outputs
from .scoring_rules import (
    compute_cost_usd,
    compute_final_score,
    compute_rule_scores,
    compute_rule_weighted_score,
    resolve_effective_weights,
)
from .schemas import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    EvidenceItem,
    PlannerDiagnostic,
    RetrievalDiagnostic,
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


def _parse_evidence_items(
    raw_items: Any,
    *,
    label: str,
    response_errors: list[str],
) -> list[EvidenceItem]:
    parsed: list[EvidenceItem] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append(f"{label} must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            response_errors.append(f"{label}[{index}] must be an object")
            continue
        try:
            parsed.append(EvidenceItem.model_validate(item))
        except Exception as exc:
            response_errors.append(f"{label}[{index}] invalid: {exc}")
    return parsed


def _parse_retrieval_diagnostics(
    raw_items: Any,
    *,
    response_errors: list[str],
) -> list[RetrievalDiagnostic]:
    parsed: list[RetrievalDiagnostic] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append("debug.retrieval_diagnostics must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            response_errors.append(f"debug.retrieval_diagnostics[{index}] must be an object")
            continue
        try:
            parsed.append(RetrievalDiagnostic.model_validate(item))
        except Exception as exc:
            response_errors.append(f"debug.retrieval_diagnostics[{index}] invalid: {exc}")
    return parsed


def _parse_planner_diagnostics(
    raw_item: Any,
    *,
    response_errors: list[str],
) -> PlannerDiagnostic | None:
    if raw_item is None:
        return None
    if not isinstance(raw_item, dict):
        response_errors.append("debug.planner_diagnostics must be an object")
        return None
    try:
        return PlannerDiagnostic.model_validate(raw_item)
    except Exception as exc:
        response_errors.append(f"debug.planner_diagnostics invalid: {exc}")
        return None


def _parse_latency_breakdown(
    raw_item: Any,
    *,
    response_errors: list[str],
) -> LatencyBreakdownModel | None:
    if raw_item is None:
        return None
    if not isinstance(raw_item, dict):
        response_errors.append("debug.latency_breakdown must be an object")
        return None
    try:
        return LatencyBreakdownModel.model_validate(raw_item)
    except Exception as exc:
        response_errors.append(f"debug.latency_breakdown invalid: {exc}")
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

    runtime_errors: list[str] = []
    response_errors: list[str] = []
    judge_errors: list[str] = []

    request_payload: dict[str, Any] = {
        "query": case.query,
        "session_id": session_id,
        "include_debug": True,
    }
    if case.slack_channel_id:
        request_payload["slack_channel_id"] = case.slack_channel_id
    if case.slack_user_id:
        request_payload["slack_user_id"] = case.slack_user_id
    if case.slack_email:
        request_payload["slack_email"] = case.slack_email

    upload_path: str | None = None
    try:
        upload_path = _build_upload_path(case=case, fixtures_path=fixtures_path, session_id=session_id)
        if upload_path:
            request_payload["upload_file_path"] = upload_path
    except Exception as exc:
        runtime_errors.append(str(exc))

    http_status = 0
    response_text = ""
    response_payload: dict[str, Any] | None = None
    response_evidence: list[EvidenceItem] = []
    observed_evidence: list[EvidenceItem] = []
    retrieval_diagnostics: list[RetrievalDiagnostic] = []
    planner_diagnostics: PlannerDiagnostic | None = None
    response_trace: str | None = None
    response_file_path: str | None = None
    latency_ms_e2e: int | None = None
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    model_name: str | None = None
    tool_calls: list[str] = []
    token_usage: TokenUsage | None = None

    if not runtime_errors:
        started = time.monotonic()
        try:
            response = requests.post(endpoint_url, json=request_payload, timeout=timeout_seconds)
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            http_status = response.status_code

            if response.status_code != 200:
                runtime_errors.append(_build_error_message_from_response(response))
            else:
                try:
                    body = response.json()
                except json.JSONDecodeError:
                    response_errors.append("response is not valid JSON")
                    body = {}

                if isinstance(body, dict):
                    response_trace = body.get("trace")
                    response_file_path = body.get("file_path")

                    response_raw = body.get("response")
                    if not isinstance(response_raw, dict):
                        response_errors.append("response payload must be an object")
                    else:
                        response_payload = response_raw
                        answer = response_raw.get("answer")
                        if isinstance(answer, str):
                            response_text = answer
                        else:
                            response_errors.append("response.answer must be a string")
                        if not response_text.strip():
                            response_errors.append("response.answer is empty")

                        response_evidence = _parse_evidence_items(
                            response_raw.get("evidence"),
                            label="response.evidence",
                            response_errors=response_errors,
                        )

                    debug_payload = body.get("debug")
                    if isinstance(debug_payload, dict):
                        tool_calls = [
                            str(name)
                            for name in (debug_payload.get("tool_calls") or [])
                            if name
                        ]
                        latency_raw = debug_payload.get("latency_ms_server")
                        if latency_raw is not None:
                            try:
                                latency_ms_server = int(latency_raw)
                            except (TypeError, ValueError):
                                response_errors.append("debug.latency_ms_server must be an integer")
                        model_name = str(debug_payload.get("model_name")) if debug_payload.get("model_name") else None
                        token_usage = _parse_token_usage(debug_payload)
                        observed_evidence = _parse_evidence_items(
                            debug_payload.get("observed_evidence"),
                            label="debug.observed_evidence",
                            response_errors=response_errors,
                        )
                        retrieval_diagnostics = _parse_retrieval_diagnostics(
                            debug_payload.get("retrieval_diagnostics"),
                            response_errors=response_errors,
                        )
                        planner_diagnostics = _parse_planner_diagnostics(
                            debug_payload.get("planner_diagnostics"),
                            response_errors=response_errors,
                        )
                        latency_breakdown = _parse_latency_breakdown(
                            debug_payload.get("latency_breakdown"),
                            response_errors=response_errors,
                        )
                    else:
                        response_errors.append("debug payload is missing (include_debug=true expected)")
        except requests.Timeout:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            runtime_errors.append("request timeout")
        except requests.RequestException as exc:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            runtime_errors.append(f"request failed: {exc}")
        except Exception as exc:
            latency_ms_e2e = int((time.monotonic() - started) * 1000)
            runtime_errors.append(f"unexpected error: {exc}")

    effective_weights, weights_error = resolve_effective_weights(
        base_weights=config.weights,
        case_override=case.weight_override,
    )
    if weights_error:
        runtime_errors.append(f"weight_override error: {weights_error}")

    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    if response_text.strip() and config.judge_enabled:
        llm_judge_score, llm_judge_reason, judge_error = judge.score_case(
            case=case,
            response_text=response_text,
            tool_calls=tool_calls,
        )
        if judge_error:
            judge_errors.append(judge_error)

    rule_scores = compute_rule_scores(
        case=case,
        response_text=response_text,
        called_tools=tool_calls,
        response_evidence=response_evidence,
        observed_evidence=observed_evidence,
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        judge_errors=judge_errors,
    )
    rule_weighted = compute_rule_weighted_score(rule_scores, effective_weights)

    final_score = compute_final_score(
        rule_weighted_score=rule_weighted,
        llm_judge_score=llm_judge_score,
        weights=effective_weights,
    )
    passed = final_score >= 0.75
    cost = compute_cost_usd(token_usage=token_usage, pricing=config.pricing)

    result = CaseResult(
        run_id=run_id,
        case_id=case.case_id,
        category=case.category,
        scenario=case.scenario,
        query=case.query,
        session_id=session_id,
        endpoint=endpoint_url,
        upload_fixture=case.upload_fixture,
        request_payload=request_payload,
        http_status=http_status,
        response_text=response_text,
        response_payload=response_payload,
        evidence=response_evidence,
        observed_evidence=observed_evidence,
        retrieval_diagnostics=retrieval_diagnostics,
        planner_diagnostics=planner_diagnostics,
        file_path=response_file_path,
        trace=response_trace,
        latency_ms_e2e=latency_ms_e2e,
        latency_ms_server=latency_ms_server,
        latency_breakdown=latency_breakdown,
        tool_calls=tool_calls,
        token_usage=token_usage,
        model_name=model_name,
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        judge_errors=judge_errors,
        effective_weights=effective_weights.as_dict(),
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
