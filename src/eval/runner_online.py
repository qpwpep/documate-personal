from __future__ import annotations

import json
import re
import shutil
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

from ..answer_schema import ClaimItem, filter_claims_by_evidence
from ..contracts.debug import DEBUG_CRITICAL_FIELDS, DEBUG_REQUIRED_FIELDS, DEBUG_SCHEMA_VERSION
from ..contracts.boundary.debug import parse_llm_calls, parse_retry_state, parse_token_usage
from ..contracts.boundary.planner import parse_planner_diagnostic
from ..contracts.boundary.retrieval import parse_retrieval_diagnostics
from ..latency import LatencyBreakdownModel
from .judge_llm import LLMJudge
from .reporting import build_summary, write_run_outputs
from .scoring_rules import (
    compute_composite_quality_score,
    compute_cost_usd,
    compute_rule_scores,
    compute_rule_weighted_score,
    resolve_effective_weights,
)
from .schemas import (
    BenchmarkCase,
    BenchmarkConfig,
    CaseResult,
    EvidenceItem,
    JudgeSubscores,
    LLMCallMetadata,
    PlannerDiagnostic,
    RetrievalDiagnostic,
    RunSummary,
    TokenUsage,
    load_cases_jsonl,
)


_DEFAULT_JUDGE_MIN_SCORES: dict[str, float] = {
    "docs_only": 0.70,
    "hybrid": 0.70,
}
_REQUEST_ID_PATTERN = re.compile(r"Request ID:\s*([^,\s]+)")
_PRODUCT_PASS_FLOOR = 0.75


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
    return parse_token_usage(raw_debug.get("token_usage"))


def _parse_llm_calls(
    raw_items: Any,
    *,
    response_errors: list[str],
) -> list[LLMCallMetadata]:
    parsed: list[LLMCallMetadata] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append("debug.llm_calls must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        if not isinstance(item, dict):
            response_errors.append(f"debug.llm_calls[{index}] must be an object")
    return parse_llm_calls(raw_items)


def _parse_string_list(
    raw_items: Any,
    *,
    label: str,
    response_errors: list[str],
) -> list[str]:
    parsed: list[str] = []
    if raw_items is None:
        return parsed

    if not isinstance(raw_items, list):
        response_errors.append(f"{label} must be a list")
        return parsed

    for index, item in enumerate(raw_items):
        text = str(item or "").strip()
        if not text:
            response_errors.append(f"{label}[{index}] must be a non-empty string")
            continue
        parsed.append(text)
    return parsed


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
    return parse_retrieval_diagnostics(raw_items)


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
    return parse_planner_diagnostic(raw_item)


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


def _parse_validator_metadata(
    raw_item: Any,
    *,
    response_errors: list[str],
) -> tuple[str | None, str | None]:
    if raw_item is None:
        return None, None
    if not isinstance(raw_item, dict):
        response_errors.append("debug.retry_context must be an object")
        return None, None

    retry_context = parse_retry_state(raw_item)
    validator_reason = retry_context.retry_reason
    validator_feedback = retry_context.retrieval_feedback
    return (
        str(validator_reason).strip() if validator_reason else None,
        str(validator_feedback).strip() if validator_feedback else None,
    )


def _resolve_judge_min_score(case: BenchmarkCase, config: BenchmarkConfig) -> float | None:
    if case.judge_min_score is not None:
        return float(case.judge_min_score)
    configured_threshold = config.judge_min_score.for_category(case.category)
    if configured_threshold is not None:
        return float(configured_threshold)
    return _DEFAULT_JUDGE_MIN_SCORES.get(case.category)


def _extract_request_id(trace: str | None) -> str | None:
    if not trace:
        return None
    match = _REQUEST_ID_PATTERN.search(str(trace))
    if not match:
        return None
    request_id = str(match.group(1)).strip()
    return request_id or None


def _extract_request_id_from_response(response: Any, trace: str | None) -> str | None:
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        header_value = headers.get("x-request-id") or headers.get("X-Request-Id")
        if header_value:
            return str(header_value).strip()
    return _extract_request_id(trace)


def _parse_claims_from_response_payload(response_payload: dict[str, Any] | None) -> list[ClaimItem]:
    if not isinstance(response_payload, dict):
        return []
    raw_claims = response_payload.get("claims")
    if not isinstance(raw_claims, list):
        return []
    claims: list[ClaimItem] = []
    for item in raw_claims:
        if not isinstance(item, dict):
            continue
        try:
            claims.append(ClaimItem.model_validate(item))
        except Exception:
            continue
    return claims


def _build_gate_failures(
    *,
    runtime_errors: list[str],
    response_errors: list[str],
    debug_errors: list[str],
    missing_required_debug_fields: list[str],
    product_pass: bool | None,
    judge_pass: bool | None,
    judge_errors: list[str],
) -> list[str]:
    failures: list[str] = []
    if runtime_errors:
        failures.append("runtime_error")
    if response_errors:
        failures.append("response_contract_error")
    if debug_errors:
        failures.append("debug_error")
    if missing_required_debug_fields:
        failures.append("missing_debug_fields")
    if product_pass is False:
        failures.append("product_quality_below_floor")
    if judge_pass is False:
        failures.append("judge_min_score_audit_failed")
    if any(str(error).startswith("invalid_eval:") for error in judge_errors):
        failures.append("invalid_eval")
    if any("judge payload is incomplete" in str(error) for error in judge_errors):
        failures.append("judge_input_incomplete")
    return failures


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
    debug_errors: list[str] = []
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
    validator_reason: str | None = None
    validator_feedback: str | None = None
    response_trace: str | None = None
    request_id: str | None = None
    response_file_path: str | None = None
    latency_ms_e2e: int | None = None
    latency_ms_server: int | None = None
    latency_breakdown: LatencyBreakdownModel | None = None
    model_name: str | None = None
    models_used: list[str] = []
    tool_calls: list[str] = []
    tool_call_count = 0
    token_usage: TokenUsage | None = None
    llm_calls: list[LLMCallMetadata] = []
    planner_errors: list[str] = []
    debug_schema_version: int | None = None
    debug_observability_status: str | None = None
    missing_required_debug_fields: list[str] = []
    judge_subscores: JudgeSubscores | None = None
    synthesis_mode: str | None = None
    judge_input_complete: bool | None = None
    valid_claim_count = 0
    invalid_claim_count = 0

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
                    request_id = _extract_request_id_from_response(response, response_trace)
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
                        present_debug_keys = {str(key) for key in debug_payload.keys()}
                        missing_required_debug_fields = [
                            field for field in DEBUG_REQUIRED_FIELDS if field not in present_debug_keys
                        ]
                        schema_version_raw = debug_payload.get("schema_version")
                        if schema_version_raw is None:
                            response_errors.append("debug.schema_version is missing")
                        else:
                            try:
                                debug_schema_version = int(schema_version_raw)
                            except (TypeError, ValueError):
                                response_errors.append("debug.schema_version must be an integer")
                        if debug_schema_version is not None and debug_schema_version != DEBUG_SCHEMA_VERSION:
                            response_errors.append(
                                f"debug.schema_version must be {DEBUG_SCHEMA_VERSION}"
                            )
                        observability_status_raw = debug_payload.get("observability_status")
                        if observability_status_raw is None:
                            response_errors.append("debug.observability_status is missing")
                        else:
                            normalized_observability_status = str(observability_status_raw).strip().lower()
                            if normalized_observability_status in {"ok", "degraded", "failed"}:
                                debug_observability_status = normalized_observability_status
                            else:
                                response_errors.append(
                                    "debug.observability_status must be one of ok/degraded/failed"
                                )
                        if debug_payload.get("missing_required_debug_fields") is None:
                            response_errors.append("debug.missing_required_debug_fields is missing")
                        else:
                            self_reported_missing_fields = _parse_string_list(
                                debug_payload.get("missing_required_debug_fields"),
                                label="debug.missing_required_debug_fields",
                                response_errors=response_errors,
                            )
                            for field_name in self_reported_missing_fields:
                                if field_name not in missing_required_debug_fields:
                                    missing_required_debug_fields.append(field_name)
                        critical_missing_debug_fields = [
                            field
                            for field in missing_required_debug_fields
                            if field in DEBUG_CRITICAL_FIELDS
                        ]
                        if critical_missing_debug_fields:
                            response_errors.append(
                                "critical debug fields missing: "
                                + ", ".join(critical_missing_debug_fields)
                            )
                        tool_calls = [
                            str(name)
                            for name in (debug_payload.get("tool_calls") or [])
                            if name
                        ]
                        try:
                            tool_call_count = int(
                                debug_payload.get("tool_call_count", len(tool_calls)) or len(tool_calls)
                            )
                        except (TypeError, ValueError):
                            response_errors.append("debug.tool_call_count must be an integer")
                            tool_call_count = len(tool_calls)
                        latency_raw = debug_payload.get("latency_ms_server")
                        if latency_raw is not None:
                            try:
                                latency_ms_server = int(latency_raw)
                            except (TypeError, ValueError):
                                response_errors.append("debug.latency_ms_server must be an integer")
                        model_name = str(debug_payload.get("model_name")) if debug_payload.get("model_name") else None
                        models_used_raw = debug_payload.get("models_used")
                        if isinstance(models_used_raw, list):
                            models_used = [str(name) for name in models_used_raw if name]
                        elif model_name:
                            models_used = [model_name]
                        token_usage = _parse_token_usage(debug_payload)
                        llm_calls = _parse_llm_calls(
                            debug_payload.get("llm_calls"),
                            response_errors=response_errors,
                        )
                        debug_errors = _parse_string_list(
                            debug_payload.get("errors"),
                            label="debug.errors",
                            response_errors=response_errors,
                        )
                        planner_errors = _parse_string_list(
                            debug_payload.get("planner_errors"),
                            label="debug.planner_errors",
                            response_errors=response_errors,
                        )
                        if not models_used and llm_calls:
                            models_used = []
                            for llm_call in llm_calls:
                                response_metadata = llm_call.response_metadata
                                model_name_candidate = response_metadata.get("model_name") or response_metadata.get("model")
                                if model_name_candidate and model_name_candidate not in models_used:
                                    models_used.append(str(model_name_candidate))
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
                        validator_reason, validator_feedback = _parse_validator_metadata(
                            debug_payload.get("retry_context"),
                            response_errors=response_errors,
                        )
                        latency_breakdown = _parse_latency_breakdown(
                            debug_payload.get("latency_breakdown"),
                            response_errors=response_errors,
                        )
                        if latency_breakdown is not None and latency_breakdown.synthesis_attempts:
                            synthesis_mode = latency_breakdown.synthesis_attempts[0].mode
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

    response_claims = _parse_claims_from_response_payload(response_payload)
    if response_claims:
        valid_claims, invalid_claims = filter_claims_by_evidence(
            claims=response_claims,
            evidence_items=observed_evidence or response_evidence,
        )
        valid_claim_count = len(valid_claims)
        invalid_claim_count = len(invalid_claims)
    retrieval_warnings = sorted(
        {
            str(warning).strip()
            for diagnostic in retrieval_diagnostics
            for warning in diagnostic.warnings
            if str(warning).strip()
        }
    )
    if retrieval_warnings:
        warning_text = ", ".join(retrieval_warnings)
        if validator_feedback:
            validator_feedback = f"{validator_feedback} | retrieval_warnings={warning_text}"
        else:
            validator_feedback = f"retrieval_warnings={warning_text}"

    llm_judge_score: float | None = None
    llm_judge_reason: str | None = None
    if response_text.strip() and config.judge_enabled:
        judge_payload_builder = getattr(judge, "build_case_payload", LLMJudge.build_case_payload)
        judge_payload_validator = getattr(judge, "is_payload_complete", LLMJudge.is_payload_complete)
        judge_payload = judge_payload_builder(
            case=case,
            response_text=response_text,
            tool_calls=tool_calls,
            claims=response_claims,
            response_evidence=response_evidence,
            observed_evidence=observed_evidence,
            retrieval_diagnostics=retrieval_diagnostics,
            planner_diagnostics=planner_diagnostics,
            validator_reason=validator_reason,
            synthesis_mode=synthesis_mode,
            valid_claim_count=valid_claim_count,
            invalid_claim_count=invalid_claim_count,
            tool_call_count=tool_call_count,
        )
        judge_input_complete = judge_payload_validator(judge_payload)
        try:
            judge_result = judge.score_case(
                case=case,
                response_text=response_text,
                tool_calls=tool_calls,
                claims=response_claims,
                response_evidence=response_evidence,
                observed_evidence=observed_evidence,
                retrieval_diagnostics=retrieval_diagnostics,
                planner_diagnostics=planner_diagnostics,
                validator_reason=validator_reason,
                synthesis_mode=synthesis_mode,
                valid_claim_count=valid_claim_count,
                invalid_claim_count=invalid_claim_count,
                tool_call_count=tool_call_count,
            )
        except TypeError:
            judge_result = judge.score_case(case, response_text, tool_calls)
        if len(judge_result) >= 4:
            llm_judge_score, llm_judge_reason, judge_error, judge_subscores = judge_result
        else:
            llm_judge_score, llm_judge_reason, judge_error = judge_result[:3]
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
        validator_reason=validator_reason,
        response_claims=response_claims,
        retrieval_diagnostics=retrieval_diagnostics,
        synthesis_mode=synthesis_mode,
        valid_claim_count=valid_claim_count,
        invalid_claim_count=invalid_claim_count,
    )
    rule_weighted = compute_rule_weighted_score(rule_scores, effective_weights)

    composite_quality_score = compute_composite_quality_score(
        rule_weighted_score=rule_weighted,
        llm_judge_score=llm_judge_score,
        weights=effective_weights,
    )
    judge_min_score = _resolve_judge_min_score(case, config)
    judge_gate_passed: bool | None = None
    if judge_min_score is not None and response_text.strip():
        judge_gate_passed = False if llm_judge_score is None else llm_judge_score >= judge_min_score
    if judge_min_score is not None and llm_judge_score is not None and response_text.strip():
        judge_gate_passed = llm_judge_score >= judge_min_score
        if judge_gate_passed is False:
            judge_errors.append(
                "judge_min_score audit failed: "
                f"score={llm_judge_score:.3f} threshold={judge_min_score:.3f}"
            )
    product_pass = composite_quality_score >= _PRODUCT_PASS_FLOOR
    if any(str(error).startswith("invalid_eval:") for error in judge_errors):
        judge_pass = False
    else:
        judge_pass = judge_gate_passed if judge_min_score is not None else (
            True if (config.judge_enabled and not judge_errors and llm_judge_score is not None) else None
        )
    release_pass = bool(product_pass and not runtime_errors and not response_errors)
    gate_failures = _build_gate_failures(
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        debug_errors=debug_errors,
        missing_required_debug_fields=missing_required_debug_fields,
        product_pass=product_pass,
        judge_pass=judge_pass,
        judge_errors=judge_errors,
    )
    cost = compute_cost_usd(
        token_usage=token_usage,
        llm_calls=[call.model_dump() for call in llm_calls],
        pricing=config.pricing,
    )

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
        request_id=request_id,
        http_status=http_status,
        response_text=response_text,
        response_payload=response_payload,
        response_claims=response_claims,
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
        tool_call_count=tool_call_count,
        token_usage=token_usage,
        model_name=model_name,
        models_used=models_used,
        llm_calls=llm_calls,
        planner_errors=planner_errors,
        debug_errors=debug_errors,
        runtime_errors=runtime_errors,
        response_errors=response_errors,
        judge_errors=judge_errors,
        validator_reason=validator_reason,
        validator_feedback=validator_feedback,
        effective_weights=effective_weights.as_dict(),
        rule_scores=rule_scores,
        rule_score_total=rule_weighted,
        debug_schema_version=debug_schema_version,
        debug_observability_status=debug_observability_status,
        missing_required_debug_fields=missing_required_debug_fields,
        judge_subscores=judge_subscores,
        judge_score_total=llm_judge_score,
        llm_judge_score=llm_judge_score,
        llm_judge_reason=llm_judge_reason,
        judge_min_score_applied=judge_min_score,
        judge_input_complete=judge_input_complete,
        judge_gate_passed=judge_gate_passed,
        invalid_eval=any(str(error).startswith("invalid_eval:") for error in judge_errors),
        valid_claim_count=valid_claim_count,
        invalid_claim_count=invalid_claim_count,
        synthesis_mode=synthesis_mode,
        gate_failures=gate_failures,
        composite_quality_score=composite_quality_score,
        product_pass=product_pass,
        judge_pass=judge_pass,
        release_pass=release_pass,
        final_score=composite_quality_score,
        passed=release_pass,
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
            f"[{index}/{len(cases)}] {case.case_id} score={float(result.composite_quality_score or 0.0):.3f} "
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
