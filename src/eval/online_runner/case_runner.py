from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO
from uuid import uuid4

from src.app.client import AgentSessionClient, AgentStreamEvent, UploadAPIError, build_agent_payload
from src.app.uploads import build_upload_sync_request, discard_staged_files
from src.core.contracts.outcome import TurnResult
from src.infra.runtime_paths import get_upload_session_dir
from src.infra.settings import get_settings

from ..judge_llm import LLMJudge
from ..approved_uploads import ApprovedUpload, select_approved_uploads, verify_upload_manifest
from ..evidence_scope import assess_evidence_scope
from ..config_models import BenchmarkCase, BenchmarkConfig, BenchmarkLiveSlackConfig, SetupTurnExpectation
from ..io import load_cases_jsonl
from ..reporting.summary import build_summary
from ..reporting.writer import write_run_outputs
from ..result_models import CaseResult, ScenarioTurnResult
from ..summary_models import ReleaseSuiteContract, RunSummary, RunTrack, release_suite_gate
from ..weighting import ResolvedWeights, resolve_case_weights
from .scenario_inputs import case_context, resolve_fixture_uploads
from .response_parser import ParsedResponseData, parse_agent_response
from .result_builder import build_case_result


def _parse_final(body: dict, *, previous: ParsedResponseData, validated_result: TurnResult | None = None,
                 expected_partial_problem: str | None = None) -> ParsedResponseData:
    try:
        return parse_agent_response(body, http_status=previous.http_status, execution_id=previous.execution_id,
                                    validated_result=validated_result,
                                    expected_partial_problem=expected_partial_problem)
    except Exception as exc:
        # Diagnostics are external input too. An unexpected shape must not lose
        # the case, its original debug data, or subsequent scenario results.
        return ParsedResponseData(http_status=previous.http_status, execution_id=previous.execution_id,
                                  turn_result=validated_result,
                                  response=validated_result.response if validated_result is not None else None,
                                  debug=body.get("debug") if isinstance(body.get("debug"), dict) else None,
                                  response_trace=body.get("trace") if isinstance(body.get("trace"), str) else None,
                                  runtime_errors=[f"unexpected error parsing evaluation diagnostics: {exc}"])


def _record_client_error(event: AgentStreamEvent, parsed: ParsedResponseData, *, done_received: bool) -> None:
    """Keep evaluation error buckets while the app owns transport and validation."""
    observation = event.observation
    code = event.data.get("code")
    detail = observation.exception_detail or event.data.get("message", "")
    if observation.error_source == "server":
        parsed.runtime_errors.append(f"SSE error: {event.data.get('message') or 'server reported an error'}")
    elif code == "http_error":
        body = (observation.http_body or "").strip()
        parsed.runtime_errors.append(f"HTTP {parsed.http_status}: {body[:300] + ' ...' if len(body) > 300 else body}")
    elif code in {"empty_stream", "missing_final_response"}:
        reason = "done received" if done_received else ("empty stream" if code == "empty_stream" else "stream ended")
        parsed.response_errors.append(f"SSE final_response missing ({reason})")
    elif code == "timeout":
        parsed.runtime_errors.append("SSE stream timeout before final_response" if observation.stream_opened else "request timeout")
    elif code in {"connection_error", "connection_interrupted"}:
        prefix = "SSE stream disconnected before final_response" if observation.stream_opened else "request failed"
        parsed.runtime_errors.append(f"{prefix}: {detail}")
    elif code == "invalid_stream":
        parsed.response_errors.append(f"SSE protocol error: {detail}")
    else:
        parsed.runtime_errors.append(f"unexpected error: {detail}")


def _run_turn(client: AgentSessionClient, query: str, *,
              role: str = "question", prior_turns: list[ScenarioTurnResult] | None = None,
              expectation: SetupTurnExpectation | None = None) -> tuple[ParsedResponseData, ScenarioTurnResult]:
    payload = build_agent_payload(query, client.context, uploads=client.upload_context())
    parsed = ParsedResponseData()
    elapsed = None
    done_received = False
    final_received = False
    client_failed = False
    raw_final = None
    for event in client.stream(query):
        observation = event.observation
        parsed.http_status = observation.http_status or parsed.http_status
        parsed.execution_id = observation.execution_id or parsed.execution_id
        elapsed = round(observation.elapsed_ms)
        if event.event == "final_response" and event.result is not None:
            final = _parse_final(event.data, previous=parsed, validated_result=event.result.to_turn_result(),
                                 expected_partial_problem=expectation.problem_code if expectation else None)
            final.runtime_errors.extend(parsed.runtime_errors)
            final.response_errors.extend(parsed.response_errors)
            parsed = final
            final_received = True
        elif event.event == "done":
            done_received = True
        elif event.event == "error":
            client_failed = client_failed or observation.error_source != "server"
            raw_final = event.data.get("raw_final_response")
            if isinstance(raw_final, dict):
                # Malformed public responses stay unusable; keep their diagnostics
                # and raw envelope so contract failures do not hide model usage.
                rejected = _parse_final(raw_final, previous=parsed)
                rejected.response = None
                rejected.response_text = ""
                rejected.actions = []
                rejected.runtime_errors.extend(parsed.runtime_errors)
                rejected.response_errors.extend(parsed.response_errors)
                parsed = rejected
            _record_client_error(event, parsed, done_received=done_received)
    if not final_received and not parsed.response_errors and not client_failed:
        parsed.response_errors.append(f"SSE final_response missing ({'done received' if done_received else 'stream ended'})")
    if parsed.response is not None:
        if parsed.evidence_assessment is None:
            parsed.evidence_assessment = assess_evidence_scope(
                response=parsed.response, provenance=parsed.answer_provenance,
                observed_hits=parsed.observed_hits, tool_calls=parsed.tool_calls,
                session_id=payload["session_id"], prior_turns=prior_turns or [],
            )
        parsed.response_errors.extend(f"evidence scope: {error}" for error in parsed.evidence_assessment.errors
                                      if error not in parsed.response_errors)
    turn = ScenarioTurnResult(
        role=role, llm_calls=parsed.llm_calls,
        query=query, request_payload=payload, http_status=parsed.http_status, execution_id=parsed.execution_id,
        response=parsed.response, turn_result=parsed.turn_result, trace=parsed.response_trace, debug=parsed.debug,
        upload_manifest=client.manifest, question_response_ms=elapsed,
        runtime_errors=parsed.runtime_errors, response_errors=parsed.response_errors,
        raw_final_response=raw_final,
        answer_provenance=parsed.answer_provenance, evidence_assessment=parsed.evidence_assessment,
        observed_hits=parsed.observed_hits, tool_calls=parsed.tool_calls,
        execution_evidence=parsed.execution_evidence,
    )
    return parsed, turn


def _setup_outcome_matches(parsed: ParsedResponseData, expectation: SetupTurnExpectation | None) -> bool:
    outcome = parsed.turn_result
    if outcome is None:
        return False
    if expectation is None:
        return outcome.status in {"completed", "needs_input"}
    return (outcome.status == expectation.status
            and (outcome.problem.code if outcome.problem else None) == expectation.problem_code)


def _run_single_case(
    *,
    run_id: str,
    endpoint: str,
    fixtures_path: Path,
    case: BenchmarkCase,
    timeout_seconds: int,
    judge: LLMJudge,
    config: BenchmarkConfig,
    resolved_weights: ResolvedWeights,
    live_slack: BenchmarkLiveSlackConfig | None = None,
    verified_uploads: Mapping[str, ApprovedUpload] | None = None,
) -> CaseResult:
    started = time.monotonic()
    created_at = datetime.now(timezone.utc).isoformat()
    session_id = str(uuid4())
    resolved_live_slack = live_slack or BenchmarkLiveSlackConfig()
    context = case_context(endpoint=endpoint, session_id=session_id, case=case,
                           timeout_seconds=timeout_seconds, live_slack=resolved_live_slack)
    client = AgentSessionClient(context)
    endpoint_url = endpoint.rstrip("/") + "/agent/stream"
    parsed_response = ParsedResponseData()
    turns: list[ScenarioTurnResult] = []
    attachment_setup_ms = None
    question_response_ms = None
    # No question payload exists before its attachment context is confirmed.
    request_payload = {}
    files = []
    staged = None
    try:
        approved_files = (select_approved_uploads(case.resolved_upload_fixtures, verified_uploads)
                          if verified_uploads is not None else None)
        client.refresh_uploads()
        files = approved_files if approved_files is not None else resolve_fixture_uploads(fixtures_path, case)
        if files:
            settings = get_settings()
            staged = client.stage_files(files, get_upload_session_dir(session_id),
                                        max_files=settings.upload_max_files, max_file_mib=settings.upload_max_file_mib,
                                        max_total_mib=settings.upload_max_total_mib)
            if staged.errors:
                raise ValueError("; ".join(staged.errors))
            manifest = client.manifest
            client.sync_uploads(build_upload_sync_request(manifest, files=staged.files))
            discard_staged_files(staged.files, get_upload_session_dir(session_id))
        if approved_files is not None:
            verify_upload_manifest(approved_files, client.manifest)
        attachment_setup_ms = round((time.monotonic() - started) * 1000)
        for index, query in enumerate([*case.setup_turns, case.query]):
            if index and approved_files is not None:
                # The app adopts each answer's manifest for the following turn.
                verify_upload_manifest(approved_files, client.manifest)
            is_setup = index < len(case.setup_turns)
            expectation = case.setup_expectations[index] if is_setup and case.setup_expectations is not None else None
            parsed, turn = _run_turn(client, query, role="setup" if is_setup else "question",
                                     prior_turns=turns, expectation=expectation)
            turns.append(turn)
            if index == len(case.setup_turns):
                parsed_response = parsed
                request_payload = turn.request_payload
                question_response_ms = turn.question_response_ms
            elif (not _setup_outcome_matches(parsed, expectation)
                  or parsed.runtime_errors or parsed.response_errors
                  or (parsed.debug_errors and not (expectation and expectation.status == "partial"))
                  or parsed.planner_errors or parsed.missing_required_debug_fields
                  or parsed.debug_observability_status == "failed"):
                parsed_response = ParsedResponseData(
                    http_status=parsed.http_status,
                    runtime_errors=[f"setup turn {index + 1} failed; dependent question was not sent",
                                    *parsed.runtime_errors],
                    response_errors=list(parsed.response_errors),
                )
                break
    except UploadAPIError as exc:
        parsed_response.http_status = exc.status_code or 0
        parsed_response.runtime_errors.append(f"attachment error {exc.code}: {exc}")
        if staged is not None and (exc.status_code in {400, 409, 413, 422}
                                   or (exc.status_code == 503 and exc.code == "UPLOAD_INDEX_FAILED")):
            discard_staged_files(staged.files, get_upload_session_dir(session_id))
    except (OSError, ValueError) as exc:
        parsed_response.runtime_errors.append(f"scenario preparation failed: {exc}")
    except Exception as exc:
        parsed_response.runtime_errors.append(f"unexpected error: {exc}")
    finally:
        if attachment_setup_ms is None:
            attachment_setup_ms = round((time.monotonic() - started) * 1000)
    scenario_total_ms = round((time.monotonic() - started) * 1000)
    cleanup_errors = []
    # Only confirmed state can be cleaned synchronously. After an uncertain POST,
    # leave resources to the server's normal TTL/LRU cleanup; never replay it.
    if client.manifest is not None and client.manifest.files:
        try:
            client.sync_uploads(build_upload_sync_request(client.manifest, clear=True))
        except UploadAPIError as exc:
            cleanup_errors.append(f"{exc.code}: {exc}")
    result = build_case_result(
        run_id=run_id,
        endpoint_url=endpoint_url,
        case=case,
        judge=judge,
        config=config,
        resolved_weights=resolved_weights,
        session_id=session_id,
        created_at=created_at,
        request_payload=request_payload,
        question_response_ms=question_response_ms,
        parsed_response=parsed_response,
        slack_delivery_required=resolved_live_slack.applies_to_case(case),
        prior_turns=turns[:len(case.setup_turns)],
        scenario_turns=turns,
    )
    result.attachment_setup_ms = attachment_setup_ms
    result.scenario_total_ms = scenario_total_ms
    result.attachment_fingerprints = ({name: upload.fingerprint for name, upload in
                                      zip(case.resolved_upload_fixtures, files, strict=True) if upload.fingerprint is not None}
                                     if files else {})
    result.cleanup_errors = cleanup_errors
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
    release_review: Path | None = None,
    release_design: Path | None = None,
) -> tuple[Path, list[CaseResult], RunSummary]:
    if track == "release" and limit is not None:
        raise ValueError("Release runs cannot use --limit; the complete approved suite is required.")
    if limit is not None and limit <= 0:
        raise ValueError("--limit must be a positive integer.")
    if track == "release" and not config.judge_enabled:
        raise ValueError(
            "Release runs require judge evaluation; judge_enabled is false. "
            "Run a smoke track for rule-only diagnostics."
        )
    from ..release_dataset import DEFAULT_DESIGN, load_release_input, load_reviewed_release

    verified_uploads = None
    dataset_approval = {"status": "not_verified"}
    approved = None
    release_suite = None
    if track == "release":
        cases, approved = load_release_input(fixtures_path, review=release_review, design=release_design)
        if approved is None:
            raise ValueError("Release runs require an approved complete suite.")
        release_suite = ReleaseSuiteContract(
            approved_case_ids=tuple(case.case_id for case in approved.cases),
            review_sha256=approved.review_sha256,
            candidate_sha256=approved.inspection["sha256"],
        )
        if not release_suite_gate(
            track=track, requested_limit=limit, planned_ids={case.case_id for case in cases},
            release_suite=release_suite,
        ).passed:
            raise ValueError("Release execution must include every approved case exactly once.")
    elif release_review is not None:
        approved = load_reviewed_release(
            fixtures_path, review=release_review, design=release_design or DEFAULT_DESIGN,
        )
        cases = approved.cases
    else:
        cases = load_cases_jsonl(fixtures_path)
    if approved is not None:
        verified_uploads = approved.uploads
        dataset_approval = {
            "status": "verified", "review_sha256": approved.review_sha256,
            "candidate_sha256": approved.inspection["sha256"],
            "artifact_hashes": approved.inspection["artifact_hashes"],
        }
    requested_limit = _normalize_limit(limit)
    if requested_limit is not None:
        cases = cases[:requested_limit]
    if not cases:
        raise ValueError("No benchmark cases found.")
    # Resolve every selected case before creating a judge or executing any scenario.
    resolved_case_weights = [resolve_case_weights(case=case, profiles=config.weights) for case in cases]
    if track == "release":
        missing_save_contracts = [case.case_id for case in cases
                                  if "save_text" in case.expected_tools and case.save_expectation is None]
        if missing_save_contracts:
            raise ValueError("Release save cases require save_expectation: " + ", ".join(missing_save_contracts))
    _validate_live_slack_targets(cases=cases, live_slack=live_slack)

    started_at = datetime.now(timezone.utc)
    run_id = started_at.strftime("%Y%m%d_%H%M%S_%f")
    run_dir = output_root / run_id
    results: list[CaseResult] = []
    manifest: dict[str, object] = {
        "run_id": run_id, "status": "running", "track": track,
        "started_at_utc": started_at.isoformat(), "finished_at_utc": None,
        "endpoint": endpoint, "fixtures_path": str(fixtures_path), "config_path": str(config_path),
        "config": config.model_dump(mode="json"), "case_ids": [case.case_id for case in cases],
        "completed_cases": 0, "dataset_approval": dataset_approval,
        "execution_options": (live_slack or BenchmarkLiveSlackConfig()).model_dump(mode="json"),
    }
    with _record_run_progress(run_dir, manifest, results) as records:
        judge = LLMJudge(
            model_name=config.judge_model,
            enabled=config.judge_enabled,
            max_input_tokens=config.judge_max_input_tokens,
            max_output_tokens=config.judge_max_output_tokens,
        )
        for index, (case, resolved_weights) in enumerate(zip(cases, resolved_case_weights, strict=True), 1):
            result = _run_single_case(
                run_id=run_id,
                endpoint=endpoint,
                fixtures_path=fixtures_path,
                case=case,
                timeout_seconds=config.request_timeout_seconds,
                judge=judge,
                config=config,
                resolved_weights=resolved_weights,
                live_slack=live_slack,
                verified_uploads=verified_uploads,
            )
            # Keep the case observation before run-end artifact checks can change
            # its assessment. Only final outputs establish release eligibility.
            records.write(result.model_dump_json() + "\n")
            records.flush()
            os.fsync(records.fileno())
            results.append(result)
            manifest["completed_cases"] = len(results)
            _write_run_manifest(run_dir, manifest)
            composite_text = (
                f"{result.composite_quality_score:.3f}"
                if result.composite_quality_score is not None
                else "n/a"
            )
            print(
                f"[{index}/{len(cases)}] {case.case_id} composite={composite_text} "
                f"judge={result.judge_status} release={'PASS' if result.release_pass else 'FAIL'} "
                f"status={result.http_status} latency={result.question_response_ms}ms"
            )

        from ..save_outcomes import revalidate_saved_artifacts
        revalidate_saved_artifacts(cases=cases, results=results, timeout=config.request_timeout_seconds)
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
            attachment_fingerprints={f"{result.case_id}/{name}": digest for result in results
                                     for name, digest in result.attachment_fingerprints.items()},
            execution_options=(live_slack or BenchmarkLiveSlackConfig()).model_dump(mode="json"),
            release_suite=release_suite,
        )
        summary.audit_metrics["dataset_approval"] = dataset_approval
        write_run_outputs(output_dir=run_dir, results=results, summary=summary)
    latest_run_pointer_path(output_root, track).write_text(run_id + "\n", encoding="utf-8")
    return run_dir, results, summary


@contextmanager
def _record_run_progress(run_dir: Path, manifest: dict[str, object],
                         results: list[CaseResult]) -> Iterator[TextIO]:
    run_dir.mkdir(parents=True, exist_ok=False)
    _write_run_manifest(run_dir, manifest)
    try:
        with (run_dir / "case_results.jsonl").open("x", encoding="utf-8") as records:
            yield records
    except (KeyboardInterrupt, SystemExit):
        manifest["status"] = "interrupted"
        raise
    except Exception:
        manifest["status"] = "failed"
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest["completed_cases"] = len(results)
        manifest["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        _write_run_manifest(run_dir, manifest)


def _write_run_manifest(run_dir: Path, manifest: dict[str, object]) -> None:
    pending = run_dir / "manifest.pending.json"
    with pending.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    pending.replace(run_dir / "manifest.json")


__all__ = [
    "_run_single_case",
    "latest_run_pointer_path",
    "run_online_benchmark",
]
