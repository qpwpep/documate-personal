"""Verify save obligations against the public artifact download boundary."""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from pathlib import PurePosixPath, PureWindowsPath
from urllib.parse import quote

import requests

from src.core.answer_schema import ActionReceipt, AnswerResponse, export_answer_text
from src.core.contracts.provenance import AnswerProvenance
from .config_models import BenchmarkCase
from .result_models import CaseResult, SaveAssessment, ScenarioTurnResult


def _download(endpoint: str, filename: str, timeout: int) -> requests.Response:
    root = endpoint.removesuffix("/agent/stream").rstrip("/")
    return requests.get(f"{root}/download/{quote(filename, safe='')}", timeout=timeout)


def _error_code(response: requests.Response) -> str | None:
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return None
    return detail.get("code") if isinstance(detail, dict) else None


def assess_save_outcome(
    *, case: BenchmarkCase, response: AnswerResponse | None, actions: list[ActionReceipt],
    called_tools: list[str], prior_turns: list[ScenarioTurnResult], session_id: str,
    endpoint: str, timeout: int = 10, provenance: AnswerProvenance | None = None,
    phase: str = "case",
) -> SaveAssessment:
    """A receipt is checked against an independently selected body and fetched bytes."""
    expectation = case.save_expectation
    receipts = [action for action in actions if action.kind == "save_text"]
    attempted = "save_text" in called_tools or any(
        receipt.status != "skipped" or receipt.operation is not None or receipt.artifact is not None
        or receipt.file_path is not None for receipt in receipts
    )
    details = {"outcome": expectation.outcome if expectation else None, "phase": phase,
               "checked_at_utc": datetime.now(timezone.utc).isoformat()}

    def fail(code: str, *, unknown: bool = False) -> SaveAssessment:
        return SaveAssessment(status="unverifiable" if unknown else "failed", passed=False,
                              failure_codes=[code], **details)

    def verified() -> SaveAssessment:
        return SaveAssessment(status="verified", passed=True, **details)

    if expectation is None:
        if "save_text" in case.expected_tools:
            return fail("save_expectation_missing", unknown=True)
        if attempted:
            return fail("save_unexpected_execution")
        return SaveAssessment(status="not_applicable", **details)
    if expectation.outcome == "must_not_execute":
        return fail("save_unexpected_execution") if attempted else verified()
    if not receipts:
        return fail("save_receipt_missing")
    if len(receipts) != 1:
        return fail("save_receipt_ambiguous")
    receipt = receipts[0]
    if expectation.outcome == "required_success" and receipt.status in {"error", "skipped"}:
        return fail("save_failed")
    if receipt.status == "unknown":
        return fail("save_unverifiable", unknown=True)
    operation = receipt.operation
    if operation is None:
        return fail("save_unverifiable", unknown=True)
    if operation.session_id != session_id:
        return fail("save_request_mismatch")
    if provenance is None or not provenance.request_id or provenance.contract_revision is None:
        return fail("save_request_unverifiable", unknown=True)
    if (operation.request_id != provenance.request_id
            or operation.contract_revision > provenance.contract_revision):
        return fail("save_request_mismatch")
    if not provenance.save_operation_binding_sha256:
        return fail("save_request_unverifiable", unknown=True)
    if operation.binding_sha256 != provenance.save_operation_binding_sha256:
        return fail("save_operation_mismatch")
    if response is None:
        return fail("save_target_unverifiable", unknown=True)
    target = expectation.target
    expected = response
    pending_replay = (
        provenance.body_kind == "copy_answer" and provenance.source is not None
        and provenance.source.ref == "pending"
        and provenance.source.response_hash == operation.answer_hash
        and operation.contract_revision < provenance.contract_revision
    )
    if target.kind == "setup_answer":
        index = target.setup_turn_index
        if index >= len(prior_turns) or prior_turns[index].response is None:
            return fail("save_target_unverifiable", unknown=True)
        expected = prior_turns[index].response
        if not pending_replay and (operation.target_kind != "copy_answer" or operation.source_hash != expected.content_hash):
            return fail("save_target_mismatch")
    elif provenance is not None:
        if not pending_replay and operation.target_kind != provenance.body_kind:
            return fail("save_target_mismatch")
        if not pending_replay and provenance.source is not None and operation.source_hash != provenance.source.response_hash:
            return fail("save_target_mismatch")
        if operation.target_kind in {"compose", "extract"} and operation.source_hash != response.content_hash:
            return fail("save_target_mismatch")
    expected_bytes = export_answer_text(expected, include_sources=True).encode("utf-8-sig")
    expected_hash = hashlib.sha256(expected_bytes).hexdigest()
    details.update(expected_sha256=expected_hash, expected_byte_count=len(expected_bytes))
    if (operation.answer_hash != expected.content_hash or operation.payload_sha256 != expected_hash
            or operation.byte_count != len(expected_bytes)):
        return fail("save_content_mismatch")
    # A copy case independently chooses the preparation answer. The final body
    # cannot silently select another answer and still satisfy the same save.
    if export_answer_text(response, include_sources=True).encode("utf-8-sig") != expected_bytes:
        return fail("save_target_mismatch")

    if expectation.outcome == "expected_failure":
        if (receipt.status != "error" or receipt.verification != "failed"
                or receipt.error_code not in expectation.error_codes or not receipt.error
                or receipt.artifact is not None):
            return fail("save_expected_failure_mismatch")
        from src.infra.saved_artifacts import artifact_filename
        filename = artifact_filename(operation)
        try:
            observation = _download(endpoint, filename, timeout)
        except requests.RequestException:
            return fail("save_unverifiable", unknown=True)
        if observation.status_code == 404 and _error_code(observation) in {"manifest_missing", "artifact_missing"}:
            return verified()
        if observation.status_code in {200, 409, 410}:
            return fail("save_unexpected_artifact")
        return fail("save_unverifiable", unknown=True)

    artifact = receipt.artifact
    if receipt.status != "success" or receipt.verification != "verified" or artifact is None:
        return fail("save_unverifiable", unknown=True)
    details["artifact_id"] = artifact.artifact_id
    if not receipt.file_path or artifact.filename not in {
        PurePosixPath(receipt.file_path).name, PureWindowsPath(receipt.file_path).name,
    }:
        return fail("save_artifact_mismatch")
    if artifact.sha256 != expected_hash or artifact.byte_count != len(expected_bytes):
        return fail("save_content_mismatch")
    try:
        observation = _download(endpoint, artifact.filename, timeout)
    except requests.RequestException:
        return fail("save_unverifiable", unknown=True)
    if observation.status_code in {404, 410}:
        return fail("save_artifact_missing")
    if observation.status_code == 409:
        return fail("save_artifact_mismatch")
    if observation.status_code != 200:
        return fail("save_unverifiable", unknown=True)
    payload = observation.content
    details.update(observed_sha256=hashlib.sha256(payload).hexdigest(), observed_byte_count=len(payload))
    if payload != expected_bytes:
        return fail("save_content_mismatch")
    if (observation.headers.get("X-Save-Binding-SHA256") != operation.binding_sha256
            or observation.headers.get("X-Artifact-Id") != artifact.artifact_id):
        return fail("save_artifact_binding_mismatch")
    return verified()


def assess_setup_saves(*, turns: list[ScenarioTurnResult], session_id: str, endpoint: str,
                      timeout: int = 10, phase: str = "case") -> dict[int, SaveAssessment]:
    """Every successful setup save creates a preservation obligation for the run."""
    assessments = {}
    for index, turn in enumerate(turns):
        if turn.response is None or not any(action.kind == "save_text" and action.status == "success"
                                            for action in turn.response.actions):
            continue
        expectation = BenchmarkCase(
            case_id=f"setup-save-{index}", category="tool_action", query=turn.query,
            save_expectation={"outcome": "required_success", "target": {"kind": "final_answer"}},
        )
        assessments[index] = assess_save_outcome(
            case=expectation, response=turn.response, actions=turn.response.actions, called_tools=turn.tool_calls,
            prior_turns=turns[:index], session_id=session_id, endpoint=endpoint, timeout=timeout,
            provenance=turn.answer_provenance, phase=phase,
        )
    return assessments


def revalidate_saved_artifacts(*, cases: list[BenchmarkCase], results: list[CaseResult], timeout: int = 10) -> None:
    """Observe preservation after all later saves; never retroactively promote a failure."""
    case_map = {case.case_id: case for case in cases}
    for result in results:
        case = case_map.get(result.case_id)
        if case is None:
            continue
        prior_turns = result.scenario_turns[:len(case.setup_turns)]
        setup_assessments = assess_setup_saves(
            turns=prior_turns, session_id=result.session_id, endpoint=result.endpoint, timeout=timeout, phase="run_end",
        )
        for index, assessment in setup_assessments.items():
            original = result.setup_save_assessments.get(index)
            if original is None or original.passed is True:
                result.setup_save_assessments[index] = assessment
        assessments = list(result.setup_save_assessments.values())
        if (result.save_assessment is not None and result.save_assessment.passed is True
                and case.save_expectation is not None and case.save_expectation.outcome != "must_not_execute"):
            result.save_assessment = assess_save_outcome(
                case=case, response=result.response, actions=result.actions, called_tools=result.tool_calls,
                prior_turns=prior_turns, session_id=result.session_id, endpoint=result.endpoint,
                timeout=timeout, provenance=result.answer_provenance, phase="run_end",
            )
            assessments.append(result.save_assessment)
        for assessment in assessments:
            if assessment.passed is False:
                result.product_pass = False
                result.release_pass = result.passed = False
                result.gate_failures = list(dict.fromkeys([*result.gate_failures, *assessment.failure_codes]))
                if assessment.status == "unverifiable" and result.eval_validity == "valid":
                    result.eval_validity = "incomplete"
                    if "incomplete_eval" not in result.gate_failures:
                        result.gate_failures.append("incomplete_eval")
