"""Saving is an observed result, never a score that another dimension can offset."""

from __future__ import annotations

import pytest
import requests
from pathlib import Path
from pydantic import ValidationError
from src.core.save_contract import SaveManifest

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.reporting.summary import build_summary
from src.eval.result_models import CaseResult
from src.eval.save_outcomes import revalidate_saved_artifacts
from tests.eval.response_fixtures import answer_provenance, artifact_http_response, plain_response, saved_receipt
from tests.eval.test_release_eval_contract import _final_payload, _judge, _judge_payload, _run_case, _stored_result


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")


def _save_case(**overrides):
    return BenchmarkCase(
        case_id="required-save", category="tool_action", query="이 답변을 저장해줘",
        expected_tools=["save_text"],
        save_expectation={"outcome": "required_success", "target": {"kind": "final_answer"}},
        **overrides,
    )


def test_failed_save_cannot_release_with_perfect_judge_and_rule_scores(tmp_path):
    response = plain_response("요청한 답변 본문")
    response["actions"] = [{"kind": "save_text", "status": "error", "error": "disk full"}]
    result = _run_case(
        _save_case(), _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[_final_payload(response, tool_calls=["save_text"], debug_overrides={"errors": ["disk full"]})],
    )
    assert result.composite_quality_score == 1.0
    assert result.release_pass is False
    assert result.passed is False
    assert "save_failed" in result.gate_failures


def test_called_save_without_receipt_cannot_release(tmp_path):
    result = _run_case(
        _save_case(), _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[_final_payload(plain_response("요청한 답변 본문"), tool_calls=["save_text"])],
    )
    assert result.release_pass is False
    assert "save_receipt_missing" in result.gate_failures


def test_legacy_success_receipt_is_unverifiable_not_release_success(tmp_path):
    response = plain_response("요청한 답변 본문")
    response["actions"] = [{"kind": "save_text", "status": "success", "file_path": "outputs/answer.txt"}]
    result = _run_case(
        _save_case(), _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[_final_payload(response, tool_calls=["save_text"])],
    )
    assert result.release_pass is False
    assert result.eval_validity == "incomplete"
    assert "save_unverifiable" in result.gate_failures


@pytest.fixture
def artifact_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr("src.eval.online_runner.case_runner.uuid4", lambda: "save-contract-session")
    monkeypatch.setattr("src.eval.save_outcomes.requests.get",
                        lambda url, **kwargs: artifact_http_response(tmp_path, url.rsplit("/", 1)[-1]))
    return tmp_path


def _committed_response(root):
    response = plain_response("요청한 답변과 한글 내용을 정확히 보존합니다.\n두 번째 줄")
    response["actions"] = [saved_receipt(response, root, session_id="save-contract-session")]
    return response


def _binding(root):
    manifests = list(root.glob("*.txt.json"))
    return SaveManifest.model_validate_json(manifests[0].read_bytes()).operation.binding_sha256 if manifests else None


def _evaluate(response, root, *, case=None, debug_overrides=None):
    debug = {"answer_provenance": answer_provenance(response, request_id="contract-request", contract_revision=1,
                                                     save_operation_binding_sha256=_binding(root))}
    debug.update(debug_overrides or {})
    return _run_case(
        case or _save_case(), _judge(_judge_payload(1.0)), tmp_path=root,
        turns=[_final_payload(response, tool_calls=["save_text"], debug_overrides=debug)],
    )


def test_real_committed_artifact_can_release_and_preserves_observation_after_reload(artifact_boundary):
    response = _committed_response(artifact_boundary)
    result = _evaluate(response, artifact_boundary)
    assert result.release_pass is True
    assert result.gate_failures == []
    assert result.save_assessment.status == "verified"
    assert result.save_assessment.expected_sha256 == result.save_assessment.observed_sha256
    assert result.save_assessment.expected_byte_count == result.save_assessment.observed_byte_count
    restored = CaseResult.model_validate_json(result.model_dump_json())
    assert restored.save_assessment == result.save_assessment
    assert restored.passed is restored.release_pass is True


@pytest.mark.parametrize("mutation, code", [
    ("delete", "save_artifact_missing"), ("overwrite", "save_artifact_mismatch"),
    ("session", "save_request_mismatch"), ("request", "save_request_mismatch"),
    ("path", "save_artifact_mismatch"),
])
def test_success_receipt_cannot_hide_artifact_or_request_mismatch(artifact_boundary, mutation, code):
    response = _committed_response(artifact_boundary)
    receipt = response["actions"][0]
    path = Path(receipt["file_path"])
    if mutation == "delete":
        path.unlink()
    elif mutation == "overwrite":
        path.write_bytes("다른 요청의 답변".encode("utf-8-sig"))
    elif mutation == "session":
        receipt["operation"]["session_id"] = "another-session"
    elif mutation == "request":
        receipt["operation"]["request_id"] = "earlier-request-in-the-same-session"
    else:
        receipt["file_path"] = str(path.with_name("different.txt"))
    result = _evaluate(response, artifact_boundary)
    assert result.release_pass is False
    assert result.eval_validity == "valid"
    assert code in result.gate_failures


def test_evaluator_compares_http_bytes_even_if_server_reports_success(artifact_boundary, monkeypatch):
    response = _committed_response(artifact_boundary)
    observed = artifact_http_response(artifact_boundary, response["actions"][0]["artifact"]["filename"])
    observed._content = b"wrong artifact bytes"
    monkeypatch.setattr("src.eval.save_outcomes.requests.get", lambda *args, **kwargs: observed)
    result = _evaluate(response, artifact_boundary)
    assert result.release_pass is False
    assert "save_content_mismatch" in result.gate_failures


def test_download_of_unrelated_operation_cannot_verify_matching_body(artifact_boundary, monkeypatch):
    response = _committed_response(artifact_boundary)
    observed = artifact_http_response(artifact_boundary, response["actions"][0]["artifact"]["filename"])
    observed.headers["X-Save-Binding-SHA256"] = "0" * 64
    monkeypatch.setattr("src.eval.save_outcomes.requests.get", lambda *args, **kwargs: observed)
    result = _evaluate(response, artifact_boundary)
    assert result.release_pass is False
    assert "save_artifact_binding_mismatch" in result.gate_failures


def test_unreachable_artifact_is_incomplete_evaluation_not_success(artifact_boundary, monkeypatch):
    response = _committed_response(artifact_boundary)
    def offline(*args, **kwargs):
        raise requests.Timeout("artifact endpoint unavailable")
    monkeypatch.setattr("src.eval.save_outcomes.requests.get", offline)
    result = _evaluate(response, artifact_boundary)
    assert result.release_pass is False
    assert result.eval_validity == "incomplete"
    assert result.save_assessment.status == "unverifiable"


def test_run_end_readback_detects_later_overwrite_and_serializes_failure(artifact_boundary):
    response = _committed_response(artifact_boundary)
    result = _evaluate(response, artifact_boundary)
    assert result.release_pass is True
    Path(response["actions"][0]["file_path"]).write_bytes(b"overwritten later")
    revalidate_saved_artifacts(cases=[_save_case()], results=[result])
    restored = CaseResult.model_validate_json(result.model_dump_json())
    assert restored.release_pass is restored.passed is False
    assert restored.save_assessment.phase == "run_end"
    assert "save_artifact_mismatch" in restored.gate_failures


def test_expected_failure_requires_known_error_and_no_downloadable_artifact(artifact_boundary):
    response = _committed_response(artifact_boundary)
    receipt = response["actions"][0]
    Path(receipt["file_path"]).unlink()
    receipt.update(status="error", verification="failed", artifact=None, file_path=None,
                   error_code="write_failed", error="저장에 실패했습니다.")
    case = BenchmarkCase(
        case_id="expected-save-failure", category="tool_action", query="답변을 저장해줘", expected_tools=["save_text"],
        save_expectation={"outcome": "expected_failure", "target": {"kind": "final_answer"},
                          "error_codes": ["write_failed"]},
    )
    result = _evaluate(response, artifact_boundary, case=case, debug_overrides={"errors": ["write_failed"]})
    assert result.release_pass is True
    assert result.gate_failures == []
    assert result.save_assessment.outcome == "expected_failure"
    receipt["error_code"] = "other_failure"
    rejected = _evaluate(response, artifact_boundary, case=case)
    assert rejected.release_pass is False
    assert "save_expected_failure_mismatch" in rejected.gate_failures


def test_expected_failure_does_not_accept_a_committed_artifact(artifact_boundary):
    response = _committed_response(artifact_boundary)
    response["actions"][0].update(status="error", verification="failed", artifact=None, file_path=None,
                                  error_code="write_failed", error="저장에 실패했습니다.")
    case = BenchmarkCase(
        case_id="expected-save-failure", category="tool_action", query="답변을 저장해줘", expected_tools=["save_text"],
        save_expectation={"outcome": "expected_failure", "target": {"kind": "final_answer"},
                          "error_codes": ["write_failed"]},
    )
    result = _evaluate(response, artifact_boundary, case=case)
    assert result.release_pass is False
    assert "save_unexpected_artifact" in result.gate_failures


def test_expected_failure_cannot_probe_an_unrelated_missing_operation(artifact_boundary):
    response = _committed_response(artifact_boundary)
    response["actions"][0].update(status="error", verification="failed", artifact=None, file_path=None,
                                  error_code="write_failed", error="저장에 실패했습니다.")
    response["actions"][0]["operation"]["operation_id"] = "different-nonexistent-operation"
    case = BenchmarkCase(
        case_id="expected-save-failure", category="tool_action", query="답변을 저장해줘", expected_tools=["save_text"],
        save_expectation={"outcome": "expected_failure", "target": {"kind": "final_answer"},
                          "error_codes": ["write_failed"]},
    )
    result = _evaluate(response, artifact_boundary, case=case)
    assert result.release_pass is False
    assert "save_operation_mismatch" in result.gate_failures


def test_receipt_without_independent_server_operation_binding_is_unverifiable(artifact_boundary):
    response = _committed_response(artifact_boundary)
    result = _evaluate(response, artifact_boundary, debug_overrides={
        "answer_provenance": answer_provenance(response, request_id="contract-request", contract_revision=1),
    })
    assert result.release_pass is False
    assert result.eval_validity == "incomplete"
    assert "save_request_unverifiable" in result.gate_failures


def test_must_not_execute_is_a_distinct_expectation(tmp_path):
    case = BenchmarkCase(case_id="no-save", category="tool_action", query="파일 저장을 취소해줘",
                         save_expectation={"outcome": "must_not_execute"})
    result = _run_case(case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
                       turns=[_final_payload(plain_response("저장을 취소했습니다."))])
    assert result.release_pass is True
    assert result.save_assessment.passed is True
    unexpected = _evaluate(plain_response("저장을 취소했습니다."), tmp_path, case=case)
    assert unexpected.release_pass is False
    assert "save_unexpected_execution" in unexpected.gate_failures


def test_one_required_save_failure_blocks_run_despite_ninety_percent_pass_rate(tmp_path):
    response = plain_response("요청한 답변 본문")
    response["actions"] = [{"kind": "save_text", "status": "error", "error": "disk full"}]
    failed = _evaluate(response, tmp_path)
    good_cases = [BenchmarkCase(case_id=f"good-{i}", category="tool_action", query="설명해줘") for i in range(9)]
    good_results = [_stored_result(case, latency_ms_e2e=1) for case in good_cases]
    summary = build_summary(
        run_id="required-save-run", endpoint="http://fixture", fixtures_path="cases.jsonl", config_path="config.toml",
        track="release", requested_limit=None, config=BenchmarkConfig(), cases=[*good_cases, _save_case()],
        results=[*good_results, failed],
    )
    assert next(gate for gate in summary.gates if gate.name == "release_pass_rate").passed is True
    save_gate = next(gate for gate in summary.gates if gate.name == "save_outcome_contract")
    assert save_gate.passed is False
    assert save_gate.actual == 1
    assert summary.overall_passed is False


def test_stored_release_cannot_claim_success_with_blocking_save_gate(artifact_boundary):
    result = _evaluate(_committed_response(artifact_boundary), artifact_boundary)
    payload = result.model_dump()
    payload["gate_failures"] = ["save_content_mismatch"]
    with pytest.raises(ValidationError, match="blocking gate failures"):
        CaseResult.model_validate(payload)


def test_verified_retry_passes_without_inventing_another_tool_call(artifact_boundary):
    response = _committed_response(artifact_boundary)
    result = _run_case(
        _save_case(), _judge(_judge_payload(1.0)), tmp_path=artifact_boundary,
        turns=[_final_payload(response, tool_calls=[], debug_overrides={
            "answer_provenance": answer_provenance(response, request_id="contract-request", contract_revision=2,
                                                    save_operation_binding_sha256=_binding(artifact_boundary)),
        })],
    )
    assert result.release_pass is True
    assert result.rule_scores["tool_choice"] == 1.0
    assert result.tool_calls == [] and result.tool_call_count == 0
    summary = build_summary(
        run_id="retry-run", endpoint="http://fixture", fixtures_path="cases.jsonl", config_path="config.toml",
        track="release", requested_limit=None, config=BenchmarkConfig(), cases=[_save_case()], results=[result],
    )
    assert summary.metrics.tool_recall == summary.metrics.tool_precision == 1.0
    assert summary.overall_passed is False  # Per-case observation is not a run-end preservation check.
    assert summary.metrics.save_contract_failures == 1
    revalidate_saved_artifacts(cases=[_save_case()], results=[result])
    summary = build_summary(
        run_id="retry-run", endpoint="http://fixture", fixtures_path="cases.jsonl", config_path="config.toml",
        track="release", requested_limit=None, config=BenchmarkConfig(), cases=[_save_case()], results=[result],
    )
    assert summary.overall_passed is True


@pytest.mark.parametrize("target", [{"kind": "final_answer"}, {"kind": "setup_answer", "setup_turn_index": 0}])
def test_pending_retry_retains_the_original_save_operation(artifact_boundary, target):
    response = _committed_response(artifact_boundary)
    prior = dict(response, actions=[])
    case = BenchmarkCase(case_id="pending-retry", category="tool_action", query="다시 저장해줘",
                         expected_tools=["save_text"], setup_turns=["답변을 작성해줘"],
                         save_expectation={"outcome": "required_success", "target": target})
    result = _run_case(case, _judge(_judge_payload(1.0)), tmp_path=artifact_boundary, turns=[
        _final_payload(prior), _final_payload(response, tool_calls=["save_text"], debug_overrides={
            "answer_provenance": answer_provenance(
                response, body_kind="copy_answer", request_id="contract-request", contract_revision=2,
                save_operation_binding_sha256=_binding(artifact_boundary),
                source={"ref": "pending", "response_hash": response["content_hash"], "citation_ids": []},
            ),
        }),
    ])
    assert result.judge_pass is True
    assert result.release_pass is True
    assert result.save_assessment.status == "verified"


def test_nonexecuted_skip_receipt_satisfies_must_not_execute(tmp_path):
    response = plain_response("저장 대상을 확인할 때까지 저장하지 않았습니다.")
    response["actions"] = [{"kind": "save_text", "status": "skipped", "message": "대상 확인 필요"}]
    case = BenchmarkCase(case_id="deferred-save", category="tool_action", query="대상을 먼저 확인해줘",
                         save_expectation={"outcome": "must_not_execute"})
    result = _run_case(case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
                       turns=[_final_payload(response, tool_calls=[])])
    assert result.release_pass is True
    assert result.save_assessment.passed is True


def test_run_end_also_preserves_successful_setup_saves(artifact_boundary):
    prepared = _committed_response(artifact_boundary)
    case = BenchmarkCase(case_id="setup-save", category="tool_action", query="새 파일은 저장하지 마",
                         setup_turns=["이 답변을 작성하고 저장해줘"], save_expectation={"outcome": "must_not_execute"})
    result = _run_case(case, _judge(_judge_payload(1.0)), tmp_path=artifact_boundary, turns=[
        _final_payload(prepared, tool_calls=["save_text"], debug_overrides={
            "answer_provenance": answer_provenance(prepared, request_id="contract-request", contract_revision=1,
                                                    save_operation_binding_sha256=_binding(artifact_boundary)),
        }),
        _final_payload(plain_response("새 파일을 저장하지 않았습니다.")),
    ])
    assert result.release_pass is True
    Path(prepared["actions"][0]["file_path"]).write_bytes(b"later request destroyed earlier save")
    revalidate_saved_artifacts(cases=[case], results=[result])
    assert result.release_pass is False
    assert result.save_assessment.outcome == "must_not_execute" and result.save_assessment.passed is True
    assert result.setup_save_assessments[0].status == "failed"
    assert result.setup_save_assessments[0].phase == "run_end"


def test_setup_target_comes_from_case_not_self_consistent_wrong_receipt(artifact_boundary):
    original = plain_response("사용자가 저장하도록 요청한 이전 답변")
    wrong = plain_response("이전 답변과 무관한 다른 본문")
    wrong["actions"] = [saved_receipt(wrong, artifact_boundary, session_id="save-contract-session",
                                      target_kind="copy_answer", source_hash=original["content_hash"])]
    case = BenchmarkCase(
        case_id="copy-specific-answer", category="tool_action", query="직전 답변을 저장해줘", setup_turns=["먼저 설명해줘"],
        expected_tools=["save_text"], save_expectation={"outcome": "required_success",
            "target": {"kind": "setup_answer", "setup_turn_index": 0}},
    )
    result = _run_case(case, _judge(_judge_payload(1.0)), tmp_path=artifact_boundary, turns=[
        _final_payload(original), _final_payload(wrong, tool_calls=["save_text"], debug_overrides={
            "answer_provenance": answer_provenance(wrong, body_kind="copy_answer",
                                                    request_id="contract-request", contract_revision=1,
                                                    save_operation_binding_sha256=_binding(artifact_boundary)),
        }),
    ])
    assert result.release_pass is False
    assert "save_content_mismatch" in result.gate_failures


@pytest.mark.parametrize("expectation", [
    {"outcome": "required_success"},
    {"outcome": "expected_failure", "target": {"kind": "final_answer"}},
    {"outcome": "required_success", "target": {"kind": "setup_answer"}},
    {"outcome": "must_not_execute", "target": {"kind": "final_answer"}},
])
def test_save_expectations_reject_incomplete_or_contradictory_oracles(expectation):
    with pytest.raises(ValidationError):
        BenchmarkCase(case_id="invalid", category="tool_action", query="저장", save_expectation=expectation)
