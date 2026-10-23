"""Expected partial turns remain usable answers for a dependent user request."""
from __future__ import annotations

import json
from urllib.parse import unquote

import pytest
import requests
from pydantic import ValidationError

from src.core.answer_schema import ActionReceipt, AnswerDocument, export_answer_text, finalize_answer
from src.core.contracts.debug import DebugPayload
from src.core.contracts.provenance import AnswerProvenance, AnswerSource
from src.core.llm_errors import make_problem
from src.core.save_contract import SaveOperation
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.online_runner.case_runner import run_online_benchmark
from src.infra.saved_artifacts import ArtifactError, read_saved_artifact, save_artifact


def _answer():
    return finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{
            "text": "확인한 내용만 제공합니다. 나머지 동작은 확인하지 못했습니다.",
            "basis": "interaction", "refs": [],
        }]}],
    }), [])


def _body(answer, *, setup, session_id):
    request_id = "setup-request" if setup else "question-request"
    provenance = AnswerProvenance(
        body_kind="compose" if setup else "copy_answer",
        response_hash=answer.content_hash, evidence_packet=[],
        source=None if setup else AnswerSource(
            ref="previous", response_hash=answer.content_hash, citation_ids=[],
        ),
    )
    debug = DebugPayload(
        answer_provenance=provenance, llm_calls=[],
        execution_evidence={"schema_version": 1, "request_id": request_id,
                            "status": "complete", "events": []},
    )
    return {
        "status": "partial" if setup else "completed",
        "response": answer.model_dump(mode="json"),
        "problem": make_problem("evidence_insufficient", "validation").model_dump(mode="json") if setup else None,
        "request_id": request_id,
        "upload_manifest": {"epoch": session_id, "revision": 0, "files": []},
        "debug": debug.model_dump(mode="json"),
    }


def _run_scenario(monkeypatch, tmp_path, *, expectation, setup_code="evidence_insufficient",
                  setup_status="partial", defect=None, save_retry=False, question_partial=False):
    answer = _answer()
    sent = []
    operation = None
    artifact_dir = tmp_path / "artifacts"

    def http_request(_session, method, url, **kwargs):
        nonlocal operation
        response = requests.Response()
        response.status_code = 200
        if "/download/" in url:
            manifest, payload = read_saved_artifact(artifact_dir, unquote(url.rsplit("/", 1)[1]))
            response._content = payload
            response.headers.update({"X-Save-Binding-SHA256": manifest.operation.binding_sha256,
                                     "X-Artifact-Id": manifest.artifact.artifact_id})
        elif method.lower() == "get":
            session_id = url.split("/sessions/", 1)[1].split("/", 1)[0]
            body = {"epoch": session_id, "revision": 0, "files": []}
            response._content = json.dumps(body).encode()
            response.headers["Content-Type"] = "application/json"
        else:
            payload = kwargs["json"]
            sent.append(payload)
            body = _body(answer, setup=len(sent) == 1, session_id=payload["session_id"])
            prefix = ""
            if save_retry:
                content = export_answer_text(answer, include_sources=True)
                invocation_id = "setup-save" if len(sent) == 1 else "retry-save"
                operation = operation or SaveOperation.for_text(
                    content, operation_id="save-operation", session_id=payload["session_id"],
                    request_id="logical-request", contract_revision=1, target_kind="compose",
                    source_hash=answer.content_hash, answer_hash=answer.content_hash,
                )
                if len(sent) == 1:
                    artifact_dir.write_text("controlled write failure", encoding="utf-8")
                    with pytest.raises(ArtifactError) as failure:
                        save_artifact(artifact_dir, content, operation, ttl_seconds=60)
                    artifact_dir.unlink()
                    receipt = ActionReceipt(
                        invocation_id=invocation_id, kind="save_text", status="error",
                        operation=operation, verification="failed", error_code=failure.value.code,
                        error=str(failure.value),
                    )
                    body["debug"]["errors"] = [str(failure.value)]
                else:
                    manifest, path = save_artifact(artifact_dir, content, operation, ttl_seconds=60)
                    receipt = ActionReceipt(
                        invocation_id=invocation_id, kind="save_text", status="success",
                        operation=operation, verification="verified", artifact=manifest.artifact,
                        file_path=str(path),
                    )
                    body["debug"]["answer_provenance"]["source"]["ref"] = "pending"
                body["response"]["actions"] = [receipt.model_dump(mode="json")]
                body["debug"]["answer_provenance"].update({
                    "request_id": "logical-request", "contract_revision": len(sent),
                    "save_operation_binding_sha256": operation.binding_sha256,
                })
                body["debug"]["tool_calls"] = ["save_text"]
                body["debug"]["tool_call_count"] = 1
                body["debug"]["execution_evidence"]["events"] = [
                    {"sequence": index, "invocation_id": invocation_id, "tool_name": "save_text", "phase": phase}
                    for index, phase in enumerate(("started", "failed" if len(sent) == 1 else "succeeded"), 1)
                ]
            if len(sent) == 1:
                body["status"] = setup_status
                body["problem"] = (make_problem(setup_code, "validation").model_dump(mode="json")
                                   if setup_status in {"partial", "failed", "refused"} else None)
                if setup_status in {"failed", "needs_input", "refused"}:
                    body["response"] = None
                    body["debug"]["answer_provenance"] = None
                    body["message"] = "추가 정보를 알려 주세요."
                if defect == "missing_answer":
                    body["response"] = None
                elif defect == "debug_contract":
                    body["debug"].pop("tool_calls")
                elif defect == "planner_error":
                    body["debug"]["planner_errors"] = ["planner schema was invalid"]
                elif defect == "provenance_mismatch":
                    body["debug"]["answer_provenance"]["response_hash"] = "0" * 64
                elif defect == "diagnostic_message":
                    body["debug"]["errors"] = ["the requested source did not supply all information"]
                elif defect == "sse_error":
                    prefix = 'event: error\ndata: {"message":"unexpected server failure"}\n\n'
                elif defect == "timeout":
                    raise requests.Timeout("request timeout")
            elif question_partial:
                body["status"] = "partial"
                body["problem"] = make_problem("evidence_insufficient", "validation").model_dump(mode="json")
            response._content = (prefix + "event: final_response\ndata: " + json.dumps(body) + "\n\n").encode()
            response.headers["Content-Type"] = "text/event-stream"
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", http_request)
    case = {
        "case_id": "partial-answer-copy", "category": "tool_action",
        "setup_turns": ["확인할 수 있는 내용과 확인하지 못한 한계를 알려줘."],
        "setup_forbidden_tools": [["save_text", "slack_notify"]],
        "query": "방금 부분 답변을 내용 변경 없이 그대로 다시 보여줘.",
        "forbidden_tools": ["save_text", "slack_notify"],
    }
    if expectation is not None:
        case["setup_expectations"] = [expectation]
    if save_retry:
        case.update({
            "setup_turns": ["준비한 본문을 TXT 파일로 저장해줘."],
            "setup_forbidden_tools": [["slack_notify"]],
            "query": "그 본문 그대로 저장을 다시 시도해줘.",
            "expected_tools": ["save_text"], "forbidden_tools": ["slack_notify"],
            "save_expectation": {"outcome": "required_success", "target": {
                "kind": "setup_answer", "setup_turn_index": 0,
            }},
        })
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(json.dumps(case, ensure_ascii=False) + "\n", encoding="utf-8")
    _, results, _ = run_online_benchmark(
        fixtures_path=fixture, endpoint="http://scenario.test",
        config=BenchmarkConfig(judge_enabled=False), config_path=tmp_path / "config.json",
        output_root=tmp_path / "results", track="smoke",
    )
    return sent, results[0]


def test_expected_partial_answer_can_be_selected_by_followup(monkeypatch, tmp_path):
    sent, result = _run_scenario(monkeypatch, tmp_path, expectation={
        "status": "partial", "problem_code": "evidence_insufficient",
    })

    assert len(sent) == 2, result.runtime_errors
    assert sent[0]["session_id"] == sent[1]["session_id"]
    assert result.turn_result.status == "completed"
    assert result.response.content_hash == result.scenario_turns[0].response.content_hash
    assert result.evidence_assessment.status == "complete"
    assert not result.runtime_errors
    setup = result.scenario_turns[0]
    assert setup.turn_result.status == "partial"
    assert setup.turn_result.problem.code == "evidence_insufficient"
    assert not setup.runtime_errors


@pytest.mark.parametrize("expectation,setup_status,setup_code,defect", [
    (None, "partial", "evidence_insufficient", None),
    ({"status": "partial", "problem_code": "save_failed"}, "partial", "evidence_insufficient", None),
    ({"status": "partial", "problem_code": "evidence_insufficient"}, "completed", "evidence_insufficient", None),
    ({"status": "partial", "problem_code": "evidence_insufficient"}, "failed", "evidence_insufficient", None),
    ({"status": "partial", "problem_code": "evidence_insufficient"}, "partial", "provider_unavailable", None),
    *[({"status": "partial", "problem_code": "evidence_insufficient"}, "partial", "evidence_insufficient", defect)
      for defect in ("missing_answer", "debug_contract", "planner_error", "provenance_mismatch", "sse_error", "timeout")],
])
def test_expected_partial_does_not_hide_other_setup_failures(
    monkeypatch, tmp_path, expectation, setup_status, setup_code, defect,
):
    sent, result = _run_scenario(monkeypatch, tmp_path, expectation=expectation,
                               setup_status=setup_status, setup_code=setup_code, defect=defect)

    assert len(sent) == 1
    assert result.runtime_errors[0] == "setup turn 1 failed; dependent question was not sent"
    assert result.response is None
    if defect in {"missing_answer", "timeout"}:
        assert result.scenario_turns[0].turn_result is None
    else:
        assert result.scenario_turns[0].turn_result is not None
    assert result.product_pass is False


def test_expected_partial_preserves_diagnostic_messages_without_aborting(monkeypatch, tmp_path):
    sent, result = _run_scenario(monkeypatch, tmp_path, expectation={
        "status": "partial", "problem_code": "evidence_insufficient",
    }, defect="diagnostic_message")

    assert len(sent) == 2
    assert result.scenario_turns[0].debug["errors"] == ["the requested source did not supply all information"]
    assert result.evidence_assessment.status == "complete"


def test_expected_save_failure_continues_to_verified_retry_of_same_operation(monkeypatch, tmp_path):
    sent, result = _run_scenario(monkeypatch, tmp_path, expectation={
        "status": "partial", "problem_code": "save_failed",
    }, setup_code="save_failed", save_retry=True)

    assert len(sent) == 2, result.runtime_errors
    failed = result.scenario_turns[0]
    assert failed.turn_result.status == "partial"
    assert failed.turn_result.problem.code == "save_failed"
    assert failed.response.actions[0].error_code == "write_failed"
    assert result.turn_result.status == "completed"
    assert failed.response.actions[0].operation == result.actions[0].operation
    assert result.save_assessment.status == "verified", result.save_assessment.failure_codes
    assert result.save_assessment.phase == "run_end"
    assert result.save_assessment.expected_sha256 == result.save_assessment.observed_sha256
    assert result.evidence_assessment.status == "complete"
    assert not result.runtime_errors


def test_expected_setup_partial_does_not_accept_partial_final_answer(monkeypatch, tmp_path):
    sent, result = _run_scenario(monkeypatch, tmp_path, expectation={
        "status": "partial", "problem_code": "evidence_insufficient",
    }, question_partial=True)

    assert len(sent) == 2
    assert result.turn_result.status == "partial"
    assert result.turn_result.problem.code == "evidence_insufficient"
    assert any("evidence_insufficient (validation)" in error for error in result.runtime_errors)
    assert result.product_pass is False


@pytest.mark.parametrize("expectations", [
    [],
    [{"status": "partial"}],
    [{"status": "completed", "problem_code": "save_failed"}],
    [{"status": "partial", "problem_code": "provider_unavailable"}],
    [{"status": "failed", "problem_code": "save_failed"}],
    [{"status": "partial", "problem_code": "save_failed", "ignore_errors": True}],
])
def test_setup_expectations_reject_incomplete_or_overbroad_declarations(expectations):
    with pytest.raises(ValidationError):
        BenchmarkCase.model_validate({
            "case_id": "bad-setup", "category": "tool_action", "query": "retry",
            "setup_turns": ["prepare"], "setup_expectations": expectations,
        })
