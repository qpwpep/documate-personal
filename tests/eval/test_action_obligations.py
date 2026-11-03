"""Required actions are obligations, independent of a model's quality score."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
import requests

from src.core.answer_schema import ActionReceipt, AnswerDocument, export_answer_text, finalize_answer
from src.core.contracts.debug import DebugPayload
from src.core.contracts.provenance import AnswerProvenance
from src.core.save_contract import SaveOperation
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.decisions import policy_for_case
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner.response_parser import parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.result_models import CaseResult, ScenarioTurnResult
from src.eval.save_outcomes import revalidate_saved_artifacts
from src.eval.tool_policy import assess_execution_policy
from src.eval.weighting import resolve_case_weights
from src.infra.saved_artifacts import read_saved_artifact, save_artifact


class _PerfectJudgeBoundary:
    def invoke(self, messages):
        return SimpleNamespace(content=json.dumps({
            "score": 1.0, "reason": "The requested answer is correct.",
            "subscores": {name: 1.0 for name in (
                "answer_quality", "groundedness", "citation_traceability",
                "tool_choice", "format_language",
            )},
        }))


def _receipt(status, invocation_id="slack-call", tool_name="slack_notify"):
    if tool_name == "save_text":
        return ActionReceipt(invocation_id=invocation_id, kind=tool_name, status=status, error="Storage failure")
    delivery = {
        "invocation_id": invocation_id,
        "intent": {"state": "explicit", "selector": {"kind": "channel", "value": "CTEST"}},
        "selection": {
            "request_id": "req-final", "source": "request_input",
            "selector": {"kind": "channel", "value": "CTEST"},
        },
        "target": {"channel_id": "CTEST"},
        "status": {"success": "sent", "error": "not_sent", "unknown": "unknown"}[status],
    }
    if status == "success":
        delivery["message_ts"] = "123.456"
    else:
        delivery["failure"] = {
            "stage": "send",
            "code": "delivery_unknown" if status == "unknown" else "permission_denied",
            "message": "Delivery outcome is uncertain." if status == "unknown" else "Permission denied.",
            "next_action": "verify_delivery" if status == "unknown" else "fix_configuration",
        }
    return ActionReceipt(
        invocation_id=invocation_id, kind="slack_notify", status=status, slack=delivery,
    )


def _events(phase, invocation_id="slack-call", tool_name="slack_notify"):
    return [
        {"sequence": index, "invocation_id": invocation_id, "tool_name": tool_name, "phase": step}
        for index, step in enumerate(("started", phase), 1)
    ]


def _evidence(request_id, events):
    return {"schema_version": 1, "request_id": request_id, "status": "complete", "events": events}


def _answer(actions=()):
    return finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{"text": "요청한 내용을 처리했습니다.", "basis": "interaction"}]}],
    }), [], actions=list(actions))


def _result(*, status="success", terminal="succeeded", missing=False, reused_origin=None,
            tool_name="slack_notify", artifact_dir=None, origin_terminal="failed", evidence_available=True,
            missing_receipt_after_attempt=False, domain_request_id="req-domain",
            execution_id="req-final", evidence_execution_id="req-final", evidence_status="complete",
            provenance_hash=None):
    actions = []
    operation = None
    if artifact_dir is not None:
        body = _answer()
        content = export_answer_text(body, include_sources=True)
        operation = SaveOperation.for_text(
            content, operation_id="save-operation", session_id="session",
            request_id=domain_request_id, contract_revision=1, target_kind="compose",
            source_hash=body.content_hash, answer_hash=body.content_hash,
        )
        manifest, path = save_artifact(artifact_dir, content, operation, ttl_seconds=60)
        actions = [ActionReceipt(
            invocation_id=reused_origin or "slack-call", kind="save_text", status="success",
            file_path=str(path), operation=operation, artifact=manifest.artifact, verification="verified",
        )]
    elif not missing:
        actions = [_receipt(status, reused_origin or "slack-call", tool_name)]
    answer = _answer(actions)
    turns = []
    if reused_origin is None:
        events = [] if missing and not missing_receipt_after_attempt else _events(terminal, tool_name=tool_name)
        tools = [] if missing and not missing_receipt_after_attempt else [tool_name]
    else:
        turns = [ScenarioTurnResult(
            role="setup", query="첫 전송을 시도해줘.", request_payload={},
            request_id="req-setup", response=_answer([
                _receipt("success" if origin_terminal == "succeeded" else "error", reused_origin, tool_name),
            ]),
            execution_evidence=_evidence("req-setup", _events(origin_terminal, reused_origin, tool_name)),
            tool_calls=[tool_name], debug={"retrieval_diagnostics": []},
        )]
        tools = []
        events = [{
            "sequence": 1, "invocation_id": "reuse-call", "tool_name": tool_name,
            "phase": "reused", "origin_invocation_id": reused_origin,
        }]
    events = [
        {"sequence": 1, "invocation_id": "search-call", "tool_name": "tavily_search", "phase": "started"},
        {"sequence": 2, "invocation_id": "search-call", "tool_name": "tavily_search", "phase": "succeeded"},
        *[{**event, "sequence": event["sequence"] + 2} for event in events],
    ]
    tools = ["tavily_search", *tools]
    case = BenchmarkCase(
        case_id="action-obligation", category="tool_action", query="자료를 검색해서 정리한 내용을 Slack에 공유해줘.",
        expected_tools=["tavily_search", tool_name],
        setup_turns=[turn.query for turn in turns], setup_forbidden_tools=[[] for _ in turns],
        save_expectation=(
            {"outcome": "required_success", "target": {"kind": "final_answer"}}
            if tool_name == "save_text" else None
        ),
    )
    debug = DebugPayload(
        tool_calls=tools, tool_call_count=len(tools),
        answer_provenance=AnswerProvenance(
            body_kind="compose", response_hash=provenance_hash or answer.content_hash, evidence_packet=[],
            request_id=domain_request_id, contract_revision=1,
            save_operation_binding_sha256=operation.binding_sha256 if operation is not None else None,
        ),
        execution_evidence={
            **_evidence(evidence_execution_id, events), "status": evidence_status,
        } if evidence_available else None,
    )
    parsed = parse_agent_response({
        "response": answer.model_dump(mode="json"), "request_id": execution_id,
        "trace": f"Request ID: {execution_id}" if execution_id else "",
        "debug": debug.model_dump(mode="json"),
    })
    judge = LLMJudge(model_name="fixture", enabled=False)
    judge.enabled = True
    judge.client = _PerfectJudgeBoundary()
    config = BenchmarkConfig()
    return build_case_result(
        run_id="action-test", endpoint_url="http://fixture", case=case, judge=judge, config=config,
        resolved_weights=resolve_case_weights(case=case, profiles=config.weights),
        session_id="session", created_at="2026-01-01T00:00:00+00:00",
        request_payload={"query": case.query}, latency_ms_e2e=10,
        parsed_response=parsed, prior_turns=turns,
    )


def test_required_slack_failure_cannot_be_offset_by_perfect_judge():
    result = _result(status="error", terminal="failed")

    assert result.judge_status == "succeeded"
    assert result.composite_quality_score >= 0.75
    assert result.release_pass is False
    assert result.action_assessments["slack_notify"].status == "failed"
    assert result.eval_validity == "valid"
    assert result.product_pass is False
    assert "action_slack_notify_failed" in result.gate_failures
    assert "product_quality_below_floor" not in result.gate_failures


@pytest.mark.parametrize("missing, evidence_available", [(False, True), (True, False)])
def test_unobserved_slack_delivery_is_incomplete_not_success(missing, evidence_available):
    result = _result(status="unknown", terminal="failed", missing=missing, evidence_available=evidence_available)

    assert result.release_pass is False
    assert result.action_assessments["slack_notify"].status == "unverifiable"
    assert result.eval_validity == "incomplete"
    assert "action_slack_notify_unverifiable" in result.gate_failures


def test_complete_execution_proof_of_omitted_required_action_is_product_failure():
    result = _result(missing=True)

    assert result.policy_assessment.status == "compliant"
    assert result.action_assessments["slack_notify"].status == "failed"
    assert result.product_pass is False
    assert result.eval_validity == "valid"
    assert result.release_pass is False


@pytest.mark.parametrize("reused_origin", [None, "failed-original-call"])
@pytest.mark.parametrize("tool_name", ["slack_notify", "save_text"])
def test_success_receipt_cannot_relabel_failed_execution(tmp_path, monkeypatch, reused_origin, tool_name):
    def download(url, timeout):
        manifest, payload = read_saved_artifact(tmp_path, unquote(urlparse(url).path.rsplit("/", 1)[1]))
        return SimpleNamespace(status_code=200, content=payload, headers={
            "X-Save-Binding-SHA256": manifest.operation.binding_sha256,
            "X-Artifact-Id": manifest.artifact.artifact_id,
        })

    monkeypatch.setattr(requests, "get", download)
    result = _result(terminal="failed", reused_origin=reused_origin, tool_name=tool_name,
                     artifact_dir=tmp_path if tool_name == "save_text" else None)

    assert result.release_pass is False
    assert result.policy_assessment.status == "indeterminate"
    assert "tool_execution_receipt_outcome_conflict" in result.gate_failures


def test_successful_slack_delivery_is_verified_without_live_mode():
    result = _result()

    assert result.release_pass is True
    assert result.slack_delivery_required is True
    assert result.action_assessments["slack_notify"].status == "verified"
    assert result.action_assessments["slack_notify"].passed is True


def test_successful_reuse_preserves_verified_origin_and_saved_decision():
    result = _result(reused_origin="successful-origin", origin_terminal="succeeded")

    assert result.policy_assessment.status == "compliant"
    assert result.action_assessments["slack_notify"].status == "verified"
    assert result.release_pass is True
    restored = CaseResult.model_validate(result.model_dump(mode="json"))
    assert restored.decision == result.decision
    assert restored.action_assessments == result.action_assessments


@pytest.mark.parametrize("status, assessment_status", [("error", "failed"), ("unknown", "unverifiable")])
def test_save_failure_and_unobserved_save_remain_distinct(status, assessment_status):
    result = _result(status=status, terminal="failed", tool_name="save_text")

    assert result.composite_quality_score >= 0.75
    assert result.release_pass is False
    assert result.save_assessment.status == assessment_status
    assert result.action_assessments["save_text"].status == assessment_status
    assert result.eval_validity == ("valid" if status == "error" else "incomplete")
    assert result.product_pass is (status == "unknown")


def test_artifact_recheck_connection_failure_does_not_claim_product_failure(monkeypatch, tmp_path):
    unavailable = False

    def download(url, timeout):
        if unavailable:
            raise requests.ConnectionError("Artifact download is unavailable.")
        manifest, payload = read_saved_artifact(tmp_path, unquote(urlparse(url).path.rsplit("/", 1)[1]))
        return SimpleNamespace(status_code=200, content=payload, headers={
            "X-Save-Binding-SHA256": manifest.operation.binding_sha256,
            "X-Artifact-Id": manifest.artifact.artifact_id,
        })

    monkeypatch.setattr(requests, "get", download)
    result = _result(tool_name="save_text", artifact_dir=tmp_path)
    assert result.action_assessments["save_text"].status == "verified"
    assert result.release_pass is True

    unavailable = True
    case = BenchmarkCase(
        case_id=result.case_id, category=result.category, query=result.query,
        expected_tools=["tavily_search", "save_text"], setup_forbidden_tools=[],
        save_expectation={"outcome": "required_success", "target": {"kind": "final_answer"}},
    )
    revalidate_saved_artifacts(cases=[case], results=[result])

    assert result.save_assessment.status == "unverifiable"
    assert result.action_assessments["save_text"].status == "unverifiable"
    assert result.product_pass is True
    assert result.eval_validity == "incomplete"
    assert result.release_pass is False


def test_completed_search_can_be_reused_later_in_the_same_request():
    case = BenchmarkCase(case_id="search-reuse", category="docs_only", query="공식 자료를 찾아줘.")
    events = [
        *_events("succeeded", "search-original", "tavily_search"),
        {"sequence": 3, "invocation_id": "search-reuse", "tool_name": "tavily_search",
         "phase": "reused", "origin_invocation_id": "search-original"},
    ]
    assessment = assess_execution_policy(
        policy=policy_for_case(case), evidence=_evidence("request", events),
        request_id="request", prior_turns=[], tool_calls=["tavily_search"], tool_call_count=1,
    )

    assert assessment.status == "compliant"


def test_failed_action_reused_in_same_request_cannot_claim_success():
    case = BenchmarkCase(
        case_id="action-reuse", category="tool_action", query="Slack에 보내줘.",
        expected_tools=["slack_notify"],
    )
    events = [
        *_events("failed", "slack-original"),
        {"sequence": 3, "invocation_id": "slack-reuse", "tool_name": "slack_notify",
         "phase": "reused", "origin_invocation_id": "slack-original"},
    ]
    assessment = assess_execution_policy(
        policy=policy_for_case(case), evidence=_evidence("request", events),
        request_id="request", prior_turns=[], tool_calls=["slack_notify"], tool_call_count=1,
        actions=[_receipt("success", "slack-original")],
    )

    assert assessment.status == "indeterminate"
    assert "tool_execution_receipt_outcome_conflict" in assessment.failure_codes


@pytest.mark.parametrize("evidence_available, attempted, expected_status", [
    (True, False, "failed"),
    (False, False, "unverifiable"),
    (True, True, "unverifiable"),
])
def test_missing_save_receipt_distinguishes_proven_omission_from_missing_observation(
    evidence_available, attempted, expected_status,
):
    result = _result(
        missing=True, tool_name="save_text", evidence_available=evidence_available,
        missing_receipt_after_attempt=attempted,
    )

    assert result.save_assessment.status == expected_status
    assert result.action_assessments["save_text"].status == expected_status
    assert result.eval_validity == ("valid" if expected_status == "failed" else "incomplete")
    assert result.release_pass is False
    if attempted:
        assert result.product_pass is True


def test_omitted_save_is_product_failure_with_distinct_request_and_execution_ids():
    result = _result(missing=True, tool_name="save_text")

    assert result.answer_provenance.request_id != result.request_id
    assert result.policy_assessment.status == "compliant"
    assert result.save_assessment.status == "failed"
    assert result.save_assessment.failure_codes == ["save_receipt_missing"]
    assert result.eval_validity == "valid"
    assert result.product_pass is False
    assert result.release_pass is False


@pytest.mark.parametrize("observation", [
    {"execution_id": ""},
    {"evidence_execution_id": "another-execution"},
    {"evidence_status": "incomplete"},
    {"provenance_hash": "0" * 64},
    {"domain_request_id": None},
])
def test_missing_save_is_unverifiable_without_matching_complete_execution_and_body(observation):
    result = _result(missing=True, tool_name="save_text", **observation)

    assert result.save_assessment.status == "unverifiable"
    assert result.eval_validity == "incomplete"
    assert result.release_pass is False
