"""A settled product failure needs no paid quality verdict."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import pytest
import requests

from src.app.transport_contracts import AgentResponse
from src.core.answer_schema import ActionReceipt, AnswerDocument, finalize_answer
from src.core.contracts.debug import DebugPayload
from src.core.contracts.provenance import AnswerProvenance
from src.core.llm_errors import make_problem
from src.eval.config_models import BenchmarkConfig
from src.eval.online_runner.case_runner import run_online_benchmark
from src.eval.reporting.writer import load_run_outputs, write_run_outputs


def _run(monkeypatch, tmp_path, *, status="completed", slack=None, forbidden_save=False,
         judge_enabled=True, save_receipt_conflict=False):
    judge_calls = []

    class JudgeBoundary:
        def invoke(self, messages):
            judge_calls.append(messages)
            return SimpleNamespace(content=json.dumps({
                "score": 1.0, "reason": "The requested result is correct.",
                "subscores": {key: 1.0 for key in (
                    "answer_quality", "groundedness", "citation_traceability",
                    "tool_choice", "format_language",
                )},
            }))

    monkeypatch.setattr("src.eval.judge_llm.ChatOpenAI", lambda **kwargs: JudgeBoundary())
    actions = []
    events = []
    tools = []
    if forbidden_save:
        tools.append("save_text")
        actions.append(ActionReceipt(kind="save_text", status="error", invocation_id="save-call",
                                     error="Storage write failed."))
    if slack is not None:
        tools.append("slack_notify")
        payload = "요청한 내용입니다.".encode("utf-8")
        receipt = {
            "kind": "slack_notify", "status": slack, "invocation_id": "slack-call",
            "slack": {
                "invocation_id": "slack-call",
                "operation": {"operation_id": "send-1", "request_id": "request-1",
                              "payload_sha256": hashlib.sha256(payload).hexdigest(), "byte_count": len(payload)},
                "intent": {"state": "explicit", "selector": {"kind": "channel", "value": "C123"}},
                "selection": {"request_id": "request-1", "source": "request_input",
                              "selector": {"kind": "channel", "value": "C123"}},
                "target": {"channel_id": "C123"},
                "status": "sent" if slack == "success" else "not_sent",
            },
        }
        if slack == "success":
            receipt["slack"]["message_ts"] = "123.456"
        else:
            receipt["slack"]["failure"] = {
                "stage": "send", "code": "permission_denied", "message": "Permission denied.",
                "next_action": "fix_configuration",
            }
        actions.append(ActionReceipt.model_validate(receipt))
    for action in actions:
        terminal = "succeeded" if action.status == "success" or (save_receipt_conflict and action.kind == "save_text") else "failed"
        for phase in ("started", terminal):
            events.append({"sequence": len(events) + 1, "invocation_id": action.invocation_id,
                           "tool_name": action.kind, "phase": phase})
    answer = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{"text": "요청한 내용입니다.", "basis": "interaction"}]}],
    }), [], actions=actions)

    def http_request(_session, method, url, **kwargs):
        response = requests.Response()
        response.status_code = 200
        if method.lower() == "get":
            session_id = url.split("/sessions/", 1)[1].split("/", 1)[0]
            response._content = json.dumps({"epoch": session_id, "revision": 0, "files": []}).encode()
            response.headers["Content-Type"] = "application/json"
        else:
            has_answer = status in {"completed", "partial"}
            debug = DebugPayload(
                tool_calls=tools, tool_call_count=len(tools), llm_calls=[],
                answer_provenance=(AnswerProvenance(
                    body_kind="compose", response_hash=answer.content_hash, evidence_packet=[],
                ) if has_answer else None),
                execution_evidence={"schema_version": 1, "execution_id": "execution-1",
                                    "status": "complete", "events": events},
            )
            body = AgentResponse(
                status=status, response=answer if has_answer else None,
                problem=(make_problem("evidence_insufficient" if status == "partial" else "answer_validation_failed",
                                      "validation") if status in {"partial", "failed"} else None),
                message="어느 자료를 사용해야 하나요?" if status == "needs_input" else "",
                execution_id="execution-1", trace="Execution ID: execution-1", debug=debug,
                upload_manifest={"epoch": kwargs["json"]["session_id"], "revision": 0, "files": []},
            ).model_dump(mode="json")
            response._content = ("event: final_response\ndata: " + json.dumps(body) + "\n\n").encode()
            response.headers["Content-Type"] = "text/event-stream"
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", http_request)
    case = {"case_id": "product-preflight", "category": "tool_action", "query": "요청한 내용을 처리해줘.",
            "expected_tools": ["slack_notify"] if slack is not None else [],
            "forbidden_tools": ["save_text"] if forbidden_save else []}
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(json.dumps(case) + "\n", encoding="utf-8")
    run_dir, results, summary = run_online_benchmark(
        fixtures_path=fixture, endpoint="http://preflight.test", config=BenchmarkConfig(judge_enabled=judge_enabled),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke",
    )
    return results[0], summary, judge_calls, run_dir


@pytest.mark.parametrize("options", [{"status": "failed"}, {"status": "partial"}, {"slack": "error"}])
def test_settled_product_failure_skips_judge_and_survives_reload(monkeypatch, tmp_path, options):
    result, summary, calls, run_dir = _run(monkeypatch, tmp_path, **options)

    assert calls == []
    assert result.judge_status == "not_run"
    assert result.judge_status_reason == "product_failure"
    assert result.product_pass is False
    assert result.eval_validity == "valid"
    assert result.composite_quality_score is None
    assert result.judge_pass is None
    assert result.release_pass is False
    assert "judge_not_passed" not in result.gate_failures
    assert "incomplete_eval" not in result.gate_failures
    assert summary.metrics.judge_skipped_product_failure_cases == 1
    assert summary.metrics.judge_required_cases == 0
    assert summary.release_decision.scope == "diagnostic"
    loaded_summary, loaded = load_run_outputs(run_dir)
    assert loaded[0].judge_status_reason == "product_failure"
    assert loaded_summary.metrics.judge_skipped_product_failure_cases == 1


def test_forbidden_save_does_not_erase_verified_slack_delivery(monkeypatch, tmp_path):
    result, _, calls, _ = _run(monkeypatch, tmp_path, slack="success", forbidden_save=True)

    assert result.policy_assessment.status == "violated"
    assert result.policy_assessment.evidence_issues == []
    assert result.action_assessments["save_text"].status == "failed"
    assert result.action_assessments["slack_notify"].status == "verified"
    assert result.eval_validity == "valid"
    assert result.release_pass is False
    assert calls == []


def test_save_receipt_conflict_is_not_a_slack_observation_gap(monkeypatch, tmp_path):
    result, _, _, _ = _run(monkeypatch, tmp_path, slack="success", forbidden_save=True,
                            save_receipt_conflict=True)

    assert result.policy_assessment.evidence_issues
    assert {issue.tool_name for issue in result.policy_assessment.evidence_issues} == {"save_text"}
    assert result.action_assessments["slack_notify"].status == "verified"
    assert result.eval_validity == "valid"
    assert result.release_pass is False


def test_product_failure_skip_cannot_be_fabricated_without_execution_evidence(monkeypatch, tmp_path):
    result, summary, _, run_dir = _run(monkeypatch, tmp_path, status="partial")
    result.runtime_errors.clear()

    with pytest.raises(ValueError, match="requires an observed product failure"):
        write_run_outputs(output_dir=run_dir, results=[result], summary=summary)


def test_judge_skip_summary_must_match_retained_results(monkeypatch, tmp_path):
    result, summary, _, run_dir = _run(monkeypatch, tmp_path, status="partial")
    summary.metrics.judge_skipped_product_failure_cases = 0

    with pytest.raises(ValueError, match="judge counts"):
        write_run_outputs(output_dir=run_dir, results=[result], summary=summary)


@pytest.mark.parametrize("status", ["completed", "needs_input"])
def test_usable_answer_or_clarification_still_reaches_judge(monkeypatch, tmp_path, status):
    result, summary, calls, _ = _run(monkeypatch, tmp_path, status=status)

    assert len(calls) == 1
    assert result.judge_status == "succeeded"
    assert summary.metrics.judge_required_cases == 1
    assert summary.metrics.judge_skipped_product_failure_cases == 0


def test_disabled_smoke_judge_is_not_reported_as_product_skip(monkeypatch, tmp_path):
    result, summary, calls, _ = _run(monkeypatch, tmp_path, status="partial", judge_enabled=False)

    assert calls == []
    assert result.judge_status == "disabled"
    assert result.judge_status_reason is None
    assert result.eval_validity == "incomplete"
    assert summary.metrics.judge_disabled_cases == 1
    assert summary.metrics.judge_skipped_product_failure_cases == 0
