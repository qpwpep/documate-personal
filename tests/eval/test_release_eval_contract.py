"""핵심 실패가 고득점·집계 과정에서 release 통과로 바뀌지 않는지 검증한다."""

from __future__ import annotations

import json
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

import pytest
import requests

from src.core.answer_schema import (
    ActionReceipt, AnswerDocument, SemanticValidation, export_answer_text, finalize_answer,
    semantic_evidence_fingerprint,
)
from src.core.contracts.debug import DebugPayload
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.usage import LLMCallRecord, TokenUsage
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.save_contract import SaveOperation
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner.response_parser import ParsedResponseData, parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.reporting.summary import build_summary
from src.eval.reporting.writer import load_run_outputs, write_run_outputs
from src.eval.result_models import ScenarioTurnResult
from src.eval.save_outcomes import revalidate_saved_artifacts
from src.eval.summary_models import ReleaseSuiteContract
from src.eval.weighting import resolve_case_weights
from src.infra.saved_artifacts import read_saved_artifact, save_artifact


class _JudgeBoundary:
    """LLM 호출 경계만 대체하고 출력 검증·채점·판정은 실제 코드를 사용한다."""

    def __init__(self, failure=None):
        self.failure = failure

    def invoke(self, messages):
        if self.failure == "invocation_error":
            raise ConnectionError("judge service unavailable")
        payload = {
            "score": 1.0,
            "reason": "The cited answer correctly describes pandas.merge.",
            "subscores": {name: 1.0 for name in (
                "answer_quality", "groundedness", "citation_traceability",
                "tool_choice", "format_language",
            )},
        }
        if self.failure == "missing_score":
            payload.pop("score")
        return SimpleNamespace(content=json.dumps(payload))


def _config():
    return BenchmarkConfig(pricing={"models": {"fixture-model": {
        "prompt_per_1k_usd": 0.00015, "completion_per_1k_usd": 0.0006,
    }}})


def _response_payload(*, forbidden_phase=None, execution_evidence=True, semantic_checks=None):
    source_text = "pandas.merge combines rows from two DataFrames by matching columns."
    snapshot = build_snapshot(
        source_uri="https://pandas.pydata.org/docs/", title="pandas.merge",
        media_type="text/plain", source_type="official", content=source_text,
        parser="eval-fixture", parser_version="1",
    )
    source = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(
            element_id="body", kind="paragraph", text=source_text,
            anchors=[SourceAnchor(kind="web", start=0, end=len(source_text))],
        ),
    )
    answer = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{
            "type": "paragraph",
            "content": [{
                "text": "merge는 두 DataFrame의 일치하는 열을 기준으로 행을 결합합니다.",
                "basis": "source", "refs": [source.id],
            }],
        }],
    }), [source])
    if semantic_checks is not None:
        answer.semantic_validation = SemanticValidation(
            content_hash=answer.content_hash,
            evidence_fingerprint=semantic_evidence_fingerprint(answer.citations),
            model="gpt-6-luna", threshold=0.8, checks=semantic_checks,
        )
    tools = ["tavily_search", *(["slack_notify"] if forbidden_phase else [])]
    events = []
    for tool in tools:
        terminal = forbidden_phase if tool == "slack_notify" else "succeeded"
        for phase in ("started", terminal):
            events.append({
                "sequence": len(events) + 1, "invocation_id": f"call-{tool}",
                "tool_name": tool, "phase": phase,
            })
    debug = DebugPayload(
        llm_calls=[LLMCallRecord(
            stage="synthesis", attempt=1, path="structured", model_name="fixture-model",
            usage=TokenUsage(input_tokens=10, output_tokens=5),
        )],
        tool_calls=tools, tool_call_count=len(tools),
        observed_hits=[SearchHit(
            evidence=source, score=RetrievalScore(metric="rank", raw=1, direction="lower"),
            rank=1,
        ).model_dump(mode="json")],
        answer_provenance=AnswerProvenance(
            body_kind="compose", response_hash=answer.content_hash, evidence_packet=[source],
        ),
        execution_evidence={
            "schema_version": 1, "request_id": "req-1", "status": "complete", "events": events,
        } if execution_evidence else None,
    )
    return {
        "response": answer.model_dump(mode="json"), "trace": "Request ID: req-1",
        "debug": debug.model_dump(mode="json"),
    }


def _case_result(index=0, *, judge_failure=None, missing_response=False, save_required=False,
                 prior_turns=None, **response_options):
    prior_turns = prior_turns or []
    case = BenchmarkCase(
        case_id=f"case-{index}", category="docs_only", query="pandas merge는 어떻게 동작해?",
        expected_tools=["tavily_search", *(["save_text"] if save_required else [])],
        forbidden_tools=["slack_notify"],
        require_official_citation=True,
        setup_turns=[turn.query for turn in prior_turns],
        setup_forbidden_tools=[[] for _ in prior_turns],
        save_expectation=(
            {"outcome": "required_success", "target": {"kind": "final_answer"}}
            if save_required else None
        ),
    )
    judge = LLMJudge(model_name="fixture-judge", enabled=False)
    judge.enabled = True
    judge.client = _JudgeBoundary(judge_failure)
    parsed = (
        ParsedResponseData(runtime_errors=["request timeout"])
        if missing_response else parse_agent_response(_response_payload(**response_options))
    )
    config = _config()
    result = build_case_result(
        run_id="release-contract", endpoint_url="http://fixture", case=case, judge=judge,
        config=config, resolved_weights=resolve_case_weights(case=case, profiles=config.weights),
        session_id=f"session-{index}", created_at="2026-01-01T00:00:00+00:00",
        request_payload={"query": case.query}, latency_ms_e2e=10, parsed_response=parsed,
        prior_turns=prior_turns,
    )
    return case, result


def _summary(pairs, *, results=None):
    return build_summary(
        run_id="release-contract", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=_config(), cases=[case for case, _ in pairs],
        results=[result for _, result in pairs] if results is None else results,
        release_suite=ReleaseSuiteContract(
            approved_case_ids=tuple(case.case_id for case, _ in pairs),
            review_sha256="1" * 64, candidate_sha256="2" * 64,
        ),
    )


def test_complete_authorized_evaluation_can_release():
    pair = _case_result()

    assert pair[1].response_errors == []
    assert pair[1].judge_status == "succeeded"
    assert pair[1].composite_quality_score == pytest.approx(1.0)
    assert pair[1].release_pass is True
    assert _summary([pair]).overall_passed is True


@pytest.mark.parametrize("probability", [0.99, 0.2])
def test_semantically_assessed_units_are_not_counted_as_unchecked(probability):
    _, result = _case_result(semantic_checks=[
        {"name": "request_fulfilled", "probability": 0.99},
        {"name": "grounded", "unit_id": "b0.content.0", "probability": probability},
    ])

    assert result.response_errors == []
    assert result.response.checks[0].support_status == "not_evaluated"
    assert result.response.semantic_validation.passed is (probability >= 0.8)
    assert result.unchecked_unit_count == 0


def test_request_fulfillment_assessment_does_not_cover_factual_units():
    _, result = _case_result(semantic_checks=[
        {"name": "request_fulfilled", "probability": 0.99},
    ])

    assert result.response_errors == []
    assert result.unchecked_unit_count == 1


def test_refused_semantic_check_remains_unchecked():
    _, result = _case_result(semantic_checks=[
        {"name": "request_fulfilled", "probability": 0.99},
        {"name": "grounded", "unit_id": "b0.content.0", "status": "refused", "probability": None},
    ])

    assert result.response_errors == []
    assert result.response.semantic_validation.passed is False
    assert result.unchecked_unit_count == 1


@pytest.mark.parametrize("failure", ["invocation_error", "missing_score"])
def test_judge_failure_remains_in_release_denominator(failure):
    passing = _case_result()
    failed = _case_result(1, judge_failure=failure)
    result = failed[1]

    assert result.judge_status == "failed"
    assert result.eval_validity == "invalid"
    assert result.llm_judge_score is None
    assert result.judge_pass is None
    assert result.composite_quality_score is None
    assert result.release_pass is False
    assert result.rule_scores == passing[1].rule_scores
    assert "judge_failed" in result.gate_failures

    summary = _summary([passing, failed])
    assert summary.metrics.planned_cases == 2
    assert summary.metrics.total_cases == 2
    assert summary.metrics.scored_cases == 1
    assert summary.metrics.judge_failed_cases == 1
    assert summary.metrics.release_passed_cases == 1
    assert summary.metrics.release_pass_rate == pytest.approx(0.5)
    assert not next(gate for gate in summary.gates if gate.name == "evaluation_completeness").passed
    assert summary.overall_passed is False


def test_missing_product_response_is_unscored_but_remains_in_release_denominator():
    passing = _case_result()
    failed = _case_result(1, missing_response=True)
    result = failed[1]

    assert result.judge_status == "not_run"
    assert result.judge_status_reason == "missing_final_response"
    assert result.llm_judge_score is None
    assert result.composite_quality_score is None
    assert result.eval_validity == "valid"
    assert result.release_pass is False

    summary = _summary([passing, failed])
    assert summary.metrics.planned_cases == summary.metrics.total_cases == 2
    assert summary.metrics.judge_failed_cases == 0
    assert summary.metrics.judge_not_run_cases == 1
    assert summary.metrics.release_passed_cases == 1
    assert summary.metrics.release_pass_rate == pytest.approx(0.5)
    assert summary.overall_passed is False


def test_missing_save_response_remains_a_product_failure_through_reporting(tmp_path):
    passing = _case_result()
    failed = _case_result(1, missing_response=True, save_required=True)
    case, result = failed

    assert result.judge_status_reason == "missing_final_response"
    assert result.product_pass is False
    assert result.eval_validity == "valid"
    assert result.save_assessment.status == "unverifiable"
    assert result.action_assessments["save_text"].status == "unverifiable"
    assert result.release_pass is False
    assert "incomplete_eval" not in result.gate_failures

    revalidate_saved_artifacts(cases=[case], results=[result])
    summary = _summary([passing, failed])
    assert result.eval_validity == "valid"
    assert summary.metrics.product_pass_rate == pytest.approx(0.5)
    assert summary.metrics.incomplete_eval_cases == 0
    assert summary.metrics.save_unverifiable_cases == 1
    assert summary.metrics.action_unverifiable_cases == 1
    assert summary.overall_passed is False

    output = tmp_path / "run"
    write_run_outputs(output_dir=output, results=[passing[1], result], summary=summary)
    restored_summary, restored_results = load_run_outputs(output)
    restored = restored_results[1]
    assert restored.eval_validity == "valid"
    assert restored.product_pass is False
    assert restored.save_assessment.status == "unverifiable"
    assert restored.action_assessments["save_text"].status == "unverifiable"
    assert restored.release_pass is False
    assert restored_summary.metrics.product_pass_rate == pytest.approx(0.5)
    assert restored_summary.metrics.incomplete_eval_cases == 0


def _saved_setup_turn(artifact_dir):
    answer = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{"text": "준비한 내용입니다.", "basis": "interaction"}]}],
    }), [])
    content = export_answer_text(answer, include_sources=True)
    operation = SaveOperation.for_text(
        content, operation_id="setup-save", session_id="session-1", request_id="req-setup",
        contract_revision=1, target_kind="compose",
        source_hash=answer.content_hash, answer_hash=answer.content_hash,
    )
    manifest, path = save_artifact(artifact_dir, content, operation, ttl_seconds=60)
    answer.actions = [ActionReceipt(
        invocation_id="setup-save", kind="save_text", status="success", file_path=str(path),
        operation=operation, artifact=manifest.artifact, verification="verified",
    )]
    return ScenarioTurnResult(
        role="setup", query="준비한 내용을 저장해줘.", request_payload={},
        request_id="req-setup", response=answer, tool_calls=["save_text"],
        debug={"retrieval_diagnostics": []},
        answer_provenance=AnswerProvenance(
            body_kind="compose", response_hash=answer.content_hash, evidence_packet=[],
            request_id="req-setup", contract_revision=1,
            save_operation_binding_sha256=operation.binding_sha256,
        ),
        execution_evidence={
            "schema_version": 1, "request_id": "req-setup", "status": "complete",
            "events": [
                {"sequence": index, "invocation_id": "setup-save", "tool_name": "save_text", "phase": phase}
                for index, phase in enumerate(("started", "succeeded"), 1)
            ],
        },
    )


@pytest.mark.parametrize("missing_response", [False, True])
@pytest.mark.parametrize("unavailable_phase", ["case", "run_end"])
def test_unverifiable_setup_save_preserves_settled_product_failure(
    monkeypatch, tmp_path, missing_response, unavailable_phase,
):
    unavailable = unavailable_phase == "case"
    turn = _saved_setup_turn(tmp_path / "artifacts")

    def download(url, timeout):
        if unavailable:
            raise requests.ConnectionError("Artifact download is unavailable.")
        manifest, payload = read_saved_artifact(
            tmp_path / "artifacts", unquote(urlparse(url).path.rsplit("/", 1)[1]),
        )
        return SimpleNamespace(status_code=200, content=payload, headers={
            "X-Save-Binding-SHA256": manifest.operation.binding_sha256,
            "X-Artifact-Id": manifest.artifact.artifact_id,
        })

    monkeypatch.setattr(requests, "get", download)
    passing = _case_result()
    failed = _case_result(1, missing_response=missing_response, prior_turns=[turn])
    case, result = failed
    validity = "valid" if missing_response else "incomplete"
    assert result.eval_validity == (validity if unavailable else "valid")
    assert result.setup_save_assessments[0].status == ("unverifiable" if unavailable else "verified")

    unavailable = True
    revalidate_saved_artifacts(cases=[case], results=[result])
    assert result.eval_validity == validity
    assert result.setup_save_assessments[0].status == "unverifiable"
    assert result.product_pass is not missing_response
    assert result.release_pass is False
    assert ("incomplete_eval" in result.gate_failures) is not missing_response

    summary = _summary([passing, failed])
    assert summary.metrics.incomplete_eval_cases == (0 if missing_response else 1)
    assert summary.metrics.product_pass_rate == pytest.approx(0.5 if missing_response else 1.0)
    assert summary.overall_passed is False
    output = tmp_path / "run"
    write_run_outputs(output_dir=output, results=[passing[1], result], summary=summary)
    restored_summary, restored_results = load_run_outputs(output)
    assert restored_results[1].eval_validity == validity
    assert restored_results[1].product_pass is not missing_response
    assert restored_results[1].setup_save_assessments[0].status == "unverifiable"
    assert restored_results[1].release_pass is False
    assert restored_summary.metrics.product_pass_rate == summary.metrics.product_pass_rate


@pytest.mark.parametrize("defect", ["missing", "duplicate"])
def test_incomplete_result_set_blocks_release_despite_passing_scores(defect):
    pairs = [_case_result(index) for index in range(10)]
    results = [result for _, result in pairs]
    if defect == "missing":
        results.pop()
    else:
        results.append(results[0].model_copy(deep=True))

    summary = _summary(pairs, results=results)
    gates = {gate.name: gate for gate in summary.gates}

    assert summary.metrics.missing_result_cases == (1 if defect == "missing" else 0)
    assert summary.metrics.duplicate_result_cases == (1 if defect == "duplicate" else 0)
    assert gates["release_pass_rate"].passed is True
    assert gates["evaluation_completeness"].passed is False
    assert summary.overall_passed is False


@pytest.mark.parametrize("phase", ["failed", "succeeded"])
def test_nine_perfect_cases_cannot_dilute_one_forbidden_execution(phase):
    pairs = [_case_result(index) for index in range(9)]
    forbidden = _case_result(9, forbidden_phase=phase)
    pairs.append(forbidden)
    result = forbidden[1]

    assert result.composite_quality_score == pytest.approx(1.0)
    assert result.product_pass is True
    assert result.judge_pass is True
    assert result.release_pass is False
    assert "forbidden_tool_execution" in result.gate_failures

    summary = _summary(pairs)
    gates = {gate.name: gate for gate in summary.gates}
    assert gates["release_pass_rate"].passed is True
    assert gates["tool_precision"].passed is True
    assert gates["tool_execution_policy"].passed is False
    assert gates["tool_execution_policy"].actual == 1
    assert "forbidden_tool_execution" in summary.release_decision.failure_codes
    assert summary.overall_passed is False


def test_missing_execution_evidence_blocks_a_perfect_evaluation():
    pair = _case_result(execution_evidence=False)

    assert pair[1].judge_pass is True
    assert pair[1].release_pass is False
    assert "tool_execution_evidence_missing" in pair[1].gate_failures
    assert _summary([pair]).overall_passed is False
