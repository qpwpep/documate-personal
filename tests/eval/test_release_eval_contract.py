"""핵심 실패가 고득점·집계 과정에서 release 통과로 바뀌지 않는지 검증한다."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.core.answer_schema import (
    AnswerDocument, SemanticValidation, finalize_answer, semantic_evidence_fingerprint,
)
from src.core.contracts.debug import DebugPayload
from src.core.contracts.provenance import AnswerProvenance
from src.core.contracts.usage import LLMCallRecord, TokenUsage
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner.response_parser import ParsedResponseData, parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.reporting.summary import build_summary
from src.eval.weighting import resolve_case_weights


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


def _case_result(index=0, *, judge_failure=None, missing_response=False, **response_options):
    case = BenchmarkCase(
        case_id=f"case-{index}", category="docs_only", query="pandas merge는 어떻게 동작해?",
        expected_tools=["tavily_search"], forbidden_tools=["slack_notify"],
        require_official_citation=True,
    )
    judge = LLMJudge(model_name="fixture-judge", enabled=False)
    judge.enabled = True
    judge.client = _JudgeBoundary(judge_failure)
    parsed = (
        ParsedResponseData(runtime_errors=["request timeout"])
        if missing_response else parse_agent_response(_response_payload(**response_options))
    )
    config = BenchmarkConfig()
    result = build_case_result(
        run_id="release-contract", endpoint_url="http://fixture", case=case, judge=judge,
        config=config, resolved_weights=resolve_case_weights(case=case, profiles=config.weights),
        session_id=f"session-{index}", created_at="2026-01-01T00:00:00+00:00",
        request_payload={"query": case.query}, latency_ms_e2e=10, parsed_response=parsed,
    )
    return case, result


def _summary(pairs, *, results=None):
    return build_summary(
        run_id="release-contract", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=BenchmarkConfig(), cases=[case for case, _ in pairs],
        results=[result for _, result in pairs] if results is None else results,
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
