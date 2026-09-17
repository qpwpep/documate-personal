"""Release 판정은 평가 실행 상태·평가 유효성·답변 품질 합격을 분리해 검증한다.

거짓 통과 반례를 실제 파이프라인(HTTP SSE fake → 응답 파서 → judge 경계 fake →
사례 판정 → 집계 → 직렬화)으로 고정한다. 저장소가 소유한 채점·판정·직렬화 코드는
실제 구현을 사용하고, HTTP/LLM 호출 같은 시스템 경계만 대체한다.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from src.core.answer_schema import AnswerDocument, AnswerResponse, finalize_answer
from src.core.contracts.debug import DebugPayload, TokenUsage
from src.core.evidence import RetrievalScore, SearchHit
from src.core.save_contract import SaveOperation
from src.eval.config_models import BenchmarkCase, BenchmarkConfig, CaseWeightOverride
from src.eval.io import dump_jsonl
from src.eval.judge_llm import LLMJudge
from src.eval.main import command_run
from src.eval.online_runner import _run_single_case, run_online_benchmark
from src.eval.online_runner.response_parser import ParsedResponseData, parse_agent_response
from src.eval.online_runner.result_builder import build_case_result
from src.eval.reporting.summary import build_summary
from src.eval.result_models import CaseResult
from tests.eval.response_fixtures import (
    answer_provenance,
    artifact_http_response,
    plain_response,
    saved_receipt,
    source_evidence,
    sse_http_response,
)


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")


class _JudgeBoundary:
    """외부 judge 모델 경계 — 결정적 응답이나 오류만 반환한다."""

    def __init__(self, payload=None, error=None):
        self.payload = payload
        self.error = error
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        if self.error is not None:
            raise self.error
        body = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(content=body)


def _judge(payload=None, error=None, *, enabled=True, with_client=True) -> LLMJudge:
    judge = LLMJudge(model_name="fixture-judge", enabled=False)
    judge.enabled = enabled
    judge.client = _JudgeBoundary(payload, error) if with_client else None
    return judge


def _judge_payload(score, reason="fixture verdict", **subscores):
    merged = {
        "answer_quality": score,
        "groundedness": score,
        "citation_traceability": score,
        "tool_choice": score,
        "format_language": score,
    }
    merged.update(subscores)
    return {"score": score, "reason": reason, "subscores": merged}


def _docs_case(**overrides):
    return BenchmarkCase(
        case_id=overrides.pop("case_id", "docs-1"),
        category="docs_only",
        query=overrides.pop("query", "pandas merge는 어떻게 동작해?"),
        expected_tools=["tavily_search"],
        require_official_citation=True,
        **overrides,
    )


def _cited_answer(text="merge는 달에서 커피를 만드는 기능입니다."):
    """구조와 출처 연결은 정상이고 의미만 틀린 답변."""
    evidence = source_evidence(
        text="pandas.merge combines rows from two DataFrames by matching columns.",
        source_uri="https://pandas.pydata.org/docs/",
    )
    document = AnswerDocument.model_validate(
        {
            "blocks": [
                {
                    "type": "paragraph",
                    "content": [{"text": text, "basis": "source", "refs": [evidence.id]}],
                }
            ],
        }
    )
    return finalize_answer(document, [evidence]).model_dump(mode="json"), evidence


def _hit(evidence) -> SearchHit:
    return SearchHit(
        evidence=evidence,
        score=RetrievalScore(metric="rank", raw=1, direction="lower"),
        rank=1,
    )


def _final_payload(response, *, tool_calls=(), observed_hits=(), debug_overrides=None):
    debug = DebugPayload(
        token_usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        tool_calls=list(tool_calls),
        tool_call_count=len(tool_calls),
        models_used=["fixture-model"],
        model_usage_status="llm_used",
        observed_hits=[hit.model_dump(mode="json") for hit in observed_hits],
    ).model_dump(mode="json")
    debug["answer_provenance"] = answer_provenance(response)
    debug.update(debug_overrides or {})
    return {"response": response, "trace": "Request ID: req-1", "debug": debug}


def _run_case(case, judge, *, config=None, turns, tmp_path):
    """제품 응답 수집 → judge → 사례 판정을 실제 코드로 연결한다."""
    responses = [sse_http_response(200, payload) for payload in turns]
    with patch("src.app.client.requests.post", side_effect=responses):
        return _run_single_case(
            run_id="contract-run",
            endpoint="http://fixture",
            fixtures_path=tmp_path / "cases.jsonl",
            case=case,
            timeout_seconds=5,
            judge=judge,
            config=config or BenchmarkConfig(),
        )


def _wrong_answer_turn():
    response, evidence = _cited_answer()
    return _final_payload(
        response,
        tool_calls=["tavily_search"],
        observed_hits=[_hit(evidence)],
    )


# --- A. 명백한 오답 + 정상 완료된 judge 0점 -------------------------------------


def test_wrong_answer_with_zero_judge_score_is_a_valid_failed_evaluation(tmp_path):
    """judge 0점은 정상 평가된 불합격이며 release를 차단한다."""
    result = _run_case(
        _docs_case(),
        _judge(_judge_payload(0.0)),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "succeeded"
    assert result.eval_validity == "valid"
    assert result.llm_judge_score == 0.0
    assert result.judge_subscores is not None and result.judge_subscores.answer_quality == 0.0
    assert result.judge_pass is False
    assert result.invalid_eval is False
    assert result.release_pass is False
    # The rule scores stay high; the judge gate, not the weighted sum, blocks release.
    assert result.composite_quality_score == pytest.approx(0.8, abs=1e-6)
    assert result.product_pass is True
    assert any("judge_min_score" in failure for failure in result.judge_audit_failures)
    assert any("answer_quality" in failure for failure in result.judge_audit_failures)


# --- B. 동일 입력 + judge 호출 실패 --------------------------------------------


def test_judge_invocation_failure_marks_invalid_eval_and_blocks_release(tmp_path):
    """judge 호출 실패는 점수 상승이나 release 통과로 이어지지 않는다."""
    passing = _run_case(
        _docs_case(),
        _judge(_judge_payload(0.9)),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )
    failed = _run_case(
        _docs_case(),
        _judge(error=ConnectionError("judge service unavailable")),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert failed.judge_status == "failed"
    assert failed.eval_validity == "invalid"
    assert failed.invalid_eval is True
    assert failed.llm_judge_score is None
    assert failed.judge_pass is None
    assert failed.composite_quality_score is None
    assert failed.product_pass is None
    assert failed.release_pass is False
    # 같은 제품 응답에 judge 가용성만 바꿨으므로 규칙 점수는 동일해야 한다.
    assert failed.rule_scores == passing.rule_scores
    assert failed.rule_score_total == passing.rule_score_total
    assert "judge_failed" in failed.gate_failures


def test_judge_success_and_product_quality_are_independent_verdicts(tmp_path):
    """높은 judge 점수로 통과하는 동일 입력과 실패 입력의 규칙 점수가 같다."""
    result = _run_case(
        _docs_case(),
        _judge(_judge_payload(0.9)),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "succeeded"
    assert result.judge_pass is True
    assert result.release_pass is True
    assert result.composite_quality_score is not None and result.composite_quality_score >= 0.75


# --- C. tool_action 의미 평가 실패 ---------------------------------------------


def _copy_answer_case():
    return BenchmarkCase(
        case_id="copy-action",
        category="tool_action",
        query="직전 답변을 파일로 저장해줘",
        setup_turns=["이 파일을 설명해줘"],
        expected_tools=["save_text"],
        save_expectation={"outcome": "required_success", "target": {"kind": "setup_answer", "setup_turn_index": 0}},
    )


def _save_receipt():
    return {"kind": "save_text", "status": "success", "file_path": "outputs/answer.md",
            "target": None, "message": None, "error": None}


def test_tool_action_semantic_failure_cannot_release_on_rule_scores(tmp_path):
    """높은 규칙 점수도 judge가 확인한 의미 실패를 상쇄하지 못한다."""
    prior = _final_payload(plain_response("준비된 본문"))
    unrelated = plain_response("어제 먹은 점심 메뉴는 김치찌개였습니다.")
    unrelated["actions"] = [_save_receipt()]
    final = _final_payload(unrelated, tool_calls=["save_text"])

    result = _run_case(
        _copy_answer_case(),
        _judge(_judge_payload(0.0)),
        turns=[prior, final],
        tmp_path=tmp_path,
    )

    assert result.rule_score_total is not None and result.rule_score_total >= 0.75
    assert result.judge_status == "succeeded"
    assert result.judge_pass is False
    assert result.release_pass is False


def test_requested_answer_copy_can_release_when_judge_passes(tmp_path, monkeypatch):
    """요청된 선행 답변 복사는 judge 품질 기준 충족 시 통과할 수 있다."""
    prior_body = "정확한 설명 본문"
    prior_response = plain_response(prior_body)
    prior = _final_payload(prior_response)
    copied = plain_response(prior_body)
    monkeypatch.setattr("src.eval.online_runner.case_runner.uuid4", lambda: "save-contract-session")
    copied["actions"] = [saved_receipt(copied, tmp_path, session_id="save-contract-session", target_kind="copy_answer")]
    monkeypatch.setattr("src.eval.save_outcomes.requests.get",
                        lambda url, **kwargs: artifact_http_response(tmp_path, url.rsplit("/", 1)[-1]))
    final = _final_payload(copied, tool_calls=["save_text"], debug_overrides={
        "answer_provenance": answer_provenance(copied, body_kind="copy_answer",
                                                request_id="contract-request", contract_revision=1,
                                                save_operation_binding_sha256=SaveOperation.model_validate(copied["actions"][0]["operation"]).binding_sha256),
    })

    result = _run_case(
        _copy_answer_case(),
        _judge(_judge_payload(0.9)),
        turns=[prior, final],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "succeeded"
    assert result.judge_input_complete is True
    assert result.judge_pass is True
    assert result.release_pass is True


# --- 필수 의미 세부 점수 --------------------------------------------------------


def test_high_overall_score_cannot_offset_zero_answer_quality(tmp_path):
    """전체 점수가 높아도 필수 answer_quality 미달이면 불합격이다."""
    result = _run_case(
        _docs_case(),
        _judge(_judge_payload(0.95, answer_quality=0.0)),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "succeeded"
    assert result.judge_pass is False
    assert result.release_pass is False
    assert any("answer_quality" in failure for failure in result.judge_audit_failures)


def test_groundedness_below_minimum_fails_an_evidence_based_case(tmp_path):
    """근거 기반 사례에서 groundedness 미달은 다른 고득점으로 상쇄되지 않는다."""
    result = _run_case(
        _docs_case(),
        _judge(_judge_payload(0.95, groundedness=0.1)),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_pass is False
    assert result.release_pass is False
    assert any("groundedness" in failure for failure in result.judge_audit_failures)


@pytest.mark.parametrize("category", ["rag_only", "tool_action"])
def test_judge_minimum_score_applies_to_every_category(category, tmp_path):
    """rag_only와 tool_action에도 명시적 judge 품질 기준이 적용된다."""
    case = BenchmarkCase(
        case_id=f"{category}-1",
        category=category,
        query="설명해줘",
        expected_tools={"rag_only": ["rag_search"], "tool_action": ["save_text"]}[category],
    )
    result = _run_case(
        case,
        _judge(_judge_payload(0.0)),
        turns=[_final_payload(plain_response("무관한 본문"), tool_calls=[])],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "succeeded"
    assert result.llm_judge_score == 0.0
    assert result.judge_pass is False
    assert result.release_pass is False
    assert result.judge_min_score_applied == 0.70


def test_threshold_boundary_uses_the_declared_comparison(tmp_path):
    """임계값과 같은 점수는 통과하고 바로 아래 점수는 불합격이다."""
    at_threshold = _run_case(
        _docs_case(), _judge(_judge_payload(0.70)), turns=[_wrong_answer_turn()], tmp_path=tmp_path,
    )
    below = _run_case(
        _docs_case(), _judge(_judge_payload(0.69)), turns=[_wrong_answer_turn()], tmp_path=tmp_path,
    )

    assert at_threshold.judge_pass is True
    assert at_threshold.release_pass is True
    assert below.judge_pass is False
    assert below.release_pass is False


def test_zero_llm_judge_weight_cannot_bypass_the_required_quality_gate(tmp_path):
    """가중치 override로 llm_judge를 0으로 만들어도 의미 평가는 필수다."""
    case = _docs_case(weight_override=CaseWeightOverride(llm_judge=0.0))
    result = _run_case(
        case, _judge(_judge_payload(0.0)), turns=[_wrong_answer_turn()], tmp_path=tmp_path,
    )

    assert result.effective_weights["llm_judge"] == 0.0
    assert result.judge_pass is False
    assert result.release_pass is False


# --- judge 입력 완결성 ----------------------------------------------------------


def test_incomplete_judge_input_is_not_run_and_blocks_release(tmp_path):
    """setup conversation이 빠진 입력은 judge를 실행하지 않고 release를 차단한다."""
    case = BenchmarkCase(
        case_id="followup-1",
        category="docs_only",
        query="이어서 더 설명해줘",
        setup_turns=["먼저 설명해줘"],
        expected_tools=["tavily_search"],
    )
    response, evidence = _cited_answer("앞서 설명한 내용을 이어서 설명하면 다음과 같습니다.")
    body = _final_payload(response, tool_calls=["tavily_search"], observed_hits=[_hit(evidence)])
    parsed = parse_agent_response(body, http_status=200)
    assert parsed.response_errors == []

    boundary = _JudgeBoundary(_judge_payload(0.95))
    judge = _judge(enabled=True, with_client=False)
    judge.client = boundary
    result = build_case_result(
        run_id="incomplete-input",
        endpoint_url="http://fixture/agent/stream",
        case=case,
        judge=judge,
        config=BenchmarkConfig(),
        session_id="session-1",
        created_at="2026-01-01T00:00:00+00:00",
        request_payload={},
        latency_ms_e2e=1,
        parsed_response=parsed,
        prior_turns=None,
    )

    assert boundary.calls == 0
    assert result.judge_status == "not_run"
    assert result.judge_status_reason == "input_incomplete"
    assert result.judge_input_complete is False
    assert result.judge_input_issues
    assert result.eval_validity == "incomplete"
    assert result.release_pass is False


def test_product_failure_is_not_counted_as_a_judge_outage(tmp_path):
    """제품 응답 부재는 judge 미실행이지만 평가는 확정된 제품 실패다."""
    with patch("src.app.client.requests.post", side_effect=requests.Timeout):
        result = _run_single_case(
            run_id="timeout",
            endpoint="http://fixture",
            fixtures_path=tmp_path / "cases.jsonl",
            case=_docs_case(),
            timeout_seconds=1,
            judge=_judge(_judge_payload(0.95)),
            config=BenchmarkConfig(),
        )

    assert result.judge_status == "not_run"
    assert result.judge_status_reason == "missing_final_response"
    assert result.eval_validity == "valid"
    assert result.release_pass is False
    assert result.judge_errors == []


# --- judge 출력 계약 ------------------------------------------------------------


def test_missing_overall_score_is_invalid_not_a_subscore_average(tmp_path):
    """전체 점수 누락을 세부 평균으로 대체하지 않는다."""
    payload = {
        "reason": "forgot the score",
        "subscores": {key: 1.0 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")},
    }
    result = _run_case(_docs_case(), _judge(payload), turns=[_wrong_answer_turn()], tmp_path=tmp_path)

    assert result.judge_status == "failed"
    assert result.eval_validity == "invalid"
    assert result.llm_judge_score is None
    assert result.release_pass is False


@pytest.mark.parametrize(
    "raw",
    [
        "not json at all",
        "",
        "[1, 2, 3]",
        "5",
        json.dumps({"score": True, "subscores": {key: 0.9 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")}}),
        json.dumps({"score": "0.9", "subscores": {key: 0.9 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")}}),
        json.dumps({"score": 1.5, "subscores": {key: 0.9 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")}}),
        json.dumps({"score": -0.1, "subscores": {key: 0.9 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")}}),
        json.dumps({"score": None, "subscores": {key: 0.9 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")}}),
        json.dumps({"score": 0.9, "subscores": {"answer_quality": 0.9}}),
    ],
    ids=[
        "non_json", "empty", "json_list", "json_scalar", "bool_score", "string_score",
        "score_above_one", "negative_score", "null_score", "missing_subscore_fields",
    ],
)
def test_judge_output_contract_violations_fail_the_evaluation(raw, tmp_path):
    """비JSON·잘못된 타입·범위 밖 점수는 성공 보정 없이 평가 오류다."""
    result = _run_case(_docs_case(), _judge(raw), turns=[_wrong_answer_turn()], tmp_path=tmp_path)

    assert result.judge_status == "failed"
    assert result.eval_validity == "invalid"
    assert result.invalid_eval is True
    assert result.llm_judge_score is None
    assert result.judge_pass is None
    assert result.composite_quality_score is None
    assert result.release_pass is False


def test_zero_score_is_a_valid_score_not_an_error(tmp_path):
    """유효한 0점은 오류가 아니라 품질 불합격으로 저장된다."""
    result = _run_case(_docs_case(), _judge(_judge_payload(0.0)), turns=[_wrong_answer_turn()], tmp_path=tmp_path)

    assert result.judge_errors == []
    assert result.llm_judge_score == 0.0
    assert result.judge_status == "succeeded"


# --- judge 비활성화와 클라이언트 상태 불일치 -------------------------------------


def test_disabled_judge_marks_a_diagnostic_result_not_a_release(tmp_path):
    """명시적 no-judge 실행은 진단 결과를 만들지만 release PASS가 아니다."""
    result = _run_case(
        _docs_case(),
        _judge(enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "disabled"
    assert result.eval_validity == "incomplete"
    assert result.judge_pass is None
    assert result.llm_judge_score is None
    assert result.composite_quality_score is None
    assert result.release_pass is False
    assert "judge_disabled" in result.gate_failures


def test_enabled_config_with_unavailable_client_fails_evaluation(tmp_path):
    """config가 judge를 요구하는데 client가 없으면 평가는 실패다."""
    result = _run_case(
        _docs_case(),
        _judge(enabled=True, with_client=False),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "failed"
    assert result.judge_status_reason == "client_unavailable"
    assert result.release_pass is False


def test_enabled_config_with_disabled_client_fails_evaluation(tmp_path):
    """config enabled + judge client disabled 조합도 차단한다."""
    result = _run_case(
        _docs_case(),
        _judge(enabled=False),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )

    assert result.judge_status == "failed"
    assert result.judge_status_reason == "client_unavailable"
    assert result.release_pass is False


# --- 집계: 분모와 평가 완결성 ----------------------------------------------------


def _stored_result(case: BenchmarkCase, **fields) -> CaseResult:
    payload = {
        "run_id": "agg-run",
        "case_id": case.case_id,
        "category": case.category,
        "query": case.query,
        "session_id": f"session-{case.case_id}",
        "endpoint": "http://fixture/agent/stream",
        "request_payload": {"query": case.query},
        "http_status": 200,
        "created_at_utc": "2026-01-01T00:00:00+00:00",
        "judge_status": "succeeded",
        "eval_validity": "valid",
        "llm_judge_score": 0.9,
        "judge_subscores": {key: 0.9 for key in (
            "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language")},
        "judge_pass": True,
        "judge_gate_passed": True,
        "judge_input_complete": True,
        "product_pass": True,
        "release_pass": True,
        "composite_quality_score": 0.9,
        "rule_score_total": 0.85,
    }
    payload.update(fields)
    return CaseResult.model_validate(payload)


def _judge_failed_result(case: BenchmarkCase) -> CaseResult:
    return _stored_result(
        case,
        judge_status="failed",
        judge_status_reason="invocation_failed",
        eval_validity="invalid",
        invalid_eval=True,
        llm_judge_score=None,
        judge_subscores=None,
        judge_pass=None,
        judge_gate_passed=None,
        product_pass=None,
        release_pass=False,
        composite_quality_score=None,
        judge_errors=["judge invocation failed (fixture)"],
        gate_failures=["judge_failed"],
    )


def test_release_denominator_includes_unscored_and_failed_evaluations(tmp_path):
    """null 점수와 평가 오류는 release 통과율 분모에서 사라지지 않는다."""
    cases = [
        _docs_case(case_id="docs-ok"),
        _docs_case(case_id="docs-judge-fail"),
        _docs_case(case_id="docs-product-fail"),
    ]
    results = [
        _stored_result(cases[0]),
        _judge_failed_result(cases[1]),
        _stored_result(
            cases[2],
            judge_status="not_run", judge_status_reason="missing_final_response",
            llm_judge_score=None, judge_subscores=None, judge_pass=None,
            judge_gate_passed=None, judge_input_complete=None,
            product_pass=None, release_pass=False, composite_quality_score=None,
            runtime_errors=["request timeout"],
        ),
    ]

    summary = build_summary(
        run_id="agg-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=BenchmarkConfig(), cases=cases, results=results,
    )

    assert summary.metrics.planned_cases == 3
    assert summary.metrics.total_cases == 3
    # The product-failure case is a settled (valid) verdict; the judge-failed
    # case is not scoreable but still sits in the release denominator.
    assert summary.metrics.scored_cases == 2
    assert summary.metrics.release_passed_cases == 1
    assert summary.metrics.release_pass_rate == pytest.approx(1 / 3, abs=1e-4)
    assert summary.metrics.judge_failed_cases == 1
    assert summary.metrics.judge_not_run_cases == 1
    assert summary.metrics.judge_succeeded_cases == 1
    assert summary.overall_passed is False


def test_ninety_pass_plus_ten_judge_errors_is_not_a_release(tmp_path):
    """품질 합격 90 + judge 오류 10은 100%가 아니며 release 자격도 없다."""
    cases = [_docs_case(case_id=f"case-{index:03d}") for index in range(100)]
    results = [_stored_result(case) for case in cases[:90]]
    results += [_judge_failed_result(case) for case in cases[90:]]

    summary = build_summary(
        run_id="agg-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=BenchmarkConfig(), cases=cases, results=results,
    )

    assert summary.metrics.release_pass_rate == pytest.approx(0.9, abs=1e-4)
    release_gate = next(gate for gate in summary.gates if gate.name == "release_pass_rate")
    assert release_gate.passed is True
    completeness = next(gate for gate in summary.gates if gate.name == "evaluation_completeness")
    assert completeness.passed is False
    assert summary.overall_passed is False


def test_missing_duplicate_and_unexpected_results_block_release(tmp_path):
    """결과 집합 불일치는 run 평가 완결성 실패로 release를 차단한다."""
    cases = [_docs_case(case_id="case-a"), _docs_case(case_id="case-b")]
    results = [
        _stored_result(cases[0]),
        _stored_result(cases[0]),  # duplicate case_id
        _stored_result(BenchmarkCase(case_id="ghost", category="docs_only", query="없는 사례")),
    ]

    summary = build_summary(
        run_id="agg-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=BenchmarkConfig(), cases=cases, results=results,
    )

    completeness = next(gate for gate in summary.gates if gate.name == "evaluation_completeness")
    assert summary.metrics.missing_result_cases == 1
    assert summary.metrics.duplicate_result_cases == 1
    assert summary.metrics.unexpected_result_cases == 1
    assert completeness.passed is False
    assert summary.overall_passed is False


# --- 직렬화와 레거시 호환 --------------------------------------------------------


def test_json_roundtrip_preserves_failed_eval_state(tmp_path):
    """저장 후 재로딩해도 null/false와 상태가 보존된다."""
    result = _run_case(
        _docs_case(),
        _judge(error=ConnectionError("judge service unavailable")),
        turns=[_wrong_answer_turn()],
        tmp_path=tmp_path,
    )
    path = tmp_path / "raw_results.jsonl"
    dump_jsonl(path, [result])

    reloaded = CaseResult.model_validate_json(path.read_text(encoding="utf-8").splitlines()[0])

    assert reloaded.judge_status == "failed"
    assert reloaded.eval_validity == "invalid"
    assert reloaded.llm_judge_score is None
    assert reloaded.judge_pass is None
    assert reloaded.composite_quality_score is None
    assert reloaded.release_pass is False
    assert reloaded.passed is False
    assert reloaded.judge_gate_passed is None


def test_legacy_alias_fields_cannot_flip_an_explicit_failed_verdict(tmp_path):
    """새 계약 결과에 섞인 레거시 alias 값이 실패를 통과로 바꾸지 못한다."""
    result = _run_case(
        _docs_case(), _judge(_judge_payload(0.0)), turns=[_wrong_answer_turn()], tmp_path=tmp_path,
    )
    payload = result.model_dump()
    payload["passed"] = True
    payload["judge_gate_passed"] = True

    reloaded = CaseResult.model_validate(payload)

    assert reloaded.release_pass is False
    assert reloaded.judge_pass is False
    assert reloaded.passed is False
    assert reloaded.judge_gate_passed is False


def test_legacy_record_without_state_fields_keeps_legacy_unknown_status():
    """상태 필드가 없는 과거 결과는 레거시 경로로 읽히되 현재 정책 통과로 승격되지 않는다."""
    parsed = CaseResult.model_validate(
        {
            "run_id": "legacy", "case_id": "c1", "category": "docs_only",
            "query": "q", "session_id": "s", "endpoint": "http://x/agent",
            "request_payload": {}, "http_status": 200,
            "passed": True, "final_score": 0.9,
            "created_at_utc": "2026-01-01T00:00:00+00:00",
        }
    )

    assert parsed.judge_status == "legacy_unknown"
    assert parsed.eval_validity == "legacy_unknown"
    assert parsed.release_pass is True
    assert parsed.composite_quality_score == 0.9


def test_new_contract_rejects_inconsistent_eval_state():
    """상태·점수·판정이 모순되는 새 결과는 생성되지 않는다."""
    with pytest.raises(ValueError):
        _stored_result(_docs_case(case_id="bad"), judge_status="failed", llm_judge_score=0.9)
    with pytest.raises(ValueError):
        _stored_result(_docs_case(case_id="bad"), judge_status="succeeded", llm_judge_score=None)
    with pytest.raises(ValueError):
        _stored_result(_docs_case(case_id="bad"), eval_validity="invalid", release_pass=True)


# --- run/CLI 수준 계약 ----------------------------------------------------------


def _write_cases(path: Path, cases: list[BenchmarkCase]) -> Path:
    path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    return path


def test_release_track_requires_an_enabled_judge(tmp_path):
    """judge 비활성화 release 요청은 실행 전 설정 검증에서 거부된다."""
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_docs_case()])
    with pytest.raises(ValueError, match="judge"):
        run_online_benchmark(
            fixtures_path=cases_path,
            endpoint="http://fixture",
            config=BenchmarkConfig(judge_enabled=False),
            config_path=tmp_path / "config.toml",
            output_root=tmp_path / "out",
            track="release",
        )


def test_no_judge_smoke_produces_diagnostics_without_a_release_verdict(tmp_path):
    """no-judge smoke는 결과를 쓰되 release 통과로 표시하지 않는다."""
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_docs_case()])
    with patch("src.app.client.requests.post",
               return_value=sse_http_response(200, _wrong_answer_turn())):
        run_dir, results, summary = run_online_benchmark(
            fixtures_path=cases_path,
            endpoint="http://fixture",
            config=BenchmarkConfig(judge_enabled=False),
            config_path=tmp_path / "config.toml",
            output_root=tmp_path / "out",
            track="smoke",
        )

    assert results[0].judge_status == "disabled"
    assert summary.overall_passed is False
    assert (run_dir / "raw_results.jsonl").is_file()
    assert (run_dir / "summary.json").is_file()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "PASS" not in report.split("## Metrics")[0]


def test_command_run_returns_nonzero_on_failed_release_and_keeps_outputs(tmp_path, monkeypatch):
    """release FAIL은 결과를 저장한 뒤 non-zero로 종료한다."""
    cases_path = _write_cases(tmp_path / "cases.jsonl", [_docs_case()])
    monkeypatch.setattr(
        "src.eval.main.load_benchmark_cli_env_settings",
        lambda *args, **kwargs: SimpleNamespace(
            endpoint="http://fixture", judge_model=None, judge_enabled=True,
            live_slack_enabled=False, live_slack_channel_id=None,
            live_slack_user_id=None, live_slack_email=None,
        ),
    )
    monkeypatch.setattr(
        "src.eval.main.get_settings",
        lambda: SimpleNamespace(slack_default_user_id=None, slack_default_dm_email=None),
    )
    monkeypatch.setattr(
        "src.eval.online_runner.case_runner.LLMJudge",
        lambda model_name, enabled: _judge(_judge_payload(0.0)),
    )

    posts = []

    def post(url, *, json=None, **kwargs):
        posts.append(json)
        return sse_http_response(200, _wrong_answer_turn())

    monkeypatch.setattr(requests, "post", post)
    args = SimpleNamespace(
        mode="online", endpoint="http://fixture", config=tmp_path / "config.toml",
        track="release", limit=None, fixtures=cases_path,
        output_root=tmp_path / "out", live_slack=False,
        live_slack_channel_id=None, live_slack_user_id=None, live_slack_email=None,
    )
    (tmp_path / "config.toml").write_text("[runtime]\njudge_enabled = true\n", encoding="utf-8")

    exit_code = command_run(args)

    assert exit_code == 1
    run_dirs = list((tmp_path / "out").iterdir())
    assert run_dirs and (run_dirs[0] / "raw_results.jsonl").is_file()
    stored = [json.loads(line) for line in (run_dirs[0] / "raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    assert stored[0]["judge_status"] == "succeeded"
    assert stored[0]["release_pass"] is False
    assert stored[0]["llm_judge_score"] == 0.0


def test_command_run_rejects_release_with_disabled_judge(tmp_path, monkeypatch):
    """CLI도 judge 비활성화 release 조합을 non-zero로 거부한다."""
    (tmp_path / "config.toml").write_text("[runtime]\njudge_enabled = false\n", encoding="utf-8")
    monkeypatch.setattr(
        "src.eval.main.load_benchmark_cli_env_settings",
        lambda *args, **kwargs: SimpleNamespace(
            endpoint="http://fixture", judge_model=None, judge_enabled=False,
            live_slack_enabled=False, live_slack_channel_id=None,
            live_slack_user_id=None, live_slack_email=None,
        ),
    )
    monkeypatch.setattr(
        "src.eval.main.get_settings",
        lambda: SimpleNamespace(slack_default_user_id=None, slack_default_dm_email=None),
    )
    args = SimpleNamespace(
        mode="online", endpoint="http://fixture", config=tmp_path / "config.toml",
        track="release", limit=None, fixtures=tmp_path / "cases.jsonl",
        output_root=tmp_path / "out", live_slack=False,
        live_slack_channel_id=None, live_slack_user_id=None, live_slack_email=None,
    )

    assert command_run(args) == 2
    assert not (tmp_path / "out").exists()
