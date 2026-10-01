"""Release 판정은 평가 실행 상태·평가 유효성·답변 품질 합격을 분리해 검증한다.

거짓 통과 반례를 실제 파이프라인(HTTP SSE fake → 응답 파서 → judge 경계 fake →
사례 판정 → 집계 → 직렬화)으로 고정한다. 저장소가 소유한 채점·판정·직렬화 코드는
실제 구현을 사용하고, HTTP/LLM 호출 같은 시스템 경계만 대체한다.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import requests

from src.core.answer_schema import AnswerDocument, AnswerResponse, finalize_answer
from src.core.contracts.debug import DebugPayload
from src.core.evidence import RetrievalScore, SearchHit
from src.core.save_contract import SaveOperation
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.decisions import policy_for_case
from src.eval.io import dump_jsonl
from src.eval.judge_llm import LLMJudge
from src.eval.main import command_run
from src.eval.online_runner import run_online_benchmark
from tests.eval.runner_helpers import run_case_with_weights as _run_single_case
from src.eval.online_runner.response_parser import ParsedResponseData, parse_agent_response
from tests.eval.runner_helpers import build_result_with_weights as build_case_result
from src.eval.reporting.summary import build_summary
from src.eval.result_models import CaseResult
from tests.eval.response_fixtures import (
    canonical_llm_call,
    answer_provenance,
    artifact_http_response,
    plain_response,
    saved_receipt,
    source_evidence,
    sse_http_response,
    execution_evidence,
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
    response = deepcopy(response)
    debug = DebugPayload(
        llm_calls=[canonical_llm_call(10, 5)],
        tool_calls=list(tool_calls),
        tool_call_count=len(tool_calls),

        observed_hits=[hit.model_dump(mode="json") for hit in observed_hits],
    ).model_dump(mode="json")
    debug["answer_provenance"] = answer_provenance(response)
    debug["execution_evidence"] = execution_evidence(tool_calls)
    debug.update(debug_overrides or {})
    events = (debug.get("execution_evidence") or {}).get("events", [])
    for action in response.get("actions", []):
        event = next((item for item in events if item["tool_name"] == action["kind"]
                      and item["phase"] in {"started", "reused", "blocked"}), None)
        if event:
            action["invocation_id"] = event.get("origin_invocation_id") or event["invocation_id"]
    return {"response": response, "trace": "Request ID: req-1", "debug": debug,
            "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []}}


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




# --- B. 동일 입력 + judge 호출 실패 --------------------------------------------






# --- C. tool_action 의미 평가 실패 ---------------------------------------------


def _copy_answer_case():
    return BenchmarkCase(
        case_id="copy-action",
        category="tool_action",
        query="직전 답변을 파일로 저장해줘",
        setup_turns=["이 파일을 설명해줘"],
        setup_forbidden_tools=[["save_text", "slack_notify"]],
        expected_tools=["save_text"],
        save_expectation={"outcome": "required_success", "target": {"kind": "setup_answer", "setup_turn_index": 0}},
    )


def _save_receipt():
    return {"kind": "save_text", "status": "success", "file_path": "outputs/answer.md",
            "slack": None, "message": None, "error": None}






# --- 필수 의미 세부 점수 --------------------------------------------------------












# --- judge 입력 완결성 ----------------------------------------------------------






# --- judge 출력 계약 ------------------------------------------------------------






def test_zero_score_is_a_valid_score_not_an_error(tmp_path):
    """유효한 0점은 오류가 아니라 품질 불합격으로 저장된다."""
    result = _run_case(_docs_case(), _judge(_judge_payload(0.0)), turns=[_wrong_answer_turn()], tmp_path=tmp_path)

    assert result.judge_errors == []
    assert result.llm_judge_score == 0.0
    assert result.judge_status == "succeeded"


# --- judge 비활성화와 클라이언트 상태 불일치 -------------------------------------








# --- 집계: 분모와 평가 완결성 ----------------------------------------------------


def _stored_result(case: BenchmarkCase, **fields) -> CaseResult:
    payload = {
        "decision_contract_version": 1,
        "policy_snapshot": policy_for_case(case),
        "execution_evidence": execution_evidence(),
        "request_id": "req-1",
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








# --- 직렬화와 레거시 호환 --------------------------------------------------------










# --- run/CLI 수준 계약 ----------------------------------------------------------


def _write_cases(path: Path, cases: list[BenchmarkCase]) -> Path:
    path.write_text(
        "".join(case.model_dump_json() + "\n" for case in cases), encoding="utf-8"
    )
    return path








