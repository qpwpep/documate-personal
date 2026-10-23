"""Answer content and requested language do not describe execution failures."""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_openai import ChatOpenAI

from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.contracts.outcome import TurnResult
from src.core.llm_errors import make_problem
from src.eval.config_models import BenchmarkCase, CaseOracle
from src.eval.judge_llm import LLMJudge
from src.eval.metric_rules import compute_rule_scores


def _answer(text):
    return finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [
            {"text": text, "basis": "interaction", "refs": []},
        ]}],
    }), [])


def _scores(query, text=None, *, runtime_errors=(), response_errors=(), turn_result=None):
    return compute_rule_scores(
        case=BenchmarkCase(case_id="requested-language", category="tool_action", query=query),
        response=_answer(text) if text is not None else None,
        turn_result=turn_result,
        called_tools=[], observed_hits=[],
        runtime_errors=list(runtime_errors), response_errors=list(response_errors),
    )


@pytest.mark.parametrize("message", [
    "agent execution failed", "request timeout", "unexpected error",
])
def test_requested_error_log_is_content_not_a_runtime_failure(message):
    text = f"2026-10-23 ERROR {message}"

    assert _scores(f"Copy this log exactly: {text}", text)["format_language"] == 1.0


@pytest.mark.parametrize("query, text", [
    ("다음 영문을 번역이나 설명 없이 그대로 복사해줘: Ready to start.", "Ready to start."),
    ("준비가 끝났다는 뜻의 영어 문장 하나로 답해줘.", "Everything is ready."),
])
def test_query_language_does_not_override_the_requested_output(query, text):
    assert _scores(query, text)["format_language"] == 1.0


@pytest.mark.parametrize("errors", [
    {"runtime_errors": ["provider unavailable"]},
    {"response_errors": ["SSE final_response missing"]},
])
def test_structured_failure_is_not_hidden_by_a_plausible_answer(errors):
    assert _scores("준비 상태를 알려줘.", "준비됐습니다.", **errors)["format_language"] == 0.0


@pytest.mark.parametrize("status, code", [("failed", "internal_error"), ("refused", "model_refusal")])
def test_failed_turn_has_no_scorable_answer(status, code):
    result = TurnResult(status=status, problem=make_problem(code, "synthesis"))

    assert _scores("준비 상태를 알려줘.", turn_result=result)["format_language"] == 0.0


def test_clarification_may_quote_the_error_log_it_needs():
    result = TurnResult(status="needs_input", message="Please provide the request timeout log.")

    assert _scores("Analyze the log.", turn_result=result)["format_language"] == 1.0


def test_empty_output_is_not_a_valid_format():
    assert _scores("준비 상태를 알려줘.")["format_language"] == 0.0


def test_judge_retains_expected_partial_setup_separately_from_final_answer():
    case = BenchmarkCase(
        case_id="partial-then-copy", category="tool_action",
        setup_turns=["Explain the missing document."],
        setup_expectations=[{"status": "partial", "problem_code": "evidence_insufficient"}],
        query="Copy the displayed partial answer unchanged.",
    )
    partial = TurnResult(
        status="partial", response=_answer("The requested document is unavailable."),
        problem=make_problem("evidence_insufficient", "synthesis"),
    )
    final = TurnResult(status="completed", response=partial.response)
    conversation = [{"query": case.setup_turns[0], "response": partial.response.model_dump(mode="json"),
                     "turn_result": partial.model_dump(mode="json")}]

    payload = LLMJudge.build_case_payload(
        case=case, tool_calls=[], response=final.response, turn_result=final, conversation=conversation,
    )

    assert payload["case"]["setup_expectations"] == [
        {"status": "partial", "problem_code": "evidence_insufficient"},
    ]
    assert payload["conversation"][0]["turn_result"]["status"] == "partial"
    assert payload["turn_result"]["status"] == "completed"
    assert LLMJudge.is_payload_complete(payload)


def test_judge_receives_requested_language_policy_and_preserves_its_semantic_verdict(monkeypatch):
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "language-judge", "object": "chat.completion", "created": 1,
            "model": "language-judge",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps({
                    "score": 0.2, "reason": "The requested English sentence was translated into Korean.",
                    "subscores": {name: 0.0 if name == "format_language" else 1.0 for name in (
                        "answer_quality", "groundedness", "citation_traceability", "tool_choice", "format_language",
                    )},
                }),
            }}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
        })

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr("src.eval.judge_llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(**(kwargs | {
            "api_key": "test-key", "base_url": "https://language.invalid/v1",
            "http_client": client, "use_responses_api": False, "max_retries": 0,
        })))
        case = BenchmarkCase(
            case_id="requested-english", category="tool_action",
            query="다음 영문을 번역 없이 그대로 복사해줘: Ready to start.",
            oracle=CaseOracle(
                expected_behaviors=["Preserve the English input verbatim."],
                forbidden_behaviors=["Translate the input into Korean."],
            ),
        )
        result = LLMJudge(model_name="language-judge").score_case(
            case=case, tool_calls=[], response=_answer("시작할 준비가 됐습니다."),
        )

    assert result.status == "succeeded"
    assert result.score == 0.2
    assert result.subscores.format_language == 0.0
    payload = json.loads(requests[0]["messages"][1]["content"])
    assert payload["case"]["query"] == case.query
    assert payload["case"]["oracle"] == case.oracle.model_dump(mode="json")
    policy = requests[0]["messages"][0]["content"]
    assert "For Korean queries, a non-Korean answer should score 0" not in policy
    assert "language explicitly requested" in policy
    assert "faithful copies" in policy
