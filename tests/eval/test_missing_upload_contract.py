"""Missing input is successful clarification, not a required impossible search."""

import json
from pathlib import Path

import pytest

from src.core.contracts.debug import DebugPayload
from src.core.contracts.outcome import TurnResult
from src.core.llm_errors import make_problem
from src.eval.config_models import BenchmarkCase
from src.eval.io import load_cases_jsonl
from src.eval.judge_llm import LLMJudge
from src.eval.result_models import CaseResult
from tests.eval.response_fixtures import execution_evidence, plain_response
from tests.eval.test_release_eval_contract import _final_payload, _judge, _judge_payload, _run_case


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")
RELEASE = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/cases.generated.jsonl"


def missing_upload_case():
    return next(case for case in load_cases_jsonl(RELEASE) if case.case_id == "release_action_028")


def clarification_payload(message, *, tool_calls=()):
    debug = DebugPayload(
        tool_calls=list(tool_calls), tool_call_count=len(tool_calls),
        execution_evidence=execution_evidence(tool_calls),
    ).model_dump(mode="json")
    return {
        "status": "needs_input", "response": None, "message": message, "missing_slots": ["upload"],
        "request_id": "req-1", "debug": debug,
        "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []},
    }


@pytest.mark.parametrize("typed_outcome", [False, True])
def test_release_missing_upload_case_accepts_an_upload_request_without_tools(tmp_path, typed_outcome):
    case = missing_upload_case()
    message = "확인할 파일을 먼저 업로드한 뒤 다시 질문해 주세요."
    payload = clarification_payload(message) if typed_outcome else _final_payload(plain_response(message))
    result = _run_case(
        case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[payload],
    )
    assert result.runtime_errors == result.response_errors == []
    assert result.tool_calls == []
    assert result.save_assessment.passed is True
    assert result.rule_scores["tool_choice"] == 1.0
    assert result.composite_quality_score == pytest.approx(1.0)
    assert result.judge_pass is result.release_pass is True
    if typed_outcome:
        assert result.response is None
        assert result.turn_result.status == "needs_input"
        assert result.turn_result.message == message
        assert result.scenario_turns[0].turn_result == result.turn_result
        restored = CaseResult.model_validate_json(result.model_dump_json())
        assert restored.turn_result == result.turn_result
        assert restored.scenario_turns[0].turn_result == result.turn_result


@pytest.mark.parametrize("typed_outcome", [False, True])
def test_missing_upload_still_blocks_an_attempted_save_with_a_perfect_judge(tmp_path, typed_outcome):
    case = missing_upload_case()
    message = "파일이 없지만 결론을 저장했습니다."
    payload = (clarification_payload(message, tool_calls=["save_text"]) if typed_outcome else
               _final_payload(plain_response(message), tool_calls=["save_text"]))
    result = _run_case(
        case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[payload],
    )
    assert result.save_assessment.passed is False
    assert result.release_pass is False
    assert "save_unexpected_execution" in result.gate_failures


@pytest.mark.parametrize("typed_outcome", [False, True])
def test_missing_upload_blocks_forbidden_search_despite_perfect_quality(tmp_path, typed_outcome):
    case = missing_upload_case()
    message = "확인할 파일을 먼저 업로드해 주세요."
    payload = (clarification_payload(message, tool_calls=["upload_search"]) if typed_outcome else
               _final_payload(plain_response(message), tool_calls=["upload_search"]))
    result = _run_case(
        case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[payload],
    )
    assert result.rule_scores["tool_choice"] == 1.0
    assert result.composite_quality_score == pytest.approx(1.0)
    assert result.release_pass is False
    assert "forbidden_tool_execution" in result.decision.failure_codes


def test_clarification_is_judged_as_its_actual_outcome_and_low_quality_blocks_release(tmp_path):
    judge = _judge(_judge_payload(0.0))
    messages = []
    invoke = judge.client.invoke

    def capture(observed):
        messages.extend(observed)
        return invoke(observed)

    judge.client.invoke = capture
    result = _run_case(missing_upload_case(), judge, tmp_path=tmp_path,
                       turns=[clarification_payload("이미 첨부된 파일을 다시 업로드해 주세요.")])

    assert judge.client.calls == 1
    payload = json.loads(messages[-1].content)
    assert payload["response"] is None
    assert payload["turn_result"]["status"] == "needs_input"
    assert payload["turn_result"]["message"] == result.response_text
    assert payload["answer_provenance"] is None
    assert result.judge_status == "succeeded"
    assert result.judge_pass is result.release_pass is False


@pytest.mark.parametrize("status,code", [("failed", "provider_schema_invalid"), ("refused", "model_refusal")])
def test_a_technical_terminal_message_cannot_be_scored_as_successful_clarification(tmp_path, status, code):
    payload = clarification_payload("파일을 업로드해 주세요.")
    payload.update(status=status, problem=make_problem(code, "planner").model_dump(mode="json"))
    judge = _judge(_judge_payload(1.0))
    result = _run_case(missing_upload_case(), judge, tmp_path=tmp_path, turns=[payload])

    assert judge.client.calls == 0
    assert result.response_errors == []
    assert code in result.error_codes
    assert result.runtime_errors
    assert result.judge_status == "not_run"
    assert result.release_pass is False


def test_a_setup_clarification_is_retained_when_the_user_supplies_the_missing_input(tmp_path):
    case = BenchmarkCase(
        case_id="clarification-follow-up", category="tool_action", query="사용할 본문은 'hello'입니다.",
        setup_turns=["사용할 본문을 확인해 주세요."], setup_forbidden_tools=[[]],
        save_expectation={"outcome": "must_not_execute"},
    )
    judge = _judge(_judge_payload(1.0))
    messages = []
    invoke = judge.client.invoke

    def capture(observed):
        messages.extend(observed)
        return invoke(observed)

    judge.client.invoke = capture
    result = _run_case(case, judge, tmp_path=tmp_path, turns=[
        clarification_payload("어떤 본문을 사용할까요?"),
        _final_payload(plain_response("사용할 본문은 hello입니다.")),
    ])

    assert result.runtime_errors == result.response_errors == []
    assert len(result.scenario_turns) == 2
    assert result.scenario_turns[0].response is None
    assert result.scenario_turns[0].turn_result.status == "needs_input"
    assert judge.client.calls == 1
    payload = json.loads(messages[-1].content)
    prior = payload["conversation"][0]
    assert prior["query"] == case.setup_turns[0]
    assert prior["response"] is None
    assert prior["turn_result"]["message"] == "어떤 본문을 사용할까요?"
    assert result.judge_input_complete is True
    assert result.judge_pass is result.release_pass is True


def test_judge_receives_the_missing_input_and_nonexecution_contract():
    case = missing_upload_case()
    payload = LLMJudge.build_case_payload(case=case, tool_calls=[], response=None)
    assert payload["case"]["resolved_upload_fixtures"] == []
    assert payload["case"]["save_expectation"]["outcome"] == "must_not_execute"


@pytest.mark.parametrize("invalid_part", ["absent", "blank", "problem", "response", "metadata"])
def test_judge_accepts_only_a_valid_clarification_outcome(invalid_part):
    payload = LLMJudge.build_case_payload(
        case=missing_upload_case(), tool_calls=[], response=None,
        turn_result=TurnResult(status="needs_input", message="파일을 업로드해 주세요.", missing_slots=["upload"]),
    )
    assert LLMJudge.payload_completeness_issues(payload) == []

    if invalid_part == "absent":
        payload.pop("turn_result")
    elif invalid_part == "blank":
        payload["turn_result"]["message"] = "  "
    elif invalid_part == "problem":
        payload["turn_result"]["problem"] = make_problem("provider_schema_invalid", "planner").model_dump(mode="json")
    elif invalid_part == "response":
        payload["response"] = plain_response("가짜 답변")
    else:
        payload.pop("retrieval_diagnostics")
    assert LLMJudge.payload_completeness_issues(payload)


def test_generation_rejects_impossible_upload_contract_before_auth_or_artifacts(tmp_path, monkeypatch):
    from src.eval.nemo_generate import generate_candidates

    specification = missing_upload_case().model_dump(mode="json")
    specification["expected_tools"] = ["upload_search"]
    source, output, artifacts = tmp_path / "specs.jsonl", tmp_path / "candidates.jsonl", tmp_path / "artifacts"
    source.write_text(json.dumps(specification, ensure_ascii=False) + "\n", encoding="utf-8")
    # Even without credentials, the authored contract error must come first.
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    with pytest.raises(ValueError, match="release_action_028.*upload_search requires a declared attachment"):
        generate_candidates([source], output, artifacts)
    assert not output.exists()
    assert not artifacts.exists()


def test_valid_generation_preflight_preserves_authored_extension_fields(tmp_path):
    from src.eval.nemo_generate import load_specs

    specification = missing_upload_case().model_dump(mode="json")
    specification["expected_tools"] = []
    specification["author_extension"] = {"retain": ["exact", "authored", "data"]}
    source = tmp_path / "specs.jsonl"
    source.write_text(json.dumps(specification, ensure_ascii=False) + "\n", encoding="utf-8")
    assert load_specs([source]) == [specification]


@pytest.mark.parametrize("uploads,legacy,tools,valid", [
    ([], None, ["upload_search"], False),
    (["limits.py"], None, ["upload_search"], True),
    ([], "limits.py", ["upload_search"], True),
    ([], None, [], True),
    ([], None, ["save_text"], True),
    ([], None, ["slack_notify"], True),
])
def test_execution_prerequisites_distinguish_attachment_search_from_other_actions(uploads, legacy, tools, valid):
    from src.eval.scenario_contracts import validate_execution_prerequisites

    case = BenchmarkCase(case_id="prerequisite", category="tool_action", query="A request",
                         upload_fixtures=uploads, upload_fixture=legacy, expected_tools=tools)
    errors = validate_execution_prerequisites(case)
    assert (errors == []) is valid
    if not valid:
        assert errors == ["upload_search requires a declared attachment in an isolated scenario"]
