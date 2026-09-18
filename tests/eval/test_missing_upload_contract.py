"""Missing input is successful clarification, not a required impossible search."""

import json
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase
from src.eval.io import load_cases_jsonl
from src.eval.judge_llm import LLMJudge
from tests.eval.response_fixtures import plain_response
from tests.eval.test_release_eval_contract import _final_payload, _judge, _judge_payload, _run_case


pytestmark = pytest.mark.usefixtures("empty_upload_manifest_http")
RELEASE = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/cases.generated.jsonl"


def missing_upload_case():
    return next(case for case in load_cases_jsonl(RELEASE) if case.case_id == "release_action_028")


def test_release_missing_upload_case_accepts_an_upload_request_without_tools(tmp_path):
    case = missing_upload_case()
    result = _run_case(
        case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[_final_payload(plain_response("확인할 파일을 먼저 업로드한 뒤 다시 질문해 주세요."))],
    )
    assert result.runtime_errors == result.response_errors == []
    assert result.tool_calls == []
    assert result.save_assessment.passed is True
    assert result.rule_scores["tool_choice"] == 1.0
    assert result.composite_quality_score == pytest.approx(1.0)
    assert result.judge_pass is result.release_pass is True


def test_missing_upload_still_blocks_an_attempted_save_with_a_perfect_judge(tmp_path):
    case = missing_upload_case()
    result = _run_case(
        case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[_final_payload(plain_response("파일이 없지만 결론을 저장했습니다."), tool_calls=["save_text"])],
    )
    assert result.save_assessment.passed is False
    assert result.release_pass is False
    assert "save_unexpected_execution" in result.gate_failures


def test_missing_upload_does_not_reward_an_unnecessary_search(tmp_path):
    case = missing_upload_case()
    result = _run_case(
        case, _judge(_judge_payload(1.0)), tmp_path=tmp_path,
        turns=[_final_payload(plain_response("확인할 파일을 먼저 업로드해 주세요."), tool_calls=["upload_search"])],
    )
    assert result.rule_scores["tool_choice"] < 1.0


def test_judge_receives_the_missing_input_and_nonexecution_contract():
    case = missing_upload_case()
    payload = LLMJudge.build_case_payload(case=case, tool_calls=[], response=None)
    assert payload["case"]["resolved_upload_fixtures"] == []
    assert payload["case"]["save_expectation"]["outcome"] == "must_not_execute"


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
