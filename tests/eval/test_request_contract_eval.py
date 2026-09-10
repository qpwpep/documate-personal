from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from src.eval.request_contract_eval import (
    load_cases,
    load_policy,
    make_sample_plan,
    run_evaluation,
    score_observation,
    summarize_results,
)


def _observation(case):
    expected = case["expected"]
    source = expected["body"]["source"]
    body = {"kind": expected["body"]["kind"]}
    if isinstance(source, str):
        body["source"] = {"ref": source, "response_hash": f"{source}-hash"}
    elif isinstance(source, dict):
        start = case["query"].index(source["text"])
        body["source"] = {"turn_id": "u-current", "text": source["text"], "start": start, "end": start + len(source["text"])}
    if body["kind"].startswith("transform"):
        body["instruction"] = "요청한 내용으로 수정"
    if expected["body"].get("instruction_subject"):
        body["instruction"] = "Explain the meaning of this quoted string: " + expected["body"]["instruction_subject"]
    content = []
    formats = []
    for requirement in expected["requirements"]["required"]:
        item = {"kind": requirement["kind"], "mode": requirement["mode"], "value": requirement["value"]}
        if item["kind"] == "code":
            item["kind"] = "code_example"
        (formats if item["kind"] in {"line_count", "table", "ordered_list", "list", "headings"} else content).append(item)
    contract = {
        "request_id": "pending-request-1" if expected["state"]["target_request"] == "pending" else "evaluated-request",
        "revision": 2 if expected["state"]["target_request"] == "pending" else 1,
        "failure": None,
        "relation": expected["state"]["relation"],
        "target_request_id": "pending-request-1" if expected["state"]["target_request"] == "pending" else None,
        "actions": {key: {"intent": value} for key, value in expected["actions"].items()},
        "body": body, "answer": {"content": content, "format": formats, "preferences": expected["preferences"]},
        "slack_destination": expected["destination"],
        "missing_info": [{"slot": slot, "reason": "not_provided"} for slot in expected["blocker_slots"]],
    }
    pending = case["context"]["pending"]
    pending_before = {"contract": {"request_id": "pending-request-1", "revision": 1},
                      "response": {"content_hash": "pending-hash"}, "phase": "awaiting_destination",
                      "completed_actions": pending["completed_actions"]} if pending else None
    pending_after = {**pending_before, "contract": contract,
                     "completed_actions": [name for name in pending_before["completed_actions"] if expected["actions"][name] != "requested"]} if pending_before else None
    return {
        "canonical": contract, "raw_candidate": copy.deepcopy(contract), "errors": [],
        "disposition": expected["disposition"], "contract_accepted": True,
        "context": {"current_turn_id": "u-current", "user_turns": {"u-current": case["query"]},
                    "answer_hashes": {"previous": "previous-hash", "pending": "pending-hash"},
                    "pending_request_id": "pending-request-1" if pending else None, "pending_before": pending_before},
        "pending_after": pending_after,
    }


def test_frozen_v2_contains_all_known_cases_and_independent_extensions():
    """완전 배치에는 v1의 모든 21개와 별도 확장 21개가 각각 세 번 포함된다."""
    cases, policy = load_cases(), load_policy()
    history_path = Path(__file__).resolve().parents[2] / "data/benchmarks/request_contracts/history.v1.json"
    historical = json.loads(history_path.read_text(encoding="utf-8"))
    assert [case["id"] for case in cases if case["cohort"] == "known"] == [case["id"] for case in historical["original_cases"]]
    assert len(cases) == 42
    assert len(make_sample_plan(cases, repeats=policy["repeats"])) == 126
    assert historical["initial_run"]["passed"] == 12
    assert historical["initial_run"]["failed"] == 9
    assert historical["initial_run"]["raw_responses_preserved"] is False
    assert historical["selected_diagnostic_rerun"]["aggregate_with_initial_run"] is False


def test_scoring_reports_every_failed_dimension_after_an_invalid_contract():
    """계약 파싱 실패가 액션·본문·조건·경로 진단을 첫 assert에서 중단하지 않는다."""
    case = next(case for case in load_cases() if case["id"] == "transform_three_lines")
    observation = _observation(case)
    observation.update(canonical=None, contract_accepted=False, errors=["parse error"], disposition="clarify")

    scored = score_observation(case, observation, load_policy())

    assert set(scored["dimensions"]) == {"validity", "actions", "body", "requirements", "preferences", "disposition", "blockers", "state"}
    assert scored["dimensions"]["validity"]["passed"] is False
    assert scored["dimensions"]["actions"]["passed"] is False
    assert scored["dimensions"]["body"]["passed"] is False
    assert scored["dimensions"]["requirements"]["passed"] is False
    assert scored["overall"] is False


def test_rejected_raw_intent_escalation_still_fails_the_critical_gate():
    """서버가 거절했더라도 모델이 금지된 행동을 요청으로 승격한 사실은 감추지 않는다."""
    case = load_cases()[0]
    observation = _observation(case)
    observation["raw_candidate"]["actions"]["slack_notify"]["intent"] = "requested"

    score = score_observation(case, observation, load_policy())

    assert "unauthorized_intent_escalation" in score["critical_failures"]
    assert score["overall"] is False


def test_acknowledgement_does_not_force_a_delivery_body_representation():
    """단순 금지의 확인 응답은 실행 없는 본문 표현 차이로 실패하지 않는다."""
    case = load_cases()[0]
    observation = _observation(case)
    observation["canonical"]["body"] = {"kind": "compose", "instruction": "금지 요청 확인"}

    assert score_observation(case, observation, load_policy())["overall"] is True


def test_copy_input_requires_the_exact_current_turn_selection():
    """저장할 직접 입력이 다른 문자열이나 발화로 바뀌면 잘못된 본문으로 기록한다."""
    case = next(case for case in load_cases() if case["id"] == "literal_input_save")
    observation = _observation(case)
    observation["canonical"]["body"]["source"]["text"] = "다른 문장"

    score = score_observation(case, observation, load_policy())

    assert score["dimensions"]["body"]["passed"] is False
    assert "wrong_body" in score["critical_failures"]


@pytest.mark.parametrize("changed_field", ["revision", "actions", "answer"])
def test_pending_state_must_retain_the_whole_latest_canonical_contract(changed_field):
    """보류 상태의 revision·의사·조건이 최신 canonical과 다르면 상태 검증이 실패한다."""
    case = next(case for case in load_cases() if case["id"] == "pending_destination")
    observation = _observation(case)
    observation["pending_after"] = copy.deepcopy(observation["pending_after"])
    retained = observation["pending_after"]["contract"]
    if changed_field == "revision":
        retained["revision"] = 1
    elif changed_field == "actions":
        retained["actions"]["slack_notify"]["intent"] = "not_requested"
    else:
        retained["answer"]["format"] = [{"kind": "ordered_list", "mode": "required"}]

    score = score_observation(case, observation, load_policy())

    assert score["dimensions"]["state"]["passed"] is False
    assert "retained_pending_contract_not_latest" in score["dimensions"]["state"]["errors"]
    assert score["overall"] is False


def test_a_single_known_case_failure_prevents_qualification_even_above_total_threshold():
    """전체 정확도가 높아도 기존 사례 하나가 3/3을 못 채우면 통과시키지 않는다."""
    cases, policy = load_cases(), load_policy()
    results = []
    for sample in make_sample_plan(cases, repeats=3):
        case = next(case for case in cases if case["id"] == sample["case_id"])
        results.append({**sample, "score": score_observation(case, _observation(case), policy)})
    results[0]["score"]["overall"] = False

    summary = summarize_results(results, cases, policy, repeats=3)

    assert summary["rates"]["overall"] > 0.95
    assert summary["gates"]["known_cases_3_of_3"] is False
    assert summary["qualified"] is False


def test_run_records_every_sample_after_worker_error_and_refuses_overwrite(tmp_path: Path):
    """표본 하나의 예외 후에도 나머지를 저장하며 기존 run 디렉터리를 덮어쓰지 않는다."""
    cases = load_cases()[:2]

    def observer(case):
        if case["id"] == cases[0]["id"]:
            raise RuntimeError("provider unavailable")
        return _observation(case)

    manifest = {"run_id": "unit-run", "mode": "offline_test", "settings": {}, "hashes": {"code": "fixed"}}
    summary = run_evaluation(cases=cases, policy=load_policy(), observer=observer, manifest=manifest,
                             output_root=tmp_path, repeats=3, max_workers=2, code_hash=lambda: "fixed")
    rows = [json.loads(line) for line in (tmp_path / "unit-run/results.jsonl").read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 6
    assert summary["counts"]["completed"] == 6
    assert summary["qualified"] is False  # A selected diagnostic subset is never a full qualifying run.
    assert len([row for row in rows if row["observation"]["errors"]]) == 3
    with pytest.raises(FileExistsError):
        run_evaluation(cases=cases, policy=load_policy(), observer=observer, manifest=manifest,
                       output_root=tmp_path, repeats=3, max_workers=2, code_hash=lambda: "fixed")
