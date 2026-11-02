"""Request conditions survive additions and targeted corrections across turns."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.graph_state import PendingAction
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import ContentRequirement, LayoutRequirement, LineCountRequirement, UserTurnSnapshot, WireRequestContract
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import structured_model


class PlannerBoundary:
    def __init__(self, contract):
        self.contract = contract

    def invoke(self, messages):
        context = next(message for message in messages if message.name == "request_context")
        facts = json.loads(context.content.split("\n", 1)[1])
        contract = self.contract(facts) if callable(self.contract) else self.contract
        return {"use_retrieval": False, "tasks": [], "request_contract": contract, "aspect_references": []}


def _facts(query, *, content=(), format=(), pending=None, relation="new", turn_id="current"):
    return {
        **WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
        "relation": relation,
        "target_request_id": pending.contract.request_id if pending else None,
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "compose", "instruction": "" if pending else query, "evidence_ids": []},
        "answer": {
            "content": [ContentRequirement(**{**item, "evidence_ids": ["instruction"]}).model_dump(mode="json") for item in content],
            "format": [(LineCountRequirement if item["kind"] == "line_count" else LayoutRequirement)(
                **{**item, "evidence_ids": ["instruction"]}).model_dump(mode="json") for item in format],
            "sentence_counts": [],
            "preferences": [],
        },
        "evidence": [{"id": "instruction", "turn_id": turn_id, "quote": query,
                      "scope": "current_request", "interpretation": "instruction"}],
    }


def _state(query, pending=None, *, turn_id="current", history=()):
    return build_graph_state_input(
        user_input=query, current_turn_id=turn_id,
        messages=[HumanMessage(content=query, id=turn_id)],
        user_turns=(*history, UserTurnSnapshot(turn_id=turn_id, text=query)),
        pending_action=pending,
    )


def _plan(query, *, content=(), format=(), pending=None, relation="new", turn_id="current"):
    boundary = PlannerBoundary(_facts(query, content=content, format=format, pending=pending,
                                      relation=relation, turn_id=turn_id))
    return make_planner_node(structured_model(boundary.invoke, InitialPlannerOutput), False)(
        _state(query, pending, turn_id=turn_id))


def test_independent_content_requirements_can_be_required_and_forbidden():
    query = "시간복잡도는 설명하고 비공개 학습 데이터는 추측하지 마."
    result = _plan(query, content=[
        {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다"},
        {"kind": "custom", "mode": "forbidden", "description": "비공개 학습 데이터를 추측한다"},
    ])

    assert [(item.description, item.mode) for item in result["runtime"].request_contract.answer.content] == [
        ("시간복잡도를 설명한다", "required"), ("비공개 학습 데이터를 추측한다", "forbidden"),
    ]


@pytest.mark.parametrize("kind", ["custom", "explanation"])
def test_supplement_preserves_independent_conditions_of_the_same_kind(kind):
    initial = _plan("시간복잡도와 메모리복잡도를 설명해줘.", content=[
        {"kind": kind, "mode": "required", "description": "시간복잡도를 설명한다"},
        {"kind": kind, "mode": "required", "description": "메모리복잡도를 설명한다"},
    ])
    pending = PendingAction(contract=initial["runtime"].request_contract, phase="awaiting_body")
    result = _plan("한계도 설명해줘.", pending=pending, relation="supplement", turn_id="followup", content=[
        {"kind": kind, "mode": "required", "description": "한계를 설명한다"},
    ])

    contract = result["runtime"].request_contract
    assert [item.description for item in contract.answer.content] == [
        "시간복잡도를 설명한다", "메모리복잡도를 설명한다", "한계를 설명한다",
    ]
    assert contract.request_id == pending.contract.request_id
    assert contract.revision == 2


def test_correction_replaces_only_the_selected_condition_even_when_reworded():
    initial = _plan("시간복잡도와 메모리복잡도를 설명해줘.", content=[
        {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다"},
        {"kind": "custom", "mode": "required", "description": "메모리복잡도를 설명한다"},
    ])
    pending = PendingAction(contract=initial["runtime"].request_contract, phase="awaiting_body")
    query = "시간복잡도 대신 실행 시간 측정 결과를 설명해줘."

    def correction(context):
        target = next(item for item in context["pending_action"]["confirmed_facts"]["answer"]["content"]
                      if item["description"] == "시간복잡도를 설명한다")
        return _facts(query, pending=pending, relation="correction", turn_id="followup", content=[
            {"kind": "custom", "mode": "required", "description": "실행 시간 측정 결과를 설명한다",
             "replaces": target["ref"]},
        ])

    result = make_planner_node(structured_model(PlannerBoundary(correction).invoke, InitialPlannerOutput), False)(_state(query, pending, turn_id="followup"))

    content = result["runtime"].request_contract.answer.content
    assert [item.description for item in content] == ["실행 시간 측정 결과를 설명한다", "메모리복잡도를 설명한다"]
    assert all(item.replaces is None for item in content)


@pytest.mark.parametrize("scope,interpretation", [
    ("current_request", "quotation"), ("actions.save_text", "instruction"),
])
def test_past_instruction_and_unrelated_current_evidence_cannot_correct_content(scope, interpretation):
    previous_query = "시간복잡도와 메모리복잡도를 설명해줘."
    initial = _plan(previous_query, content=[
        {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다"},
        {"kind": "custom", "mode": "required", "description": "메모리복잡도를 설명한다"},
    ])
    pending = PendingAction(contract=initial["runtime"].request_contract, phase="awaiting_body")
    query = "문구는 '실행 시간 측정 결과를 설명한다'야."

    def correction(context):
        target = context["pending_action"]["confirmed_facts"]["answer"]["content"][0]
        proposal = _facts(query, pending=pending, relation="correction", turn_id="followup", content=[
            {"kind": "custom", "mode": "required", "description": "실행 시간 측정 결과를 설명한다",
             "replaces": target["ref"]},
        ])
        proposal["answer"]["content"][0]["evidence_ids"].append("past")
        proposal["evidence"][0].update(scope=scope, interpretation=interpretation)
        proposal["evidence"].append({"id": "past", "turn_id": "current", "quote": previous_query,
                                     "scope": "current_request", "interpretation": "instruction"})
        return proposal

    state = _state(query, pending, turn_id="followup", history=[
        UserTurnSnapshot(turn_id="current", text=previous_query),
    ])
    before = state["runtime"].pending_action.model_dump()
    with pytest.raises(LLMCallError) as caught:
        make_planner_node(structured_model(PlannerBoundary(correction).invoke, InitialPlannerOutput), False)(state)
    assert caught.value.problem.code == "model_output_invalid"
    assert state["runtime"].pending_action.model_dump() == before


@pytest.mark.parametrize("incoming,expected", [
    ({"kind": "line_count", "mode": "required", "value": 5}, [("forbidden", 4), ("required", 5)]),
    ({"kind": "line_count", "mode": "forbidden", "value": 6}, [("required", 3), ("forbidden", 4), ("forbidden", 6)]),
])
def test_line_count_changes_preserve_independent_forbidden_values(incoming, expected):
    initial = _plan("세 줄로 설명하고 네 줄은 쓰지 마.", format=[
        {"kind": "line_count", "mode": "required", "value": 3},
        {"kind": "line_count", "mode": "forbidden", "value": 4},
    ])
    pending = PendingAction(contract=initial["runtime"].request_contract, phase="awaiting_body")
    result = _plan("다섯 줄로 고쳐줘." if incoming["mode"] == "required" else "여섯 줄도 쓰지 마.",
                   pending=pending, relation="correction", turn_id="followup", format=[incoming])

    assert [(item.mode, item.value) for item in result["runtime"].request_contract.answer.format] == expected


@pytest.mark.parametrize("invalid_target", ["unknown", "stale_revision", "other_request", "duplicate", "new_relation"])
def test_invalid_correction_targets_leave_pending_state_unchanged(invalid_target):
    initial = _plan("시간복잡도와 메모리복잡도를 설명해줘.", content=[
        {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다"},
        {"kind": "custom", "mode": "required", "description": "메모리복잡도를 설명한다"},
    ])
    original = initial["runtime"].request_contract
    pending = PendingAction(contract=original.model_copy(update={"revision": 2}),
                            phase="awaiting_body", completed_actions=("save_text",))
    query = "시간복잡도 대신 실행 시간 측정 결과를 설명해줘."

    def correction(context):
        ref = context["pending_action"]["confirmed_facts"]["answer"]["content"][0]["ref"]
        if invalid_target == "unknown":
            ref = "not-an-offered-condition"
        elif invalid_target == "stale_revision":
            ref = next(iter(original.content_requirements_by_ref))
        elif invalid_target == "other_request":
            foreign = original.model_copy(update={"request_id": "another-request", "revision": 2})
            ref = next(iter(foreign.content_requirements_by_ref))
        content = [{"kind": "custom", "mode": "required", "description": "실행 시간 측정 결과를 설명한다",
                    "replaces": ref}]
        if invalid_target == "duplicate":
            content.append({"kind": "custom", "mode": "required", "description": "실행 환경도 설명한다", "replaces": ref})
        return _facts(query, pending=pending, relation="new" if invalid_target == "new_relation" else "correction",
                      turn_id="followup", content=content)

    state = _state(query, pending, turn_id="followup")
    before = state["runtime"].pending_action.model_dump()
    with pytest.raises(LLMCallError) as caught:
        make_planner_node(structured_model(PlannerBoundary(correction).invoke, InitialPlannerOutput), False)(state)
    assert caught.value.problem.code == "model_output_invalid"
    assert state["runtime"].pending_action.model_dump() == before


def test_same_content_condition_cannot_be_required_and_forbidden():
    with pytest.raises(LLMCallError) as caught:
        _plan("시간복잡도를 설명하고 설명하지 마.", content=[
            {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다"},
            {"kind": "custom", "mode": "forbidden", "description": "시간복잡도를 설명한다"},
        ])
    assert caught.value.problem.code == "model_output_invalid"


def test_explicit_correction_can_reverse_one_condition_without_removing_others():
    initial = _plan("시간복잡도와 메모리복잡도를 설명해줘.", content=[
        {"kind": "custom", "mode": "required", "description": "시간복잡도를 설명한다"},
        {"kind": "custom", "mode": "required", "description": "메모리복잡도를 설명한다"},
    ])
    pending = PendingAction(contract=initial["runtime"].request_contract, phase="awaiting_body")
    query = "시간복잡도는 설명하지 마."

    def correction(context):
        ref = context["pending_action"]["confirmed_facts"]["answer"]["content"][0]["ref"]
        return _facts(query, pending=pending, relation="correction", turn_id="followup", content=[
            {"kind": "custom", "mode": "forbidden", "description": "시간복잡도를 설명한다", "replaces": ref},
        ])

    result = make_planner_node(structured_model(PlannerBoundary(correction).invoke, InitialPlannerOutput), False)(_state(query, pending, turn_id="followup"))
    assert [(item.description, item.mode) for item in result["runtime"].request_contract.answer.content] == [
        ("시간복잡도를 설명한다", "forbidden"), ("메모리복잡도를 설명한다", "required"),
    ]
