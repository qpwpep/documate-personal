"""Planner boundary tests use explicit interpretations and real immutable source snapshots."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.graph_state import PendingAction
from src.core.planner_schema import PlannerOutput
from src.core.llm_errors import LLMCallError
from src.core.request_contracts import (
    ActionContract, ActionRequest, ComposeBody, ContractEvidence, RequestContract,
    UserTurnSnapshot, WireRequestContract,
)
from src.core.slack_contract import OmittedRecipient
from src.runtime.nodes.planner import make_planner_node


class PlannerResult:
    def __init__(self, result):
        self.result = result.model_dump(mode="json") if isinstance(result, PlannerOutput) else result
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.result


def state_for(query, **kwargs):
    kwargs.setdefault("current_turn_id", "current")
    kwargs.setdefault("user_turns", (UserTurnSnapshot(turn_id="current", text=query),))
    kwargs.setdefault("messages", [HumanMessage(content=query, id="current")])
    return build_graph_state_input(user_input=query, **kwargs)


def plan(query, contract, **kwargs):
    model = PlannerResult(PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract))
    state = state_for(query, **kwargs)
    result = make_planner_node(model, False)(state)
    return result, model


def original_pending(*, response, completed=()):
    clause = ContractEvidence(id="original", turn_id="old", quote="저장하고 Slack으로 보내줘",
                              scope="current_request", interpretation="instruction")
    contract = RequestContract(request_id="pending-request", actions=ActionContract(
        save_text=ActionRequest(intent="requested", evidence_ids=("original",)),
        slack_notify=ActionRequest(intent="requested", evidence_ids=("original",))),
        evidence=(clause,))
    return PendingAction(contract=contract, response=response, phase="awaiting_destination", completed_actions=completed,
                         body_prepared=True)


@pytest.mark.parametrize("payload", [
    {"use_retrieval": False, "tasks": []},
    {"use_retrieval": False, "tasks": [], "request_contract": {"actions": {"save_text": {"intent": "requested"}}}},
])
def test_missing_contract_or_ungrounded_action_never_recovers_actions_from_keywords(payload):
    state = state_for("저장하고 Slack으로 보내줘")
    model = PlannerResult(payload)
    with pytest.raises(LLMCallError) as caught:
        make_planner_node(model, False)(state)
    assert caught.value.problem.code == "model_output_invalid"
    assert state["runtime"].request_contract is None
    assert len(model.calls) == 2


def test_new_relation_with_a_matching_target_cannot_reactivate_a_completed_old_action():
    pending = original_pending(response=finalize_answer(text_document("기존 본문"), []), completed=("save_text",))
    query = "새 주제를 설명해줘"
    candidate = WireRequestContract(slack_recipient=OmittedRecipient(), relation="new", target_request_id=pending.contract.request_id,
        body=ComposeBody(instruction=query),
        actions=ActionContract(save_text=pending.contract.actions.save_text), evidence=pending.contract.evidence)
    result, _ = plan(query, candidate, pending_action=pending,
        user_turns=(UserTurnSnapshot(turn_id="old", text="저장하고 Slack으로 보내줘"), UserTurnSnapshot(turn_id="current", text=query)))
    contract = result["runtime"].request_contract
    assert contract.request_id != pending.contract.request_id
    assert not contract.execution_ready("save_text", body_ready=True)
