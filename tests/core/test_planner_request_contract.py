"""Planner boundary tests use explicit interpretations and real immutable source snapshots."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.pending_request import PendingRequest
from src.core.planner_schema import InitialPlannerOutput, RetrievalRewrite, RetrievalRewriteOutput, RetrievalTask
from src.core.llm_errors import LLMCallError
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, ComposeBody, ContractEvidence, RequestContract,
    SentenceCountRequirement, UserTurnSnapshot, WireRequestContract,
)
from src.core.slack_contract import OmittedRecipient
from src.runtime.nodes.planner import make_planner_node
from tests.core.helpers import structured_model


class PlannerResult:
    def __init__(self, result):
        self.result = result.model_dump(mode="json") if isinstance(result, InitialPlannerOutput) else result
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
    model = PlannerResult(InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=contract))
    state = state_for(query, **kwargs)
    result = make_planner_node(structured_model(model.invoke, InitialPlannerOutput), False)(state)
    return result, model


def original_pending(*, response, completed=()):
    clause = ContractEvidence(id="original", turn_id="old", quote="저장하고 Slack으로 보내줘",
                              scope="current_request", interpretation="instruction")
    contract = RequestContract(request_id="pending-request", actions=ActionContract(
        save_text=ActionRequest(intent="requested", evidence_ids=("original",)),
        slack_notify=ActionRequest(intent="requested", evidence_ids=("original",))),
        evidence=(clause,))
    return PendingRequest(contract=contract, response=response, phase="awaiting_destination", completed_actions=completed,
                         body_prepared=True)


def test_planner_boundary_separates_body_operations_from_delivery_instructions():
    query = "아니, 마지막 줄의 42만 43으로 고쳐서 새 TXT 파일로 저장해줘. 나머지 글자와 줄바꿈은 그대로 둬."
    previous = finalize_answer(text_document("첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42"), [])
    proposal = WireRequestContract.model_validate({
        "relation": "correction",
        "slack_recipient": {"state": "omitted"},
        "body": {"kind": "transform_answer", "change_requirement": "required", "source": {"ref": "previous"},
                 "instruction": "Change only the final line's 42 to 43. Preserve every other character and all line breaks.",
                 "evidence_ids": ["change"]},
        "actions": {"save_text": {"intent": "requested", "evidence_ids": ["change"]}},
        "evidence": [{"id": "change", "turn_id": "current", "quote": query,
                      "scope": "current_request", "interpretation": "correction"}],
    })

    updates, model = plan(query, proposal, previous_response=previous)

    definitions = InitialPlannerOutput.model_json_schema()["$defs"]
    for name in ("ComposeBody", "WireTransformInputBody", "WireTransformAnswerBody", "ExtractBody"):
        description = definitions[name]["properties"]["instruction"].get("description", "")
        assert "displayed body" in description
        assert "actions" in description
    instructions = model.calls[0][0].content
    assert "Keep body.instruction limited to the requested body operation" in instructions
    contract = updates["runtime"].request_contract
    assert contract.body.source.response_hash == previous.content_hash
    assert contract.body.instruction == proposal.body.instruction
    assert contract.actions.save_text.intent == "requested"


@pytest.mark.parametrize("payload", [
    {"use_retrieval": False, "tasks": []},
    {"use_retrieval": False, "tasks": [], "request_contract": {"actions": {"save_text": {"intent": "requested"}}}},
])
def test_missing_contract_or_ungrounded_action_never_recovers_actions_from_keywords(payload):
    state = state_for("저장하고 Slack으로 보내줘")
    model = PlannerResult(payload)
    with pytest.raises(LLMCallError) as caught:
        make_planner_node(structured_model(model.invoke, InitialPlannerOutput), False)(state)
    assert caught.value.problem.code == "model_output_invalid"
    assert state["runtime"].request_contract is None
    assert len(model.calls) == 2


def test_authority_repair_identifies_the_unsupported_requirement_and_its_evidence():
    query = "아래 텍스트만 글자와 줄바꿈을 바꾸지 말고 TXT로 저장해줘. 설명은 붙이지 마.\n\n첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42"
    # This candidate was rejected during a live exact-input save. The model
    # added a layout prohibition but cited evidence scoped only to the body.
    model = PlannerResult({
        "use_retrieval": False, "tasks": [], "aspect_references": [],
        "request_contract": {
            "relation": "new", "target_request_id": None,
            "actions": {
                "save_text": {"intent": "requested", "evidence_ids": ["e1"]},
                "slack_notify": {"intent": "not_requested", "evidence_ids": []},
            },
            "slack_recipient": {"state": "omitted"},
            "answer": {
                "content": [], "sentence_counts": [], "preferences": [],
                "format": [{"kind": "code_block", "mode": "forbidden", "value": None, "evidence_ids": ["e2"]}],
            },
            "evidence": [
                {"id": "e1", "turn_id": "current", "quote": "TXT로 저장해줘", "scope": "actions.save_text", "interpretation": "instruction"},
                {"id": "e2", "turn_id": "current", "quote": "아래 텍스트만 글자와 줄바꿈을 바꾸지 말고", "scope": "body", "interpretation": "instruction"},
                {"id": "e3", "turn_id": "current", "quote": "설명은 붙이지 마.", "scope": "answer.content", "interpretation": "negation"},
            ],
            "missing_info": [],
            "body": {"kind": "copy_input", "source": {
                "turn_id": "current", "quote": "첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42", "occurrence": None,
            }},
        },
    })
    state = state_for(query)

    with pytest.raises(LLMCallError) as caught:
        make_planner_node(structured_model(model.invoke, InitialPlannerOutput), False)(state)

    assert caught.value.problem.code == "model_output_invalid"
    assert state["runtime"].request_contract is None
    assert len(model.calls) == 2
    feedback = model.calls[1][-1].content
    assert "answer.format.code_block" in feedback
    assert "e2" in feedback
    assert "scope=body" in feedback
    assert "interpretation=instruction" in feedback
    assert "negation" in feedback
    assert "첫 줄: 봄" not in feedback


@pytest.mark.parametrize("route", ["docs", "upload"])
def test_initial_plan_preserves_bound_contract_with_or_without_available_retrieval(route):
    query = "numpy.reshape 사용법을 두 문장으로 설명하고 저장하지 마"
    proposal = WireRequestContract(
        slack_recipient=OmittedRecipient(), body=ComposeBody(instruction=query),
        actions=ActionContract(save_text=ActionRequest(intent="forbidden", evidence_ids=("save",))),
        answer=AnswerContract(sentence_counts=(SentenceCountRequirement(
            minimum=2, maximum=2, evidence_ids=("count",),
        ),)),
        evidence=(ContractEvidence(id="save", turn_id="current", quote="저장하지 마",
                                   scope="actions.save_text", interpretation="negation"),
                  ContractEvidence(id="count", turn_id="current", quote="두 문장으로",
                                   scope="answer.sentence_counts", interpretation="instruction")),
    )
    task = RetrievalTask(route=route, query="numpy.reshape usage", k=2)
    model = PlannerResult(InitialPlannerOutput(use_retrieval=True, tasks=[task], request_contract=proposal))

    result = make_planner_node(structured_model(model.invoke, InitialPlannerOutput), False)(state_for(query))

    contract = result["runtime"].request_contract
    assert contract.body.instruction == query
    assert contract.actions.save_text.intent == "forbidden"
    assert contract.evidence[0].quote == "저장하지 마"
    assert contract.actions.save_text.evidence_ids == (contract.evidence[0].id,)
    assert contract.evidence[0].id != "save"
    assert contract.answer.sentence_counts[0].minimum == contract.answer.sentence_counts[0].maximum == 2
    assert contract.answer.sentence_counts[0].evidence_ids == (contract.evidence[1].id,)
    assert contract.evidence[1].id != "count"
    if route == "upload":
        assert result["planner"].output.tasks == []
        assert result["planner"].guided_followup
        assert result["planner"].diagnostics.reason == "upload_retriever_missing"
    else:
        assert result["planner"].output.tasks == [task]
        assert result["planner"].guided_followup is None
    assert "request_contract" not in result["planner"].output.model_dump()


def test_retrieval_rewrite_keeps_the_original_bound_contract():
    query = "numpy.reshape 사용법을 한 문장으로 설명해줘"
    task = RetrievalTask(route="docs", query="numpy.reshape usage", k=2, requirement_id="reshape")
    proposal = WireRequestContract(
        slack_recipient=OmittedRecipient(), body=ComposeBody(instruction=query),
        answer=AnswerContract(sentence_counts=(SentenceCountRequirement(
            scope="numpy.reshape 사용법", minimum=1, maximum=1, evidence_ids=("count",),
        ),)),
        evidence=(ContractEvidence(id="count", turn_id="current", quote="한 문장으로",
                                   scope="answer.sentence_counts", interpretation="instruction"),),
    )
    initial = InitialPlannerOutput(use_retrieval=True, tasks=[task], request_contract=proposal)
    state = state_for(query)
    state.update(make_planner_node(structured_model(lambda _messages: initial, InitialPlannerOutput), False)(state))
    original_contract = state["runtime"].request_contract
    state["retry"] = state["retry"].model_copy(update={"attempt": 1, "failed_requirement_ids": ["reshape"]})
    rewrite = RetrievalRewriteOutput(rewrites=[RetrievalRewrite(
        requirement_id="reshape", query="numpy.reshape shape examples", k=4,
    )])

    result = make_planner_node(None, False, llm_planner_retry=structured_model(
        lambda _messages: rewrite, RetrievalRewriteOutput,
    ))(state)

    assert result["runtime"].request_contract == original_contract
    assert result["runtime"].request_contract.answer.sentence_counts == original_contract.answer.sentence_counts
    revised = result["planner"].output.tasks[0]
    assert (revised.query, revised.k) == ("numpy.reshape shape examples", 4)
    assert (revised.requirement_id, revised.route, revised.requirement) == (task.requirement_id, task.route, task.requirement)
    assert "request_contract" not in result["planner"].output.model_dump()


@pytest.mark.parametrize("turn_id, quote", [("unknown", "저장해줘"), ("current", "전송해줘")])
def test_initial_contract_rejects_evidence_absent_from_the_user_utterance(turn_id, quote):
    proposal = WireRequestContract(
        slack_recipient=OmittedRecipient(), body=ComposeBody(instruction="설명을 저장한다"),
        actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("save",))),
        evidence=(ContractEvidence(id="save", turn_id=turn_id, quote=quote,
                                   scope="actions.save_text", interpretation="instruction"),),
    )
    model = PlannerResult(InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=proposal))
    state = state_for("설명을 저장해줘")

    with pytest.raises(LLMCallError) as caught:
        make_planner_node(structured_model(model.invoke, InitialPlannerOutput), False)(state)

    assert caught.value.problem.code == "model_output_invalid"
    assert state["runtime"].request_contract is None


def test_pending_sentence_correction_preserves_other_scopes_and_rebinds_its_evidence():
    original_query = "alpha는 한 문장으로, beta도 한 문장으로 설명해줘"
    first, _ = plan(original_query, WireRequestContract(
        slack_recipient=OmittedRecipient(), body=ComposeBody(instruction=original_query),
        answer=AnswerContract(sentence_counts=tuple(
            SentenceCountRequirement(scope=target, minimum=1, maximum=1, evidence_ids=(target,))
            for target in ("alpha", "beta")
        )),
        evidence=tuple(
            ContractEvidence(id=target, turn_id="old", quote=quote,
                             scope="answer.sentence_counts", interpretation="instruction")
            for target, quote in (("alpha", "alpha는 한 문장으로"), ("beta", "beta도 한 문장으로"))
        ),
    ), current_turn_id="old", user_turns=(UserTurnSnapshot(turn_id="old", text=original_query),))
    original = first["runtime"].request_contract
    pending = PendingRequest(contract=original, phase="awaiting_body")
    query = "alpha 설명은 두 문장으로 바꾸고 전체는 세 문장으로 해줘"
    candidate = WireRequestContract(
        slack_recipient=OmittedRecipient(), relation="correction", target_request_id=original.request_id,
        answer=AnswerContract(sentence_counts=(
            SentenceCountRequirement(scope="alpha", minimum=2, maximum=2, evidence_ids=("change",)),
            SentenceCountRequirement(minimum=3, maximum=3, evidence_ids=("change",)),
        )),
        evidence=(ContractEvidence(id="change", turn_id="current", quote=query,
                                   scope="answer.sentence_counts", interpretation="correction"),),
    )

    result, _ = plan(query, candidate, pending_request=pending, user_turns=(
        UserTurnSnapshot(turn_id="old", text=original_query), UserTurnSnapshot(turn_id="current", text=query),
    ))

    contract = result["runtime"].request_contract
    assert contract.request_id == original.request_id
    assert contract.revision == original.revision + 1
    assert [(item.scope, item.minimum, item.maximum) for item in contract.answer.sentence_counts] == [
        ("alpha", 2, 2), ("beta", 1, 1), (None, 3, 3),
    ]
    assert contract.answer.sentence_counts[1] == original.answer.sentence_counts[1]
    changed_id, = contract.answer.sentence_counts[0].evidence_ids
    evidence_by_id = {item.id: item for item in contract.evidence}
    assert changed_id != "change"
    assert evidence_by_id[changed_id].turn_id == "current"
    assert evidence_by_id[changed_id].quote == query
    assert original.answer.sentence_counts[0].evidence_ids[0] not in evidence_by_id
    assert contract.answer.sentence_counts[2].evidence_ids == (changed_id,)
    assert pending.body_changed(contract)
    assert original.answer.sentence_counts[0].maximum == 1


def test_new_relation_with_a_matching_target_cannot_reactivate_a_completed_old_action():
    pending = original_pending(response=finalize_answer(text_document("기존 본문"), []), completed=("save_text",))
    query = "새 주제를 설명해줘"
    candidate = WireRequestContract(slack_recipient=OmittedRecipient(), relation="new", target_request_id=pending.contract.request_id,
        body=ComposeBody(instruction=query),
        actions=ActionContract(save_text=pending.contract.actions.save_text), evidence=pending.contract.evidence)
    result, _ = plan(query, candidate, pending_request=pending,
        user_turns=(UserTurnSnapshot(turn_id="old", text="저장하고 Slack으로 보내줘"), UserTurnSnapshot(turn_id="current", text=query)))
    contract = result["runtime"].request_contract
    assert contract.request_id != pending.contract.request_id
    assert not contract.execution_ready("save_text", body_ready=True)
