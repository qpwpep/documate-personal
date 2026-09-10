"""Planner boundary tests use explicit interpretations and real immutable source snapshots."""
from __future__ import annotations

import hashlib

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import ValidationError

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.debug import RetryState
from src.core.contracts.graph_state import PendingAction
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import (
    AcknowledgeBody, ActionContract, ActionRequest, AnswerReference, BoundAnswerReference, ComposeBody,
    ContractDestination, ContractEvidence, CopyAnswerBody, MissingInformation, RequestContract, TransformAnswerBody,
    UnresolvedBody, UserTurnSnapshot, WireCopyAnswerBody, WireCopyInputBody, WireRequestContract,
    WireTransformAnswerBody, WireTransformInputBody,
)
from src.runtime.nodes.planner import make_planner_node


class PlannerResult:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self.result


def wire(query, **updates):
    return WireRequestContract.model_validate({
        "evidence": [{"id": "current-evidence", "turn_id": "current", "quote": query,
                      "scope": "current_request", "interpretation": "instruction"}],
        **updates,
    })


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


def original_pending(*, response=None, phase="awaiting_destination", completed=()):
    clause = ContractEvidence(id="original", turn_id="old", quote="저장하고 Slack으로 보내줘",
                              scope="actions", interpretation="instruction")
    contract = RequestContract(request_id="pending-request", actions=ActionContract(
        save_text=ActionRequest(intent="requested", evidence_ids=("original",)),
        slack_notify=ActionRequest(intent="requested", evidence_ids=("original",))),
        evidence=(clause,))
    return PendingAction(contract=contract, response=response, phase=phase, completed_actions=completed,
                         body_prepared=response is not None and phase != "awaiting_body")


def test_initial_plan_binds_one_contract_without_an_extra_model_call():
    query = "설명은 저장하고 코드 예시는 넣지 마"
    candidate = wire(query, actions={"save_text": {"intent": "requested", "evidence_ids": ["current-evidence"]}},
                     answer={"content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["current-evidence"]}]})
    result, model = plan(query, candidate)
    confirmed = result["runtime"].request_contract
    assert confirmed.actions.save_text.intent == "requested"
    assert [(item.kind, item.mode) for item in confirmed.answer.content] == [("code_example", "forbidden")]
    assert confirmed.evidence[0].quote == query and confirmed.evidence[0].id != candidate.evidence[0].id
    assert confirmed.actions.save_text.evidence_ids == (confirmed.evidence[0].id,)
    assert confirmed.answer.content[0].evidence_ids == (confirmed.evidence[0].id,)
    assert confirmed.request_id and confirmed.revision == 1
    assert result["planner"].output.request_contract == confirmed.to_wire()
    assert len(model.calls) == 1


@pytest.mark.parametrize("payload", [
    {"use_retrieval": False, "tasks": []},
    {"use_retrieval": False, "tasks": [], "request_contract": {"status": "resolved"}},
    {"use_retrieval": False, "tasks": [], "request_contract": {"body": {"operation": "reuse", "source": "current"}}},
    {"use_retrieval": False, "tasks": [], "request_contract": {"actions": {"save_text": {"intent": "requested"}}}},
])
def test_missing_invalid_or_legacy_contract_never_recovers_actions_from_keywords(payload):
    result = make_planner_node(PlannerResult(payload), False)(state_for("저장하고 Slack으로 보내줘"))
    contract = result["runtime"].request_contract
    assert contract.failure is not None
    assert not contract.execution_ready("save_text", body_ready=True)
    assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)


def test_contract_evidence_must_reference_exact_user_text():
    result, _ = plan("Slack API를 설명해줘", wire("보내줘", actions={"slack_notify": {"intent": "requested", "evidence_ids": ["current-evidence"]}}))
    assert result["runtime"].request_contract.failure is not None


@pytest.mark.parametrize("source, reason", [
    ({"turn_id": "missing", "quote": "hello"}, "unknown_id"),
    ({"turn_id": "current", "quote": "missing quote"}, "quote_not_found"),
    ({"turn_id": "current", "quote": "hello"}, "ambiguous_scope"),
])
def test_input_selection_failures_preserve_the_requested_action_as_blocked_facts(source, reason):
    query = "hello hello를 저장해줘"
    candidate = wire(query, body={"kind": "copy_input", "source": source},
                     actions={"save_text": {"intent": "requested", "evidence_ids": ["current-evidence"]}})
    result, _ = plan(query, candidate)
    contract = result["runtime"].request_contract
    assert contract.failure is None
    assert contract.body.kind == "unresolved"
    assert contract.action_requested("save_text")
    assert [(item.slot, item.reason) for item in contract.missing_info] == [("input_reference", reason)]
    assert not contract.can_prepare_body()
    assert not contract.execution_ready("save_text", body_ready=True)


def test_input_copy_uses_exact_ledger_text_not_a_trimmed_visible_message():
    query = "앞부분\n보존할 원문 😀\n뒷부분"
    candidate = wire(query, body={"kind": "copy_input", "source": {"turn_id": "current", "quote": "보존할 원문 😀"}})
    result, _ = plan(query, candidate, messages=[HumanMessage(content="앞부분...생략", id="current")])
    source = result["runtime"].request_contract.body.source
    assert (source.text, source.start, source.end) == ("보존할 원문 😀", 4, 12)
    assert source.content_hash == hashlib.sha256(source.text.encode("utf-8")).hexdigest()


def test_an_explicit_occurrence_selects_one_exact_duplicate():
    query = "hello hello"
    candidate = wire(query, body={"kind": "copy_input", "source": {"turn_id": "current", "quote": "hello", "occurrence": 2}})
    result, _ = plan(query, candidate)
    assert result["runtime"].request_contract.body.source.start == 6


def test_translating_quoted_action_text_is_an_input_transformation_without_delivery():
    query = "'Slack으로 보내줘'를 영어로 번역해줘"
    candidate = wire(query, body={"kind": "transform_input", "source": {"turn_id": "current", "quote": "Slack으로 보내줘"},
                                  "instruction": "Translate the selected phrase into English", "evidence_ids": ["current-evidence"]})
    result, _ = plan(query, candidate)
    contract = result["runtime"].request_contract
    assert contract.body.kind == "transform_input"
    assert contract.body.source.text == "Slack으로 보내줘"
    assert contract.can_prepare_body()
    assert not contract.action_requested("slack_notify")


@pytest.mark.parametrize("kind", ["copy_answer", "transform_answer"])
def test_an_offered_answer_reference_binds_its_server_revision(kind):
    query = "방금 답변을 세 줄로 줄여 저장해줘"
    previous = finalize_answer(text_document("첫째\n둘째\n셋째\n넷째"), [])
    body = {"kind": kind, "source": {"ref": "previous"}}
    if kind == "transform_answer":
        body.update(instruction="세 줄로 줄여", evidence_ids=["current-evidence"])
    result, _ = plan(query, wire(query, body=body), previous_response=previous)
    assert result["runtime"].request_contract.body.source.response_hash == previous.content_hash
    assert result["runtime"].request_contract.body.kind == kind


def test_destination_missing_does_not_block_preparing_or_saving_the_same_body():
    query = "설명을 저장하고 Slack으로 보내줘"
    candidate = wire(query, body={"kind": "compose", "instruction": "설명을 작성해"},
                     actions={name: {"intent": "requested", "evidence_ids": ["current-evidence"]} for name in ("save_text", "slack_notify")})
    result, _ = plan(query, candidate)
    contract = result["runtime"].request_contract
    assert contract.can_prepare_body()
    assert contract.execution_ready("save_text", body_ready=True)
    assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=False)
    assert [item.slot for item in contract.missing_info] == ["slack_destination"]
    assert result["planner"].guided_followup is None


def test_unknown_subject_keeps_confirmed_actions_and_waits_for_input():
    query = "그 API를 설명하고 저장해줘"
    candidate = wire(query, body={"kind": "compose", "instruction": "요청한 API를 설명해"},
                     actions={"save_text": {"intent": "requested", "evidence_ids": ["current-evidence"]}},
                     missing_info=[{"slot": "subject", "reason": "not_provided", "question": "어떤 API인가요?"}])
    result, _ = plan(query, candidate)
    contract = result["runtime"].request_contract
    assert contract.body.kind == "unresolved"
    assert contract.body_request.kind == "compose"
    assert contract.action_requested("save_text") and not contract.can_prepare_body()
    assert result["planner"].guided_followup == "어떤 API인가요?"


def test_uncertain_slack_intent_does_not_block_an_independently_requested_body_and_save():
    query = "설명을 작성하고 저장해줘. Slack 전송 여부는 아직 모르겠어"
    candidate = wire(query, body={"kind": "compose", "instruction": "설명을 작성해"},
                     actions={"save_text": {"intent": "requested", "evidence_ids": ["current-evidence"]},
                              "slack_notify": {"intent": "unresolved", "evidence_ids": ["current-evidence"]}})
    result, _ = plan(query, candidate)
    contract = result["runtime"].request_contract
    assert contract.can_prepare_body() and contract.execution_ready("save_text", body_ready=True)
    assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)


def test_destination_supplement_preserves_request_identity_revision_and_ready_body():
    pending = original_pending(response=finalize_answer(text_document("하나\n둘\n셋"), []), completed=("save_text",))
    query = "C123"
    candidate = wire(query, relation="supplement", target_request_id=pending.contract.request_id, slack_destination={"channel_id": query})
    result, _ = plan(query, candidate, pending_action=pending, previous_response=finalize_answer(text_document("어느 채널인가요?"), []))
    contract = result["runtime"].request_contract
    assert (contract.request_id, contract.revision) == ("pending-request", 2)
    assert contract.relation == "supplement" and contract.body.kind == "copy_answer"
    assert contract.body.source.response_hash == pending.response.content_hash
    assert contract.actions.save_text.intent == "not_requested"
    assert contract.action_requested("slack_notify")


def test_forbidding_delivery_preserves_the_separately_supplied_destination_fact():
    """전송 금지는 의사를 바꾸며 같은 발화에서 확인한 목적지 정보를 지우지 않는다."""
    pending = original_pending(response=finalize_answer(text_document("원래 본문"), []), completed=("save_text",))
    query = "C123이야. 하지만 Slack에는 보내지 마."
    candidate = wire(query, relation="correction", target_request_id=pending.contract.request_id,
                     body={"kind": "acknowledge", "evidence_ids": ["current-evidence"]},
                     actions={"slack_notify": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}},
                     slack_destination={"channel_id": "C123"})

    result, _ = plan(query, candidate, pending_action=pending)
    contract = result["runtime"].request_contract

    assert contract.slack_destination == ContractDestination(channel_id="C123")
    assert contract.actions.slack_notify.intent == "forbidden"
    assert contract.can_acknowledge()
    assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)


def test_bodyless_pending_subject_supplement_prepares_body_before_asking_a_destination():
    pending = original_pending(phase="awaiting_input")
    query = "Slack chat.postMessage API"
    candidate = wire(query, relation="supplement", target_request_id=pending.contract.request_id,
                     body={"kind": "compose", "instruction": "Slack chat.postMessage API를 설명해", "evidence_ids": ["current-evidence"]})
    result, _ = plan(query, candidate, pending_action=pending)
    contract = result["runtime"].request_contract
    assert contract.can_prepare_body() and contract.body.kind == "compose"
    assert contract.action_requested("slack_notify")
    assert [item.slot for item in contract.missing_info] == ["slack_destination"]


def test_a_correction_without_pending_state_remains_a_correction_and_gets_a_new_identity():
    query = "아니, 저장하지 마"
    result, _ = plan(query, wire(query, relation="correction",
                                 actions={"save_text": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}}))
    contract = result["runtime"].request_contract
    assert contract.relation == "correction" and contract.revision == 1
    assert contract.actions.save_text.intent == "forbidden"


def test_an_unrelated_input_translation_does_not_inherit_the_pending_send():
    pending = original_pending(response=finalize_answer(text_document("기존 본문"), []))
    query = "새 요청이야. '안녕'을 영어로 번역해줘"
    candidate = wire(query, body={"kind": "transform_input", "source": {"turn_id": "current", "quote": "안녕"},
                                  "instruction": "영어로 번역해", "evidence_ids": ["current-evidence"]})
    result, _ = plan(query, candidate, pending_action=pending)
    contract = result["runtime"].request_contract
    assert contract.request_id != pending.contract.request_id
    assert contract.body.kind == "transform_input"
    assert not contract.action_requested("slack_notify")


@pytest.mark.parametrize("target", [None, "unknown-request"])
def test_relation_alone_does_not_revive_a_pending_action(target):
    result, _ = plan("C123", wire("C123", relation="supplement", target_request_id=target, slack_destination={"channel_id": "C123"}))
    contract = result["runtime"].request_contract
    assert not contract.can_prepare_body() and not contract.action_requested("slack_notify")
    assert contract.missing_info[0].slot == "pending_request"


def test_explicit_prohibition_during_a_supplement_overrides_the_old_requested_action():
    pending = original_pending(response=finalize_answer(text_document("기존 본문"), []))
    query = "C123으로 보내지 마"
    candidate = wire(query, relation="correction", target_request_id=pending.contract.request_id,
                     actions={"slack_notify": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}},
                     slack_destination={"channel_id": "C123"})
    result, _ = plan(query, candidate, pending_action=pending)
    assert result["runtime"].request_contract.actions.slack_notify.intent == "forbidden"
    assert not result["runtime"].request_contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)


def test_cancellation_preserves_forbidden_facts_and_the_target_identity():
    pending = original_pending()
    query = "그 전송은 취소해"
    candidate = wire(query, relation="cancel", target_request_id=pending.contract.request_id,
                     actions={"slack_notify": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}})
    result, _ = plan(query, candidate, pending_action=pending)
    contract = result["runtime"].request_contract
    assert contract.can_cancel_pending() and not contract.can_prepare_body()
    assert contract.actions.slack_notify.intent == "forbidden"
    assert contract.request_id == pending.contract.request_id


def test_historical_positive_evidence_cannot_authorize_an_independent_new_action():
    query = "새 주제를 설명해줘"
    candidate = WireRequestContract(body=ComposeBody(instruction=query),
        actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("old-action",))),
        evidence=(ContractEvidence(id="old-action", turn_id="old", quote="저장해줘", scope="actions.save_text", interpretation="instruction"),))
    result, _ = plan(query, candidate, user_turns=(UserTurnSnapshot(turn_id="old", text="저장해줘"), UserTurnSnapshot(turn_id="current", text=query)))
    assert not result["runtime"].request_contract.execution_ready("save_text", body_ready=True)


def test_retry_keeps_the_canonical_contract_and_revision_even_when_the_model_changes_actions():
    query = "설명하고 Slack으로 보내지 마"
    first, _ = plan(query, wire(query, actions={"slack_notify": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}}))
    fixed = first["runtime"].request_contract
    task = RetrievalTask(route="docs", query="reference", k=2, requirement_id="subject")
    state = state_for(query, request_contract=fixed, retry=RetryState(attempt=1, original_tasks=[task.model_dump()]))
    revised = wire(query, actions={"slack_notify": {"intent": "requested", "evidence_ids": ["current-evidence"]}})
    model = PlannerResult(PlannerOutput(use_retrieval=True, tasks=[task.model_copy(update={"query": "new reference"})], request_contract=revised))
    result = make_planner_node(model, False)(state)
    assert result["runtime"].request_contract == fixed
    assert result["planner"].output.request_contract == fixed.to_wire()
    assert result["planner"].output.tasks[0].query == "new reference"
    assert any("[Fixed Request Facts]" in str(message.content) for message in model.calls[0] if isinstance(message, SystemMessage))


def test_retry_model_failure_preserves_an_existing_prohibition():
    query = "저장하지 마"
    first, _ = plan(query, wire(query, actions={"save_text": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}}))
    fixed = first["runtime"].request_contract
    class Unavailable:
        def invoke(self, messages):
            raise TimeoutError("provider unavailable")
    result = make_planner_node(Unavailable(), False)(state_for(query, request_contract=fixed))
    assert result["runtime"].request_contract == fixed
    assert not fixed.execution_ready("save_text", body_ready=True)


def test_missing_upload_preserves_the_contract_while_requesting_the_file():
    query = "내 파일을 설명하되 저장하지 마"
    candidate = wire(query, actions={"save_text": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}})
    model = PlannerResult(PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="upload", query="내 파일", k=2)], request_contract=candidate))
    result = make_planner_node(model, False)(state_for(query))
    confirmed = result["runtime"].request_contract
    assert confirmed.actions.save_text.intent == "forbidden"
    assert confirmed.evidence[0].quote == query
    assert result["planner"].output.request_contract == confirmed.to_wire()
    assert result["planner"].guided_followup


def test_waiting_for_action_input_does_not_turn_an_unfinished_transform_into_a_copy():
    original_response = finalize_answer(text_document("하나\n둘\n셋\n넷"), [])
    clause = ContractEvidence(id="original", turn_id="old", quote="세 줄로 줄여 보내줘", scope="current_request", interpretation="instruction")
    original = RequestContract(request_id="pending-request",
        body=TransformAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=original_response.content_hash), instruction="세 줄로 줄여", evidence_ids=("original",)),
        body_request=WireTransformAnswerBody(source=AnswerReference(ref="previous"), instruction="세 줄로 줄여", evidence_ids=("original",)),
        actions=ActionContract(slack_notify=ActionRequest(intent="unresolved", evidence_ids=("original",))), evidence=(clause,),
        missing_info=(MissingInformation(slot="slack_intent", reason="unclear", question="보낼까요?"),))
    pending = PendingAction(contract=original, response=original_response, phase="awaiting_input", body_prepared=False)
    query = "네 C123으로 보내줘"
    candidate = wire(query, relation="supplement", target_request_id=original.request_id,
        actions={"slack_notify": {"intent": "requested", "evidence_ids": ["current-evidence"]}}, slack_destination={"channel_id": "C123"})
    result, _ = plan(query, candidate, pending_action=pending)
    contract = result["runtime"].request_contract
    assert contract.body.kind == "transform_answer"
    assert contract.body.source.ref == "pending"
    assert contract.body.source.response_hash == original_response.content_hash
    assert contract.can_prepare_body()
    assert not result["runtime"].pending_action.body_prepared


def test_action_correction_cannot_silently_fill_an_unknown_subject():
    query = "그 API를 설명하고 저장해줘"
    first, _ = plan(query, wire(query, body={"kind": "compose", "instruction": "그 API를 설명해"},
        actions={"save_text": {"intent": "requested", "evidence_ids": ["current-evidence"]}},
        missing_info=[{"slot": "subject", "reason": "not_provided", "question": "어떤 API인가요?"}]))
    original = first["runtime"].request_contract
    pending = PendingAction(contract=original, response=None, phase="awaiting_input", body_prepared=False)
    clause = ContractEvidence(id="followup", turn_id="next", quote="저장은 하지 마", scope="actions.save_text", interpretation="negation")
    candidate = WireRequestContract(relation="correction", target_request_id=original.request_id,
        actions=ActionContract(save_text=ActionRequest(intent="forbidden", evidence_ids=("followup",))), evidence=(clause,))
    state = state_for("저장은 하지 마", current_turn_id="next", user_turns=(UserTurnSnapshot(turn_id="next", text="저장은 하지 마"),), pending_action=pending)
    result = make_planner_node(PlannerResult(PlannerOutput(use_retrieval=False, tasks=[], request_contract=candidate)), False)(state)
    assert not result["runtime"].request_contract.can_prepare_body()
    assert result["runtime"].request_contract.body.kind == "unresolved"
    assert result["runtime"].request_contract.missing_info[0].slot == "subject"


@pytest.mark.parametrize("slot", ["answer_reference", "slack_intent"])
def test_unresolved_body_preserves_the_declared_missing_slot_without_inventing_a_subject(slot):
    query = "그 요청을 확인해줘"
    candidate = wire(query, body={"kind": "unresolved", "question": "무엇을 사용할까요?"},
        missing_info=[{"slot": slot, "reason": "unclear", "question": "무엇을 사용할까요?"}])
    result, _ = plan(query, candidate)
    assert [item.slot for item in result["runtime"].request_contract.missing_info] == [slot]


def test_new_relation_with_a_matching_target_cannot_reactivate_a_completed_old_action():
    pending = original_pending(response=finalize_answer(text_document("기존 본문"), []), completed=("save_text",))
    query = "새 주제를 설명해줘"
    candidate = WireRequestContract(relation="new", target_request_id=pending.contract.request_id,
        body=ComposeBody(instruction=query),
        actions=ActionContract(save_text=pending.contract.actions.save_text), evidence=pending.contract.evidence)
    result, _ = plan(query, candidate, pending_action=pending,
        user_turns=(UserTurnSnapshot(turn_id="old", text="저장하고 Slack으로 보내줘"), UserTurnSnapshot(turn_id="current", text=query)))
    contract = result["runtime"].request_contract
    assert contract.request_id != pending.contract.request_id
    assert not contract.execution_ready("save_text", body_ready=True)


def test_explicit_pending_copy_during_a_supplement_preserves_an_unfinished_transformation():
    original_response = finalize_answer(text_document("Original untranslated text"), [])
    original = original_pending(response=original_response, phase="awaiting_body").contract
    requested = WireTransformAnswerBody(source=AnswerReference(ref="previous"), instruction="한국어로 번역해", evidence_ids=("original",))
    original = original.model_copy(update={
        "body": TransformAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=original_response.content_hash),
                                    instruction="한국어로 번역해", evidence_ids=("original",)),
        "body_request": requested,
    })
    pending = PendingAction(contract=original, response=original_response, phase="awaiting_input", body_prepared=False)
    query = "C123"
    candidate = wire(query, relation="supplement", target_request_id=original.request_id,
                     body={"kind": "copy_answer", "source": {"ref": "pending"}}, slack_destination={"channel_id": query})
    result, _ = plan(query, candidate, pending_action=pending)
    assert result["runtime"].request_contract.body.kind == "transform_answer"
    assert result["runtime"].request_contract.body.source.ref == "pending"
    assert not result["runtime"].pending_action.body_prepared


def test_local_evidence_reusing_an_old_name_does_not_rewrite_the_confirmed_pending_fact():
    # Frozen run-01 pending_destination: same local name/quote, narrower scope.
    pending = original_pending(response=finalize_answer(text_document("원래 답변"), []))
    original = pending.contract.evidence[0]
    candidate = WireRequestContract(relation="supplement", target_request_id=pending.contract.request_id,
        body=WireCopyAnswerBody(source=AnswerReference(ref="pending")), slack_destination={"channel_id": "C123"},
        actions=ActionContract(slack_notify=ActionRequest(intent="requested", evidence_ids=("original",))),
        evidence=(original.model_copy(update={"scope": "actions.slack_notify"}),
                  ContractEvidence(id="destination", turn_id="current", quote="C123", scope="slack_destination", interpretation="reference")))
    result, _ = plan("C123", candidate, pending_action=pending,
        user_turns=(UserTurnSnapshot(turn_id="old", text=original.quote), UserTurnSnapshot(turn_id="current", text="C123")))
    contract = result["runtime"].request_contract
    assert contract.failure is None and contract.action_requested("slack_notify")
    assert original in contract.evidence
    assert next(item for item in contract.evidence if item.id == original.id).scope == "actions"


def test_same_local_name_in_a_later_revision_cannot_mutate_old_evidence():
    first_query = "Slack으로 보내줘"
    first, _ = plan(first_query, wire(first_query, actions={"slack_notify": {"intent": "requested", "evidence_ids": ["current-evidence"]}}))
    pending = PendingAction(contract=first["runtime"].request_contract, response=finalize_answer(text_document("원문"), []), body_prepared=True)
    next_query = "보내지 마"
    candidate = WireRequestContract(relation="correction", target_request_id=pending.contract.request_id,
        body=AcknowledgeBody(evidence_ids=("current-evidence",)),
        actions=ActionContract(slack_notify=ActionRequest(intent="forbidden", evidence_ids=("current-evidence",))),
        evidence=(ContractEvidence(id="current-evidence", turn_id="next", quote=next_query, scope="current_request", interpretation="negation"),))
    result, _ = plan(next_query, candidate, pending_action=pending, current_turn_id="next", user_turns=(UserTurnSnapshot(turn_id="next", text=next_query),))
    contract = result["runtime"].request_contract
    assert contract.failure is None and contract.actions.slack_notify.intent == "forbidden"
    assert contract.actions.slack_notify.evidence_ids != pending.contract.actions.slack_notify.evidence_ids
    assert pending.contract.actions.slack_notify.intent == "requested"


def test_pure_prohibition_acknowledges_without_a_subject_question_or_delivery_body():
    query = "Do not send this to Slack"
    candidate = wire(query, body={"kind": "acknowledge", "evidence_ids": ["current-evidence"]},
                     actions={"slack_notify": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}})
    result, _ = plan(query, candidate)
    contract = result["runtime"].request_contract
    assert contract.can_acknowledge()
    assert not contract.can_prepare_body()
    assert result["planner"].guided_followup is None and contract.missing_info == ()
    assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)


def test_acknowledging_action_prohibition_does_not_complete_an_unknown_body_subject():
    query = "그 API를 설명하고 저장해줘"
    first, _ = plan(query, wire(query, body={"kind": "compose", "instruction": "그 API를 설명해"},
        actions={"save_text": {"intent": "requested", "evidence_ids": ["current-evidence"]}},
        missing_info=[{"slot": "subject", "reason": "not_provided", "question": "어떤 API인가요?"}]))
    pending = PendingAction(contract=first["runtime"].request_contract, phase="awaiting_input", response=None, body_prepared=False)
    candidate = WireRequestContract(relation="correction", target_request_id=pending.contract.request_id,
        body=AcknowledgeBody(evidence_ids=("e1",)),
        actions=ActionContract(save_text=ActionRequest(intent="forbidden", evidence_ids=("e1",))),
        evidence=(ContractEvidence(id="e1", turn_id="next", quote="저장은 하지 마", scope="current_request", interpretation="negation"),))
    result, _ = plan("저장은 하지 마", candidate, pending_action=pending, current_turn_id="next",
                     user_turns=(UserTurnSnapshot(turn_id="next", text="저장은 하지 마"),))
    contract = result["runtime"].request_contract
    assert not contract.can_acknowledge() and not contract.can_prepare_body()
    assert contract.body.kind == "unresolved" and contract.missing_info[0].slot == "subject"


def test_a_rejected_destination_stays_unresolved_until_a_new_destination_is_supplied():
    pending = original_pending(response=finalize_answer(text_document("원래 답변"), []))
    pending = pending.model_copy(update={"contract": pending.contract.model_copy(update={"slack_destination": ContractDestination(channel_id="COLD")})})
    query = "COLD는 대상이 아니야. 채널은 아직 몰라."
    candidate = wire(query, relation="correction", target_request_id=pending.contract.request_id,
        missing_info=[{"slot": "slack_destination", "reason": "unclear", "question": "어느 채널로 보낼까요?"}])
    first, _ = plan(query, candidate, pending_action=pending)
    contract = first["runtime"].request_contract
    assert contract.slack_destination is None
    assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)
    resumed = PendingAction(contract=contract, response=pending.response, body_prepared=True)
    next_query = "계속 진행해줘"
    next_candidate = WireRequestContract(relation="supplement", target_request_id=contract.request_id,
        evidence=(ContractEvidence(id="e1", turn_id="next", quote=next_query, scope="current_request", interpretation="reference"),))
    second, _ = plan(next_query, next_candidate, pending_action=resumed, current_turn_id="next",
                     user_turns=(UserTurnSnapshot(turn_id="next", text=next_query),))
    confirmed = second["runtime"].request_contract
    assert confirmed.slack_destination is None
    assert [item.slot for item in confirmed.missing_info] == ["slack_destination"]
    assert not confirmed.execution_ready("slack_notify", body_ready=True, destination_ready=True)


@pytest.mark.parametrize("duplicate", [False, True])
def test_local_evidence_must_be_defined_once_in_the_same_model_response(duplicate):
    data = {"actions": {"save_text": {"intent": "not_requested", "evidence_ids": ["original"]}}, "evidence": []}
    if duplicate:
        item = {"id": "original", "turn_id": "current", "quote": "C123", "scope": "current_request", "interpretation": "reference"}
        data["evidence"] = [item, item]
    with pytest.raises(ValidationError):
        WireRequestContract.model_validate(data)


def test_pending_context_exposes_facts_without_server_evidence_labels():
    pending = original_pending(response=finalize_answer(text_document("원문"), []))
    result, model = plan("C123", wire("C123", relation="supplement", target_request_id=pending.contract.request_id,
                                      slack_destination={"channel_id": "C123"}), pending_action=pending)
    context = next(str(message.content) for message in model.calls[0] if message.name == "request_context")
    assert "confirmed_facts" in context and "body_prepared" in context
    assert '"evidence_ids"' not in context and '"evidence"' not in context
    assert '"filename": "server_generated"' in context
    assert result["runtime"].request_contract.action_requested("slack_notify")


def test_retrieval_retry_returns_no_request_contract_and_preserves_the_bound_revision():
    query = "설명하되 저장하지 마"
    first, _ = plan(query, wire(query, actions={"save_text": {"intent": "forbidden", "evidence_ids": ["current-evidence"]}}))
    fixed = first["runtime"].request_contract
    task = RetrievalTask(route="docs", query="reference", k=2)
    state = state_for(query, request_contract=fixed, retry=RetryState(attempt=1, original_tasks=[task.model_dump()]))
    model = PlannerResult(PlannerOutput(use_retrieval=True, tasks=[task], request_contract=None))
    result = make_planner_node(model, False)(state)
    assert result["runtime"].request_contract == fixed
    prompt = next(str(message.content) for message in model.calls[0] if "[Fixed Request Facts]" in str(message.content))
    assert "request_contract=null" in prompt
    assert all(item.id not in prompt for item in fixed.evidence)


@pytest.mark.parametrize("destination,expected_reason", [("C123", None), ("C999", "unknown_id")])
def test_a_new_destination_resolves_only_a_stale_not_provided_gap_after_binding(destination, expected_reason):
    pending = original_pending(response=finalize_answer(text_document("원래 답변"), []))
    pending = pending.model_copy(update={"contract": pending.contract.model_copy(update={
        "missing_info": (MissingInformation(slot="slack_destination", reason="unclear", question="어느 채널인가요?"),),
    })})
    candidate = wire("C123", relation="supplement", target_request_id=pending.contract.request_id,
                     slack_destination={"channel_id": destination},
                     missing_info=[{"slot": "slack_destination", "reason": "not_provided", "question": "채널을 알려 주세요."}])
    result, _ = plan("C123", candidate, pending_action=pending)
    contract = result["runtime"].request_contract
    if expected_reason is None:
        assert contract.slack_destination.channel_id == "C123"
        assert contract.missing_info == ()
        assert contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)
    else:
        assert contract.slack_destination is None
        assert [(item.slot, item.reason) for item in contract.missing_info] == [("slack_destination", expected_reason)]
        assert not contract.execution_ready("slack_notify", body_ready=True, destination_ready=True)


def test_a_destination_supplement_cannot_resolve_an_existing_action_intent_gap():
    pending = original_pending(response=finalize_answer(text_document("원래 답변"), []))
    pending = pending.model_copy(update={"contract": pending.contract.model_copy(update={
        "missing_info": (MissingInformation(slot="save_intent", reason="unclear", question="저장할까요?"),),
    })})
    candidate = wire("C123", relation="supplement", target_request_id=pending.contract.request_id,
                     slack_destination={"channel_id": "C123"})
    result, _ = plan("C123", candidate, pending_action=pending)
    contract = result["runtime"].request_contract
    assert contract.action_requested("save_text")
    assert [item.slot for item in contract.missing_info] == ["save_intent"]
    assert not contract.execution_ready("save_text", body_ready=True)
