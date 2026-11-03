"""Exact copies verify fidelity and current contract obligations before delivery."""
from hashlib import sha256
import json
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, BoundAnswerReference, BoundInputText,
    ContentRequirement, ContractEvidence, CopyAnswerBody, CopyInputBody, LineCountRequirement,
    RequestContract, SentenceCountRequirement,
)
from src.core.llm_errors import LLMCallError
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import mock_chat_model
from tests.core.test_decisions_validation import DecisionsBoundary
from tests.core.test_request_slots_delivery import DocumentModel
from tests.synthesis_fixtures import prepare_synthesis_state


ORIGINAL = "LIVE_PARTIAL_RETRY_20261023_a7c9: The worker started and a retry was scheduled."
REQUEST = ("Copy only the following input exactly, without changes or additional explanation, "
           "and save it as a TXT file. Input:\n" + ORIGINAL)


def _state(*, answer=None, request=REQUEST):
    contract = RequestContract(
        body=CopyInputBody(source=BoundInputText(
            turn_id="current", text=ORIGINAL, start=request.index(ORIGINAL), end=len(request),
            content_hash=sha256(ORIGINAL.encode("utf-8")).hexdigest(),
        )),
        actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("request",))),
        evidence=(ContractEvidence(id="request", turn_id="current", quote=request,
                                   scope="current_request", interpretation="instruction"),),
        answer=answer or AnswerContract(),
    )
    return build_graph_state_input(
        user_input=request, messages=[HumanMessage(content=request, id="current")],
        request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )


def test_exact_input_is_saved_despite_low_factual_confidence_in_quoted_claim(tmp_path, monkeypatch):
    """Copying user text does not assert that the described worker really ran."""
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    state = _state()
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    assert export_answer_text(state["response"].result) == ORIGINAL
    decisions = DecisionsBoundary({"grounded_0": 0.74, "request_fulfilled": 1.0})

    state.update(make_post_synthesis_validation_node(
        False, decisions_client=decisions, pass_threshold=0.8,
    )(state))

    assert state["response"].kind == "answer"
    # Fidelity is a deterministic check, not a fabricated model probability.
    assert state["response"].result.semantic_validation is None
    state.update(make_action_postprocess_node(build_save_text_tool(), None, False)(state))
    answer = state["response"].result
    assert export_answer_text(answer) == ORIGINAL
    receipt = answer.actions[0]
    assert receipt.kind == "save_text" and receipt.status == "success"
    assert Path(receipt.file_path).read_bytes().decode("utf-8-sig") == ORIGINAL


def test_bound_input_is_ready_without_a_model_rejudging_the_quoted_events():
    state = _state()
    decisions = DecisionsBoundary({"request_ready": 0.01, "compact_request_ready": 0.01})

    prepare_synthesis_state(state, decisions=decisions)

    assert not state["retry"].needs_retry
    assert state["retry"].retry_reason is None
    assert state["retrieval"].prepared_evidence.normal.failed_requirement_ids == []


@pytest.mark.parametrize("changed", [ORIGINAL + " Extra claim.", ORIGINAL.rstrip("."), ORIGINAL.replace(" ", "  ")])
def test_changed_copy_is_rejected_even_when_decisions_approves_it(changed):
    state = _state()
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    # Represents a producer returning a self-consistent but wrong document.
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(text_document(changed), []),
    })

    with pytest.raises(LLMCallError) as caught:
        make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert caught.value.problem.code == "model_output_invalid"


@pytest.mark.parametrize("probability, expected_kind", [(0.1, "failure"), (0.99, "answer")])
def test_copy_with_an_additional_semantic_condition_still_checks_that_condition(probability, expected_kind):
    state = _state(answer=AnswerContract(content=(ContentRequirement(
        kind="custom", mode="required", description="The copied text describes a scheduled retry.",
        evidence_ids=("request",),
    ),)))
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    decisions = DecisionsBoundary({"grounded_0": 0.01, "content_condition_0": probability})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))

    assert state["response"].kind == expected_kind
    if expected_kind == "answer":
        report = state["response"].result.semantic_validation
        assert [(check.name, check.probability) for check in report.checks] == [
            ("request_fulfilled", 0.99), ("content_condition:0", probability),
        ]


def test_copy_with_conflicting_local_format_remains_a_clarification():
    state = _state(answer=AnswerContract(format=(LineCountRequirement(
        kind="line_count", mode="required", value=2, evidence_ids=("request",),
    ),)))
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))

    assert state["response"].kind == "clarification"
    assert state["response"].result.actions == []


@pytest.mark.parametrize("minimum,maximum,expected_kind", [
    pytest.param(1, 1, "answer", id="one-sentence-copy-meets-quota"),
    pytest.param(2, 2, "clarification", id="copy-cannot-add-a-required-sentence"),
    pytest.param(0, 0, "clarification", id="copy-cannot-remove-a-forbidden-sentence"),
])
def test_exact_input_checks_whole_body_sentence_bounds_without_rewriting(
    minimum, maximum, expected_kind,
):
    request = f"The complete body must have exactly {minimum} sentences. " + REQUEST
    state = _state(request=request, answer=AnswerContract(sentence_counts=(SentenceCountRequirement(
        scope=None, minimum=minimum, maximum=maximum, evidence_ids=("request",),
    ),)))
    contract = state["runtime"].request_contract
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    # An immutable copy whose bounds conflict is clarified during preparation.
    assert state["response"].kind == ("draft" if expected_kind == "answer" else expected_kind)
    assert contract.body.source.text == ORIGINAL
    decisions = DecisionsBoundary()

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))

    assert state["response"].kind == expected_kind
    assert state["runtime"].request_contract == contract
    assert decisions.requests == []
    assert state["response"].result.actions == []
    if expected_kind == "answer":
        assert export_answer_text(state["response"].result) == ORIGINAL


@pytest.mark.parametrize("scope,probability,expected_kind", [
    pytest.param(None, 0.01, "answer", id="whole-body-count-needs-no-semantic-check"),
    pytest.param("사과 색상 설명", 0.99, "answer", id="copied-unit-belongs-to-current-scope"),
    pytest.param("바나나 색상 설명", 0.01, "failure", id="same-index-does-not-prove-new-scope"),
])
def test_copied_answer_rechecks_sentence_scope_meaning_in_the_current_contract(
    scope, probability, expected_kind,
):
    previous = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "paragraph", "content": [{
            "text": "사과는 빨갛습니다.", "basis": "interaction", "sentence_scope": 1,
        }],
    }]}), [])
    request = "이전 답변을 그대로 사용해 줘. " + (
        "전체 본문은 한 문장이어야 해." if scope is None else f"{scope}은 한 문장이어야 해."
    )
    contract = RequestContract(
        body=CopyAnswerBody(source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash)),
        answer=AnswerContract(sentence_counts=(SentenceCountRequirement(
            scope=scope, minimum=1, maximum=1, evidence_ids=("request",),
        ),)),
        evidence=(ContractEvidence(id="request", turn_id="current", quote=request,
                                   scope="current_request", interpretation="instruction"),),
    )
    state = build_graph_state_input(
        user_input=request, messages=[HumanMessage(content=request, id="current")],
        previous_response=previous, request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )
    preparation = DecisionsBoundary()
    prepare_synthesis_state(state, decisions=preparation)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    assert state["response"].result.content == previous.content
    decisions = DecisionsBoundary({"request_fulfilled": probability})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))

    assert state["response"].kind == expected_kind
    assert state["runtime"].request_contract == contract
    assert state["runtime"].previous_response == previous
    if scope is None:
        assert preparation.requests == []
        assert decisions.requests == []
    else:
        assert preparation.requests
        assert decisions.requests
        payload = json.loads(decisions.requests[0]["input"])
        assert "sentence_counts" not in payload["confirmed_request"]["answer"]
        assert payload["answer_scope"]["sentence_scope_bindings"][0]["scope"] == scope
    if expected_kind == "answer":
        assert state["response"].result.content == previous.content
    else:
        assert state["response"].result.content != previous.content
        assert state["response"].result.actions == []


def test_copy_with_semantic_report_can_be_revalidated_without_changing_the_body():
    state = _state(answer=AnswerContract(content=(ContentRequirement(
        kind="custom", mode="required", description="The copied text describes a scheduled retry.",
        evidence_ids=("request",),
    ),)))
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    validate = make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())
    state.update(validate(state))
    checked = state["response"].result

    state.update(validate(state))

    assert state["response"].kind == "answer"
    assert state["response"].result == checked


@pytest.mark.parametrize("basis", ["example", "inference"])
def test_same_text_with_changed_content_role_is_not_an_exact_input_copy(basis):
    state = _state()
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(text_document(ORIGINAL, basis=basis), []),
    })

    with pytest.raises(LLMCallError) as caught:
        make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert caught.value.problem.code == "model_output_invalid"
