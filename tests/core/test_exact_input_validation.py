"""Exact input copying verifies fidelity before authorizing delivery."""
from hashlib import sha256
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, BoundInputText, ContentRequirement, ContractEvidence,
    CopyInputBody, LineCountRequirement, RequestContract,
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


def _state(*, answer=None):
    contract = RequestContract(
        body=CopyInputBody(source=BoundInputText(
            turn_id="current", text=ORIGINAL, start=REQUEST.index(ORIGINAL), end=len(REQUEST),
            content_hash=sha256(ORIGINAL.encode("utf-8")).hexdigest(),
        )),
        actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("request",))),
        evidence=(ContractEvidence(id="request", turn_id="current", quote=REQUEST,
                                   scope="current_request", interpretation="instruction"),),
        answer=answer or AnswerContract(),
    )
    return build_graph_state_input(
        user_input=REQUEST, messages=[HumanMessage(content=REQUEST, id="current")],
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
    decisions = DecisionsBoundary({"grounded_0": 0.01, "request_fulfilled": probability})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))

    assert state["response"].kind == expected_kind
    if expected_kind == "answer":
        report = state["response"].result.semantic_validation
        assert [(check.name, check.probability) for check in report.checks] == [("request_fulfilled", probability)]


def test_copy_with_conflicting_local_format_remains_a_clarification():
    state = _state(answer=AnswerContract(format=(LineCountRequirement(
        kind="line_count", mode="required", value=2, evidence_ids=("request",),
    ),)))
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))

    assert state["response"].kind == "clarification"
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
