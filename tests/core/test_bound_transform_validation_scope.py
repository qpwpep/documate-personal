"""Bound answer edits separate operation fidelity from contextual judgments."""
from copy import deepcopy
import hashlib
import json

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import finalize_answer, text_document
from src.core.llm_errors import LLMCallError
from src.core.request_bodies import BoundInputText, ComposeBody, TransformInputBody
from src.core.request_contracts import ContractEvidence, RequestContract
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _evidence, _state


REQUEST = "마지막 줄의 42만 43으로 고쳐서 저장해줘. 추가 설명과 표는 넣지 마."


def _bound_edit_state():
    state = _state(retrieval=False, answer="마지막 줄: 43")
    previous = finalize_answer(text_document("마지막 줄: 42"), [])
    contract = RequestContract(
        request_id=state["runtime"].request_contract.request_id,
        body={"kind": "transform_answer", "change_requirement": "required", "source": {"ref": "previous", "response_hash": previous.content_hash},
              "instruction": "마지막 줄의 42만 43으로 바꾸고 나머지는 유지한다.", "evidence_ids": ["edit"]},
        evidence=[ContractEvidence(id="edit", turn_id="current", quote=REQUEST,
                                   scope="current_request", interpretation="instruction")],
        answer={"content": [{"kind": "explanation", "mode": "forbidden", "description": "추가 설명",
                             "evidence_ids": ["edit"]}],
                "format": [{"kind": "table", "mode": "forbidden", "evidence_ids": ["edit"]}]},
    )
    state["runtime"] = state["runtime"].model_copy(update={
        "user_input": REQUEST, "previous_response": previous, "request_contract": contract,
    })
    state["messages"] = [HumanMessage(content="마지막 줄: 42를 그대로 저장해줘."), HumanMessage(content=REQUEST)]
    return state


def test_bound_answer_operation_preserves_the_body_and_contextual_checks_in_separate_calls():
    state = _bound_edit_state()
    original = deepcopy(state)
    boundary = DecisionsBoundary()

    with capture_llm_diagnostics():
        request_call_budget("post_synthesis_validation").max_calls = 2
        updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    operation, = [call for call in boundary.requests
                  if [question["name"] for question in call["questions"]] == ["request_fulfilled"]]
    contextual, = [call for call in boundary.requests if call is not operation]
    scoped = json.loads(operation["input"])
    full = json.loads(contextual["input"])
    assert {question["name"] for question in contextual["questions"]} == {
        "grounded_0", "content_condition_0", "format_fulfilled",
    }
    assert scoped["confirmed_request"] == {
        "body": state["runtime"].request_contract.body.model_dump(mode="json", exclude={"evidence_ids"}),
    }
    assert "latest_user_request" not in scoped
    assert "conversation_context" not in scoped
    assert "units" not in scoped
    assert full["latest_user_request"] == REQUEST
    assert full["conversation_context"]["messages"] == [{"role": "user", "content": "마지막 줄: 42를 그대로 저장해줘."}]
    assert full["confirmed_request"]["evidence"] == [item.model_dump(mode="json") for item in state["runtime"].request_contract.evidence]
    assert full["confirmed_request"]["body"]["evidence_ids"] == ["edit"]
    for field in ("bound_source_response", "rendered_response", "answer_scope", "retrieval_required",
                  "retrieval_notice", "retrieval_gaps", "terminal_failed_requirement_ids"):
        assert scoped[field] == full[field]
    assert scoped["bound_source_response"]["content"] == state["runtime"].previous_response.content.model_dump(mode="json")
    assert updates["response"].kind == "answer"
    assert updates["response"].result.content == state["response"].result.content
    assert updates["response"].result.semantic_validation.passed
    assert state == original


@pytest.mark.parametrize("context_dependency", ["transform_input", "compose", "prior_retrieval", "current_retrieval", "prior_citation"])
def test_operations_with_other_context_dependencies_keep_the_existing_full_context(context_dependency):
    state = _bound_edit_state()
    runtime = state["runtime"]
    if context_dependency == "transform_input":
        text = "마지막 줄: 42"
        body = TransformInputBody(change_requirement="required", source=BoundInputText(
            turn_id="current", text=text, start=0, end=len(text),
            content_hash=hashlib.sha256(text.encode()).hexdigest(),
        ), instruction=runtime.request_contract.body.instruction, evidence_ids=["edit"])
        state["runtime"] = runtime.model_copy(update={"request_contract": runtime.request_contract.model_copy(update={"body": body})})
    elif context_dependency == "compose":
        state["runtime"] = runtime.model_copy(update={"request_contract": runtime.request_contract.model_copy(update={
            "body": ComposeBody(instruction="43으로 고쳐줘."),
        })})
    elif context_dependency == "prior_retrieval":
        state["runtime"] = runtime.model_copy(update={
            "previous_response": runtime.previous_response.model_copy(update={"retrieval_required": True}),
        })
    elif context_dependency == "current_retrieval":
        state["response"] = state["response"].model_copy(update={
            "result": state["response"].result.model_copy(update={"retrieval_required": True}),
        })
    else:
        evidence = _evidence("마지막 줄: 42")
        previous = finalize_answer(text_document("마지막 줄: 42", basis="source", refs=[evidence.id]), [evidence])
        body = runtime.request_contract.body.model_copy(update={
            "source": runtime.request_contract.body.source.model_copy(update={"response_hash": previous.content_hash}),
        })
        state["runtime"] = runtime.model_copy(update={
            "previous_response": previous, "request_contract": runtime.request_contract.model_copy(update={"body": body}),
        })
    boundary = DecisionsBoundary()

    make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    call, = [call for call in boundary.requests
             if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    payload = json.loads(call["input"])
    assert payload["latest_user_request"] == REQUEST
    assert payload["conversation_context"]["messages"]
    assert any(question["name"] == "grounded_0" for question in call["questions"])


def test_bound_edit_reserves_the_operation_call_before_starting_contextual_judgments():
    boundary = DecisionsBoundary()
    with capture_llm_diagnostics():
        budget = request_call_budget("post_synthesis_validation")
        budget.max_calls = 1
        with pytest.raises(LLMCallError) as failure:
            make_post_synthesis_validation_node(False, decisions_client=boundary)(_bound_edit_state())
    assert failure.value.problem.code == "call_budget_exhausted"
    assert boundary.requests == []
    assert budget.calls == 0


@pytest.mark.parametrize("failed_check", ["grounded_0", "content_condition_0", "format_fulfilled", "request_fulfilled"])
def test_a_successful_edit_never_overrides_an_independent_failed_check(failed_check):
    boundary = DecisionsBoundary({failed_check: 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(_bound_edit_state())

    assert updates["retry"].needs_retry
    assert {question["name"] for call in boundary.requests for question in call["questions"]} == {
        "grounded_0", "content_condition_0", "format_fulfilled", "request_fulfilled",
    }
