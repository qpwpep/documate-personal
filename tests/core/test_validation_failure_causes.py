"""Keep the failed answer check distinct from an already known retrieval gap."""
from __future__ import annotations

import pytest

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import GraphState, RetrievalDiagnostic, RetryState
from src.core.llm_errors import make_problem
from src.core.planner_schema import RetrievalTask
from src.core.request_contracts import AnswerContract, LayoutRequirement
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state


def _partial_state() -> GraphState:
    state = _state(repairs=0)
    state["planner"].output.tasks.append(RetrievalTask(
        route="docs", query="pathlib.Path.is_dir behavior", k=1, requirement_id="path_is_dir",
        requirement={"library": "Python", "symbols": ["pathlib.Path.is_dir"], "match": "symbol"},
    ))
    state["debug"].retrieval_diagnostics.append(RetrievalDiagnostic(
        route="docs", requirement_id="path_is_dir", status="no_result", answerability="missing", evidence_count=0,
    ))
    state["retry"] = RetryState(
        max_retries=0, max_content_repairs=0, terminal_failed_requirement_ids=["path_is_dir"],
        terminal_retrieval_reason="no_evidence",
    )
    state["response"].normal_evidence_missing_requirement_ids = ["path_is_dir"]
    return state


@pytest.mark.parametrize("predicate", ["request_fulfilled", "grounded_0", "requirement_0"])
def test_rejected_partial_answer_reports_validation_failure_and_keeps_retrieval_gap(predicate):
    state = _partial_state()
    original_body = state["response"].result.content
    decisions = DecisionsBoundary({predicate: 0.1})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["problem"]["code"] == "answer_validation_failed"
    assert outcome["problem"]["stage"] == "validation"
    assert outcome["problem"]["next_action"] == "retry_later"
    assert outcome["status"] == "partial"
    assert state["response"].result.content != original_body
    assert "retrieved_evidence_incomplete" in {issue["code"] for issue in outcome["response"]["issues"]}
    assert any(question["name"] == predicate for request in decisions.requests for question in request["questions"])


def test_local_output_condition_failure_has_validation_cause_even_with_terminal_gap():
    state = _partial_state()
    contract = state["runtime"].request_contract
    state["runtime"].request_contract = contract.model_copy(update={
        "answer": AnswerContract(format=(LayoutRequirement(kind="list", mode="required"),)),
    })

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["problem"]["code"] == "answer_validation_failed"
    assert outcome["problem"]["stage"] == "validation"
    assert "retrieved_evidence_incomplete" in {issue.code for issue in state["response"].result.issues}


def test_accepted_partial_answer_keeps_retrieval_cause_and_body():
    state = _partial_state()
    original_body = state["response"].result.content

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["problem"]["code"] == "evidence_insufficient"
    assert outcome["problem"]["stage"] == "retrieval"
    assert outcome["status"] == "partial"
    assert state["response"].result.content == original_body


@pytest.mark.parametrize("code", ["provider_unavailable", "model_output_incomplete"])
def test_existing_provider_or_generation_failure_keeps_its_cause_and_gap(code):
    state = _partial_state()
    problem = make_problem(code, "synthesis")
    state["response"].problem = problem
    decisions = DecisionsBoundary({"request_fulfilled": 0.1})

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["problem"] == problem.model_dump(mode="json")
    assert "retrieved_evidence_incomplete" in {issue["code"] for issue in outcome["response"]["issues"]}
    assert decisions.requests == []


def test_unresolved_source_reference_remains_invalid_model_output():
    state = _partial_state()
    state["response"].result = finalize_answer(
        text_document("Path.exists() returns True.", basis="source", refs=["missing-source"]),
        state["response"].evidence_packet, retrieval_required=True,
    )

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))
    outcome = ResponseAssembler().assemble(response=state, debug_info={})

    assert outcome["problem"]["code"] == "model_output_invalid"
    assert outcome["problem"]["stage"] == "validation"
    assert "retrieved_evidence_incomplete" in {issue["code"] for issue in outcome["response"]["issues"]}
