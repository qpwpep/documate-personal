"""Requirement criteria retain the provider batch limit and shared role budget."""
import json

import pytest

from src.core.contracts import PlannerState
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import PlannerOutput
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state


@pytest.mark.parametrize("requirement_count,question_batches", [(100, [200]), (101, [200, 2])])
def test_post_requirement_batch_boundary_preserves_each_check_and_fails_at_the_existing_budget(
    requirement_count, question_batches,
):
    state = _state()
    original_task = state["planner"].output.tasks[0]
    tasks = [original_task.model_copy(update={"requirement_id": f"requirement_{index}"})
             for index in range(requirement_count)]
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks))
    state["response"] = state["response"].model_copy(update={
        "evidence_requirement_map": {
            item.id: [task.requirement_id for task in tasks]
            for item in state["response"].evidence_packet
        },
    })
    boundary = DecisionsBoundary()
    validate = make_post_synthesis_validation_node(False, decisions_client=boundary)

    with capture_llm_diagnostics():
        state.update(validate(state))
        requirement_calls = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
        assert [len(call["questions"]) for call in requirement_calls] == question_batches
        checks = state["response"].result.semantic_validation.checks
        assert len([check for check in checks if check.name.startswith("requirement_")]) == 2 * requirement_count

        calls_per_revision = 2 + len(question_batches)
        for _ in range(36 // calls_per_revision - 1):
            state.update(validate(state))
        retained_response = state["response"]
        observed_requests = len(boundary.requests)
        with pytest.raises(LLMCallError) as failure:
            validate(state)
        budget = request_call_budget("post_synthesis_validation")

    assert failure.value.problem.code == "call_budget_exhausted"
    assert failure.value.diagnostic.budget_reason == "calls"
    assert state["response"] == retained_response
    assert len(boundary.requests) == observed_requests
    assert all(len(call["questions"]) <= 200 for call in boundary.requests)
    assert budget.calls == observed_requests == 36
    assert budget.max_calls == 36
    assert budget.timeout_seconds == 45
