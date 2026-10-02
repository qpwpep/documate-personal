"""Provider failures remain system failures through the public runtime nodes."""
from __future__ import annotations

import httpx
import pytest
from openai import BadRequestError, InternalServerError

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import ResponseState
from src.core.contracts.debug import RetryState
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import PlannerOutput, RetrievalPlanOutput
from src.core.request_contracts import AnswerContract, ContractEvidence, FormatRequirement, RequestContract, UnresolvedBody, WireRequestContract
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_synthesis_validation import _hit, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


class ModelReplies:
    """Only the external model boundary is replaced; all node logic stays real."""

    def __init__(self, *replies):
        self.replies = replies
        self.calls = 0

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply


def schema_error():
    return BadRequestError(
        "Invalid schema: oneOf is not permitted",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
        body={"code": "invalid_json_schema", "param": "response_format", "message": "Invalid schema"},
    )


def test_planner_provider_schema_failure_is_never_a_missing_user_contract():
    model = ModelReplies(schema_error())
    state = build_graph_state_input(user_input="Explain Python context managers")

    with pytest.raises(LLMCallError) as caught:
        make_planner_node(model, False)(state)

    assert caught.value.problem.code == "provider_schema_invalid"
    assert caught.value.problem.stage == "planner"
    assert model.calls == 1
    assert state["runtime"].request_contract is None


def test_planner_missing_initial_contract_repairs_output_before_binding():
    valid = PlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract(slack_recipient={"state": "omitted"}))
    model = ModelReplies({"use_retrieval": False, "tasks": [], "request_contract": None}, valid)

    result = make_planner_node(model, False)(build_graph_state_input(user_input="안녕"))

    assert model.calls == 2
    assert result["runtime"].request_contract.failure is None
    assert result["planner"].guided_followup is None


def test_planner_exhausted_output_validation_is_a_model_failure():
    model = ModelReplies({"use_retrieval": False, "tasks": [], "request_contract": None})

    with pytest.raises(LLMCallError) as caught:
        make_planner_node(model, False)(build_graph_state_input(user_input="안녕"))

    assert caught.value.problem.code == "model_output_invalid"
    assert model.calls == 2


def test_replanning_uses_its_own_output_contract_and_keeps_bound_facts():
    contract = RequestContract()
    initial = ModelReplies(AssertionError("initial interpreter must not run during retrieval retry"))
    retry = ModelReplies(RetrievalPlanOutput(use_retrieval=False, tasks=[]))
    state = build_graph_state_input(user_input="same request", request_contract=contract)

    result = make_planner_node(initial, False, llm_planner_retry=retry)(state)

    assert initial.calls == 0
    assert retry.calls == 1
    assert result["runtime"].request_contract == contract


def test_real_missing_user_information_is_a_successful_clarification():
    proposal = WireRequestContract(slack_recipient={"state": "omitted"}, body=UnresolvedBody(question="어떤 주제를 설명할까요?"))
    model = ModelReplies(PlannerOutput(use_retrieval=False, tasks=[], request_contract=proposal))

    result = make_planner_node(model, False)(build_graph_state_input(user_input="그것을 설명해줘"))

    assert model.calls == 1
    assert result["runtime"].request_contract.failure is None
    assert result["planner"].diagnostics.reason == "clarification_required"
    assert result["planner"].guided_followup == "어떤 주제를 설명할까요?"
