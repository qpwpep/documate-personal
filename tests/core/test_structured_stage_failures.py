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


def test_synthesis_schema_failure_does_not_retry_or_hide_behind_source_fallback():
    primary = ModelReplies(schema_error())
    compact = ModelReplies(AssertionError("compact must not run for invalid schema"))

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(primary, compact, excerpt_limits=synthesis_excerpt_limits())(_state([_hit()]))

    assert caught.value.problem.code == "provider_schema_invalid"
    assert primary.calls == 1
    assert compact.calls == 0


def test_service_retry_after_cannot_be_bypassed_by_compact_generation():
    unavailable = InternalServerError(
        "service unavailable", body={"code": "server_error"},
        response=httpx.Response(503, headers={"Retry-After": "60"},
                                request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
    )
    primary = ModelReplies(unavailable)
    compact = ModelReplies(AssertionError("Retry-After must not be bypassed by a compact request"))

    result = make_synthesize_node(primary, compact, excerpt_limits=synthesis_excerpt_limits())(_state([_hit()]))

    assert primary.calls == 1
    assert compact.calls == 0
    assert result["response"].problem.code == "provider_unavailable"
    assert result["response"].problem.retry_after_seconds == 60


def test_synthesis_invalid_output_without_sources_remains_a_model_failure():
    model = ModelReplies({"answer": "unvalidated model prose"})

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits())(_state([]))

    assert caught.value.problem.code == "model_output_invalid"
    assert model.calls == 2


def test_synthesis_source_fallback_exposes_the_original_typed_problem():
    model = ModelReplies({"answer": "unvalidated model prose"})
    hit = _hit()

    result = make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits())(_state([hit]))

    assert model.calls == 2
    assert result["response"].kind == "failure"
    assert result["response"].problem.code == "model_output_invalid"
    assert result["response"].result.citations[0].evidence == hit.evidence
    validated = make_post_synthesis_validation_node(False)({**_state([hit]), **result})
    assert validated["response"] == result["response"]
    assert not validated["retry"].needs_retry


def test_missing_contract_cannot_become_a_synthesis_clarification():
    state = _state([])
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": None})
    model = ModelReplies(AssertionError("must stop before generation"))

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits())(state)

    assert caught.value.problem.code == "internal_error"
    assert model.calls == 0


def test_exhausted_model_format_violation_is_not_missing_user_information():
    contract = RequestContract(
        answer=AnswerContract(format=(FormatRequirement(kind="line_count", mode="required", value=3, evidence_ids=("r1",)),)),
        evidence=(ContractEvidence(id="r1", turn_id="current", quote="세 줄로 설명해줘", scope="answer.format.line_count", interpretation="instruction"),),
    )
    state = build_graph_state_input(user_input="세 줄로 설명해줘", request_contract=contract,
                                    retry=RetryState(max_retries=0))
    model = ModelReplies(text_document("한 줄만 생성한 결과").model_dump(mode="json"))

    state.update(make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits())(state))
    state.update(make_post_synthesis_validation_node(False)(state))
    result = ResponseAssembler().assemble(response=state, debug_info={})

    assert result["status"] == "failed"
    assert result["response"] is None
    assert result["problem"]["code"] == "model_output_invalid"
    assert result["problem"]["stage"] == "validation"
    assert result["problem"]["next_action"] == "retry_later"
    assert "질문을 수정할 필요는 없습니다" in result["message"]


def test_unclassified_runtime_failure_does_not_claim_evidence_is_missing():
    state = build_graph_state_input(user_input="요청", request_contract=RequestContract())
    state["response"] = ResponseState(kind="failure", result=finalize_answer(text_document("실패"), []))

    result = ResponseAssembler().assemble(response=state, debug_info={})

    assert result["status"] == "failed"
    assert result["problem"]["code"] == "internal_error"
    assert result["problem"]["next_action"] == "none"
