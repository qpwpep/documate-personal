"""User-input analysis keeps request interpretation separate from source citations."""
import hashlib
import json

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, BoundAnswerReference, BoundInputText, ComposeBody,
    ContentRequirement, ContractEvidence, RequestContract, TransformAnswerBody, TransformInputBody,
)
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import mock_chat_model
from tests.core.test_decisions_validation import DecisionsBoundary
from tests.core.test_synthesis_validation import _hit, _state as retrieval_state
from tests.synthesis_fixtures import prepare_synthesis_state


CODE = """def unique_in_order(values):
    seen = set()
    result = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result"""
REQUEST = "이 코드의 중복 처리, 추가 메모리, 빈 입력의 반환값을 설명하고 예시 없이 TXT로 저장해줘.\n" + CODE
ANALYSIS = "처음 만난 값만 결과에 추가하므로 중복을 제거하며 처음 등장한 순서를 유지합니다. 서로 다른 값의 수를 k라 하면 집합과 결과 리스트의 추가 공간은 O(k)입니다. 빈 입력에는 빈 리스트를 반환합니다."
INTERPRETATION_ID = "ev:request-analysis:4:user-code:1"


class SynthesisBoundary:
    def __init__(self, document=None):
        self.document = document or text_document(ANALYSIS, basis="interaction")
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return self.document.model_dump(mode="json")


def _state(body_kind="compose"):
    instruction = "사용자가 제공한 코드의 중복 처리, 추가 메모리, 빈 입력 반환값을 설명한다."
    body = ComposeBody(instruction=instruction, evidence_ids=(INTERPRETATION_ID,))
    source_response = None
    if body_kind == "transform_input":
        body = TransformInputBody(
            source=BoundInputText(
                turn_id="user-code", text=CODE, start=0, end=len(CODE),
                content_hash=hashlib.sha256(CODE.encode()).hexdigest(),
            ),
            instruction=instruction, evidence_ids=(INTERPRETATION_ID,),
        )
    elif body_kind == "transform_answer":
        source_response = finalize_answer(text_document(CODE, basis="interaction"), [])
        body = TransformAnswerBody(
            source=BoundAnswerReference(ref="previous", response_hash=source_response.content_hash),
            instruction=instruction, evidence_ids=(INTERPRETATION_ID,),
        )
    contract = RequestContract(
        body=body, body_request=ComposeBody(instruction=instruction, evidence_ids=(INTERPRETATION_ID,)),
        answer=AnswerContract(content=(
            *(ContentRequirement(kind="explanation", mode="required", description=description,
                                 evidence_ids=(INTERPRETATION_ID,))
              for description in ("중복 처리", "추가 메모리", "빈 입력 반환값")),
            ContentRequirement(kind="example", mode="forbidden", evidence_ids=(INTERPRETATION_ID,)),
        )),
        actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=(INTERPRETATION_ID,))),
        evidence=(ContractEvidence(id=INTERPRETATION_ID, turn_id="user-code", quote=REQUEST,
                                   scope="current_request", interpretation="instruction"),),
    )
    return build_graph_state_input(
        user_input=REQUEST, messages=[HumanMessage(content=REQUEST)], request_contract=contract,
        previous_response=source_response,
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )


@pytest.mark.parametrize("body_kind", ["compose", "transform_input", "transform_answer"])
def test_synthesis_keeps_request_interpretation_ids_out_of_citation_context(body_kind):
    state = _state(body_kind)
    original = state["runtime"].request_contract.model_dump(mode="json")
    boundary = SynthesisBoundary()

    updates = make_synthesize_node(mock_chat_model(boundary.invoke))(prepare_synthesis_state(state))

    messages = [str(message.content) for message in boundary.messages]
    assert INTERPRETATION_ID not in "\n".join(messages)
    contract_message = next(message for message in messages if message.startswith("[Confirmed Request Contract]"))
    projected = json.JSONDecoder().raw_decode(contract_message[contract_message.index("{"):])[0]
    assert "evidence" not in projected
    assert "body_request" not in projected
    assert projected["request_interpretation"] == [
        {key: item[key] for key in ("turn_id", "quote", "scope", "interpretation")}
        for item in original["evidence"]
    ]
    assert projected["body"]["kind"] == body_kind
    assert projected["body"]["instruction"] == original["body"]["instruction"]
    assert [(item["kind"], item["mode"], item["description"]) for item in projected["answer"]["content"]] == [
        (item["kind"], item["mode"], item["description"]) for item in original["answer"]["content"]
    ]
    assert all("replaces" not in item for item in projected["answer"]["content"])
    assert projected["actions"]["save_text"]["intent"] == "requested"
    assert REQUEST in messages
    if body_kind == "transform_input":
        bound_message = next(message for message in messages if message.startswith("[Bound User Input]"))
        bound_input = json.loads(bound_message[bound_message.index("{"):])
        assert bound_input == original["body"]["source"]
        assert projected["body"]["source"]["content_hash"] == bound_input["content_hash"]
    elif body_kind == "transform_answer":
        assert projected["body"]["source"] == original["body"]["source"]
        bound_message = next(message for message in messages if message.startswith("[Bound Source Answer]"))
        bound_answer = json.loads(bound_message[bound_message.index("{"):])
        assert bound_answer["response_hash"] == state["runtime"].previous_response.content_hash
    assert state["runtime"].request_contract.model_dump(mode="json") == original
    assert updates["response"].result.content == boundary.document


def test_real_evidence_packet_references_remain_available_to_synthesis():
    hit = _hit("A supplied source confirms this setting.")
    state = retrieval_state([hit])
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": _state()["runtime"].request_contract,
    })
    boundary = SynthesisBoundary(text_document(hit.evidence.excerpt, basis="source", refs=["e1"]))

    updates = make_synthesize_node(mock_chat_model(boundary.invoke))(prepare_synthesis_state(state))

    packet_message = next(str(message.content) for message in boundary.messages
                          if str(message.content).startswith("[Evidence Packet]"))
    packet = json.loads(packet_message[packet_message.index("[", len("[Evidence Packet]")):])
    assert packet[0]["id"] == "e1"
    assert packet[0]["excerpt"] == hit.evidence.excerpt
    assert updates["response"].result.citations[0].evidence.id == hit.evidence.id


def test_request_interpretation_id_is_still_rejected_as_an_answer_citation():
    document = text_document(ANALYSIS, basis="source", refs=[INTERPRETATION_ID])
    state = _state()
    state.update(make_synthesize_node(mock_chat_model(SynthesisBoundary(document).invoke))(prepare_synthesis_state(state)))

    updates = make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert state["response"].result.citations == []
    assert updates["response"].kind == "failure"
    assert updates["response"].problem.code == "model_output_invalid"


def test_user_code_analysis_is_judged_against_the_supplied_code_without_invented_citations():
    state = _state()
    state.update(make_synthesize_node(mock_chat_model(SynthesisBoundary().invoke))(prepare_synthesis_state(state)))
    decisions = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=decisions)(state)

    assessment = json.loads(decisions.requests[0]["input"])
    assert assessment["conversation_context"]["messages"] == [{"role": "user", "content": REQUEST}]
    assert assessment["units"][0]["basis"] == "interaction"
    assert assessment["units"][0]["refs"] == []
    assert assessment["units"][0]["cited_sources"] == []
    assert assessment["displayed_document"] == state["response"].result.content.model_dump(mode="json")
    assert updates["response"].kind == "answer"
    assert updates["response"].result.semantic_validation.passed


def test_user_code_input_does_not_exempt_an_unsupported_execution_claim_from_validation():
    document = text_document("실제 서버에서 코드를 실행해 처리 결과를 확인했습니다.", basis="interaction")
    state = _state()
    state.update(make_synthesize_node(mock_chat_model(SynthesisBoundary(document).invoke))(prepare_synthesis_state(state)))
    decisions = DecisionsBoundary({"grounded_0": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=decisions)(state)

    assessment = json.loads(decisions.requests[0]["input"])
    assert assessment["units"][0]["text"] == "실제 서버에서 코드를 실행해 처리 결과를 확인했습니다."
    assert assessment["units"][0]["cited_sources"] == []
    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None
