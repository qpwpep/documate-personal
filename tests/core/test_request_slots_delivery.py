from hashlib import sha256

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import PlannerState, ResponseState, RuntimeState
from src.core.contracts.graph_state import PendingAction
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, BoundAnswerReference, BoundInputText, ComposeBody, ContractEvidence,
    CopyAnswerBody, CopyInputBody, MissingInformation, RequestContract, TransformAnswerBody, TransformInputBody,
)
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_pending_action_delivery import delivery_tools
from tests.synthesis_fixtures import synthesis_excerpt_limits


class DocumentModel:
    def __init__(self, answer=None):
        self.answer = answer
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        if self.answer is None:
            raise AssertionError("a copied input must not call the model")
        return text_document(self.answer).model_dump(mode="json")


def _evidence():
    return (ContractEvidence(id="request", turn_id="current", quote="요청", scope="current_request", interpretation="instruction"),)


def _contract(body, *, save="not_requested", slack="not_requested", missing_info=(), request_id="slot-request", **kwargs):
    return RequestContract(
        request_id=request_id, body=body, missing_info=missing_info, evidence=_evidence(),
        actions=ActionContract(
            save_text=ActionRequest(intent=save, evidence_ids=("request",)),
            slack_notify=ActionRequest(intent=slack, evidence_ids=("request",)),
        ), **kwargs,
    )


def _input(text):
    return BoundInputText(turn_id="current", text=text, start=0, end=len(text), content_hash=sha256(text.encode("utf-8")).hexdigest())


def _state(contract, *, pending=None):
    return {
        "runtime": RuntimeState(user_input="요청", request_contract=contract, pending_action=pending),
        "planner": PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract.to_wire())),
        "messages": [HumanMessage(content="요청", id="current")],
    }


def _prepare(state, model):
    for node in (make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits()), make_post_synthesis_validation_node(False)):
        state.update(node(state))


def test_copy_input_preserves_exact_text_in_screen_file_and_slack(delivery_tools):
    """직접 지정한 입력의 공백·개행을 모델 수정 없이 화면·파일·Slack에 동일하게 전달한다."""
    save, slack, delivered, output = delivery_tools
    original = "  첫 줄\r\n둘째 줄  \r\n"
    contract = _contract(CopyInputBody(source=_input(original)), save="requested", slack="requested",
                         slack_recipient={"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"]})
    state = _state(contract)

    _prepare(state, DocumentModel())
    state.update(make_action_postprocess_node(save, slack, False)(state))

    answer = state["response"].result
    assert state["response"].kind == "answer"
    assert export_answer_text(answer) == original
    assert next(output.glob("*.txt")).read_bytes().decode("utf-8-sig") == original
    assert [item["payload"]["text"] for item in delivered] == [original]
























