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
    for node in (make_synthesize_node(model), make_post_synthesis_validation_node(False)):
        state.update(node(state))


def test_copy_input_preserves_exact_text_in_screen_file_and_slack(delivery_tools):
    """직접 지정한 입력의 공백·개행을 모델 수정 없이 화면·파일·Slack에 동일하게 전달한다."""
    save, slack, delivered, output = delivery_tools
    original = "  첫 줄\r\n둘째 줄  \r\n"
    contract = _contract(CopyInputBody(source=_input(original)), save="requested", slack="requested",
                         slack_destination={"channel_id": "C123"})
    state = _state(contract)

    _prepare(state, DocumentModel())
    state.update(make_action_postprocess_node(save, slack, False)(state))

    answer = state["response"].result
    assert state["response"].kind == "answer"
    assert export_answer_text(answer) == original
    assert next(output.glob("*.txt")).read_bytes().decode("utf-8-sig") == original
    assert [item["payload"]["text"] for item in delivered] == [original]


def test_translated_input_delivers_the_transformation_not_the_instruction(delivery_tools):
    """입력 번역 요청은 선택한 원문을 변환한 단일 본문을 저장하고 전송한다."""
    save, slack, delivered, output = delivery_tools
    source = "안녕하세요."
    contract = _contract(TransformInputBody(source=_input(source), instruction="영어로 번역한다.", evidence_ids=("request",)),
                         save="requested", slack="requested", slack_destination={"channel_id": "C123"})
    state = _state(contract)
    model = DocumentModel("Hello.")

    _prepare(state, model)
    state.update(make_action_postprocess_node(save, slack, False)(state))

    assert source in "\n".join(str(message.content) for message in model.messages)
    assert export_answer_text(state["response"].result) == "Hello."
    assert next(output.glob("*.txt")).read_bytes().decode("utf-8-sig") == "Hello."
    assert [item["payload"]["text"] for item in delivered] == ["Hello."]


def test_known_save_and_body_proceed_while_slack_intent_waits(delivery_tools):
    """Slack 실행 의사가 불명확해도 확정된 본문과 저장은 처리하고 그 전송만 보류한다."""
    save, slack, delivered, output = delivery_tools
    contract = _contract(ComposeBody(instruction="리스트를 설명한다."), save="requested", slack="unresolved",
                         missing_info=(MissingInformation(slot="slack_intent", reason="unclear", question="Slack에도 보낼까요?"),))
    state = _state(contract)

    _prepare(state, DocumentModel("리스트는 여러 항목을 순서대로 모은 것입니다."))
    state.update(make_action_postprocess_node(save, slack, False)(state))

    assert state["response"].kind == "answer"
    assert next(output.glob("*.txt")).exists()
    assert delivered == []
    assert state["runtime"].pending_action.response.content == state["response"].result.content
    assert state["runtime"].pending_action.completed_actions == ("save_text",)
    assert state["runtime"].pending_action.phase == "awaiting_input"


def test_new_subject_clarification_preserves_request_without_promoting_question(delivery_tools):
    """본문이 없는 주제 질문은 보류 계약으로 남고 질문 문장은 전달 본문이 되지 않는다."""
    from src.core.request_contracts import UnresolvedBody

    save, slack, delivered, output = delivery_tools
    contract = _contract(UnresolvedBody(question="어느 API를 설명할까요?"), slack="requested",
                         missing_info=(MissingInformation(slot="subject", reason="not_provided", question="어느 API를 설명할까요?"),))
    state = _state(contract)

    _prepare(state, DocumentModel())
    state.update(make_action_postprocess_node(save, slack, False)(state))

    assert state["response"].kind == "clarification"
    assert state["runtime"].pending_action is not None
    assert state["runtime"].pending_action.response is None
    assert state["runtime"].pending_action.phase == "awaiting_input"
    assert delivered == []
    assert not output.exists()


def test_previous_revision_answer_is_not_delivered_as_the_current_request(delivery_tools):
    """다른 계약 revision에서 검증된 본문을 새 요청의 완성 본문으로 전송하지 않는다."""
    save, slack, delivered, output = delivery_tools
    contract = _contract(ComposeBody(instruction="새 본문을 작성한다."), save="requested", slack="requested",
                         slack_destination={"channel_id": "C123"}, revision=2)
    state = _state(contract)
    state["response"] = ResponseState(result=finalize_answer(text_document("이전 본문"), []), kind="answer",
                                      request_id=contract.request_id, contract_revision=1)

    make_action_postprocess_node(save, slack, False)(state)

    assert delivered == []
    assert not output.exists()


@pytest.mark.parametrize("slack_intent", ["not_requested", "forbidden"])
def test_body_only_clarification_keeps_known_constraints_without_a_delivery_request(delivery_tools, slack_intent):
    """저장·전송 요청이 없어도 미해결 주제의 코드 금지와 출력 조건을 보류 계약에 보존한다."""
    from src.core.request_contracts import UnresolvedBody

    save, slack, delivered, output = delivery_tools
    contract = _contract(UnresolvedBody(question="어느 API인가요?"), slack=slack_intent,
                         missing_info=(MissingInformation(slot="subject", reason="not_provided", question="어느 API인가요?"),),
                         answer={"content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["request"]}]})
    state = _state(contract)

    _prepare(state, DocumentModel())
    state.update(make_action_postprocess_node(save, slack, False)(state))

    pending = state["runtime"].pending_action
    assert pending is not None
    assert pending.response is None
    assert pending.contract.answer == contract.answer
    assert pending.contract.actions.slack_notify.intent == slack_intent
    assert pending.phase == "awaiting_input"
    assert not pending.body_prepared
    assert delivered == []
    assert not output.exists()


def test_subject_then_destination_supplements_deliver_one_prepared_revision(delivery_tools):
    """주제 질문부터 시작한 요청은 조건을 보존하여 본문을 작성·저장하고 목적지 보충 후 한 번 전송한다."""
    from src.core.request_contracts import UnresolvedBody

    save, slack, delivered, output = delivery_tools
    contract = _contract(UnresolvedBody(question="어느 API인가요?"), save="requested", slack="requested",
                         missing_info=(MissingInformation(slot="subject", reason="not_provided", question="어느 API인가요?"),),
                         answer={"content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["request"]}]})
    state = _state(contract)
    actions = make_action_postprocess_node(save, slack, False)

    _prepare(state, DocumentModel())
    state.update(actions(state))
    assert state["runtime"].pending_action.response is None
    assert not state["runtime"].pending_action.body_prepared
    assert not output.exists()

    prepared_contract = RequestContract.model_validate({
        **contract.model_dump(mode="python"), "revision": 2, "relation": "supplement",
        "target_request_id": contract.request_id, "body": ComposeBody(instruction="Slack API의 공통 동작을 설명한다."),
        "missing_info": (),
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": prepared_contract})
    _prepare(state, DocumentModel("Slack API는 앱이 메시지 등 Slack 기능을 이용할 수 있도록 합니다."))
    state.update(actions(state))
    prepared = state["response"].result
    pending = state["runtime"].pending_action
    saved_file = next(output.glob("*.txt"))
    saved_bytes = saved_file.read_bytes()
    saved_time = saved_file.stat().st_mtime_ns
    assert pending.body_prepared
    assert pending.phase == "awaiting_destination"
    assert pending.completed_actions == ("save_text",)
    assert pending.contract.answer == contract.answer
    assert delivered == []

    delivery_contract = RequestContract.model_validate({
        **prepared_contract.model_dump(mode="python"), "revision": 3,
        "body": CopyAnswerBody(source=BoundAnswerReference(ref="pending", response_hash=prepared.content_hash)),
        "slack_destination": {"channel_id": "C123"},
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": delivery_contract})
    _prepare(state, DocumentModel())
    state.update(actions(state))

    assert state["runtime"].pending_action is None
    assert state["response"].result.content == prepared.content
    assert state["response"].request_id == contract.request_id
    assert state["response"].contract_revision == 3
    assert [item["payload"]["text"] for item in delivered] == [export_answer_text(prepared, include_sources=True)]
    assert saved_file.read_bytes() == saved_bytes
    assert saved_file.stat().st_mtime_ns == saved_time


def test_body_only_request_finishes_after_subject_is_supplied(delivery_tools):
    """전달 액션 없는 주제 보충도 원래 코드 금지를 지키는 답변을 완료한 뒤 보류 상태를 정리한다."""
    from src.core.request_contracts import UnresolvedBody

    save, slack, delivered, output = delivery_tools
    contract = _contract(UnresolvedBody(question="어느 API인가요?"),
                         missing_info=(MissingInformation(slot="subject", reason="not_provided", question="어느 API인가요?"),),
                         answer={"content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["request"]}]})
    state = _state(contract)
    actions = make_action_postprocess_node(save, slack, False)
    _prepare(state, DocumentModel())
    state.update(actions(state))
    supplement = RequestContract.model_validate({
        **contract.model_dump(mode="python"), "revision": 2, "relation": "supplement",
        "target_request_id": contract.request_id, "body": ComposeBody(instruction="Slack API를 설명한다."),
        "missing_info": (),
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": supplement})

    _prepare(state, DocumentModel("앱은 Slack API를 통해 Slack 기능을 이용할 수 있습니다."))
    state.update(actions(state))

    assert state["response"].kind == "answer"
    assert state["runtime"].pending_action is None
    assert state["runtime"].request_contract.answer == contract.answer
    assert delivered == []
    assert not output.exists()


def test_failed_transformation_waiting_on_intent_retains_source_without_marking_it_prepared(delivery_tools):
    """실패한 수정의 원본을 준비된 출력으로 승격하지 않고 다음 보충에서 수정된 세 줄만 전달한다."""
    from src.core.contracts.debug import RetryState

    save, slack, delivered, output = delivery_tools
    source = finalize_answer(text_document("원문 하나\n원문 둘\n원문 셋\n원문 넷"), [])
    contract = _contract(TransformAnswerBody(
        source=BoundAnswerReference(ref="pending", response_hash=source.content_hash), instruction="세 줄로 줄인다.",
        evidence_ids=("request",),
    ), save="requested", slack="unresolved", revision=2, relation="supplement", target_request_id="slot-request",
        missing_info=(MissingInformation(slot="slack_intent", reason="unclear", question="Slack에도 보낼까요?"),),
        answer={"format": [{"kind": "line_count", "mode": "required", "value": 3, "evidence_ids": ["request"]}]})
    pending = PendingAction(contract=contract, response=source, body_prepared=False, phase="awaiting_body")
    state = _state(contract, pending=pending)
    state["retry"] = RetryState(max_retries=0)
    actions = make_action_postprocess_node(save, slack, False)

    _prepare(state, DocumentModel(export_answer_text(source)))
    state.update(actions(state))

    assert state["response"].kind == "failure"
    assert state["runtime"].pending_action.response.content_hash == source.content_hash
    assert state["runtime"].pending_action.phase == "awaiting_input"
    assert not state["runtime"].pending_action.body_prepared
    assert delivered == []
    assert not output.exists()

    ready_contract = RequestContract.model_validate({
        **contract.model_dump(mode="python"), "revision": 3, "missing_info": (),
        "slack_destination": {"channel_id": "C123"},
        "actions": {"save_text": contract.actions.save_text,
                    "slack_notify": ActionRequest(intent="requested", evidence_ids=("request",))},
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": ready_contract})
    _prepare(state, DocumentModel("요약 하나\n요약 둘\n요약 셋"))
    state.update(actions(state))

    assert state["runtime"].pending_action is None
    assert [item["payload"]["text"] for item in delivered] == ["요약 하나\n요약 둘\n요약 셋"]
    assert next(output.glob("*.txt")).read_bytes().decode("utf-8-sig") == "요약 하나\n요약 둘\n요약 셋"


@pytest.mark.parametrize("relation,blocked", [("supplement", False), ("correction", False), ("cancel", False), ("supplement", True)])
def test_unmatched_or_unresolved_pending_target_cannot_replace_the_active_request(delivery_tools, relation, blocked):
    """보충·정정·취소 대상이 틀리거나 미해결이면 현재 보류 요청의 본문과 완료 상태를 보존한다."""
    save, slack, delivered, output = delivery_tools
    original = _contract(ComposeBody(instruction="기존 본문"), slack="requested", request_id="original-request")
    pending = PendingAction(contract=original, response=finalize_answer(text_document("보존할 본문"), []),
                            body_prepared=True, completed_actions=("save_text",))
    candidate = _contract(ComposeBody(), relation=relation,
                          request_id="unmatched-request" if not blocked else original.request_id,
                          target_request_id="unknown-request" if not blocked else original.request_id,
                          slack="not_requested" if relation == "cancel" else "requested",
                          missing_info=((MissingInformation(slot="pending_request", reason="unclear", question="어떤 요청인가요?"),)
                                        if blocked else ()))
    state = _state(candidate, pending=pending)
    state["response"] = ResponseState(result=finalize_answer(text_document("어떤 요청인가요?"), []), kind="clarification",
                                      request_id=candidate.request_id, contract_revision=candidate.revision)

    state.update(make_action_postprocess_node(save, slack, False)(state))

    assert state["runtime"].pending_action == pending
    assert delivered == []
    assert not output.exists()


@pytest.mark.parametrize("delivery_tools", [[{"ok": False, "error": "internal_error"},
                                            {"ok": True, "channel": "C123", "ts": "2.0"}]], indirect=True)
def test_delivery_error_preserves_ready_body_and_completed_save_for_explicit_retry(delivery_tools):
    """저장 성공 후 전송 오류는 준비 본문과 저장 완료를 보존하고 명시적 재시도에서 전송만 수행한다."""
    save, slack, delivered, output = delivery_tools
    contract = _contract(ComposeBody(instruction="본문 작성"), save="requested", slack="requested",
                         slack_destination={"channel_id": "C123"})
    state = _state(contract)
    actions = make_action_postprocess_node(save, slack, False)
    _prepare(state, DocumentModel("검증된 전달 본문"))

    state.update(actions(state))

    pending = state["runtime"].pending_action
    assert pending is not None
    assert pending.phase == "awaiting_delivery"
    assert pending.body_prepared
    assert pending.completed_actions == ("save_text",)
    assert len(delivered) == 1
    assert [(receipt.kind, receipt.status) for receipt in state["response"].result.actions] == [
        ("save_text", "success"), ("slack_notify", "error"),
    ]
    saved_file = next(output.glob("*.txt"))
    saved_bytes = saved_file.read_bytes()
    saved_time = saved_file.stat().st_mtime_ns

    retry_contract = RequestContract.model_validate({
        **contract.model_dump(mode="python"), "revision": 2, "relation": "supplement", "target_request_id": contract.request_id,
        "body": CopyAnswerBody(source=BoundAnswerReference(ref="pending", response_hash=pending.response.content_hash)),
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": retry_contract})
    _prepare(state, DocumentModel())
    state.update(actions(state))

    assert state["runtime"].pending_action is None
    assert [item["payload"]["text"] for item in delivered] == ["검증된 전달 본문", "검증된 전달 본문"]
    assert saved_file.read_bytes() == saved_bytes
    assert saved_file.stat().st_mtime_ns == saved_time
    assert [(receipt.kind, receipt.status) for receipt in state["response"].result.actions] == [("slack_notify", "success")]


@pytest.mark.parametrize("kind,revision", [("failure", 2), ("draft", 2), ("answer", 1)])
def test_body_only_request_stays_pending_until_this_revision_has_a_checked_answer(delivery_tools, kind, revision):
    """액션 없는 보충 요청도 실패·초안·이전 revision 본문을 완료로 처리하여 기존 조건을 지우지 않는다."""
    save, slack, delivered, output = delivery_tools
    contract = _contract(ComposeBody(instruction="코드 없이 Slack API를 설명한다."), revision=2, relation="supplement",
                         target_request_id="slot-request",
                         answer={"content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["request"]}]})
    pending = PendingAction(contract=contract, response=None, phase="awaiting_body", body_prepared=False)
    state = _state(contract, pending=pending)
    state["response"] = ResponseState(result=finalize_answer(text_document("미완성 안내"), []), kind=kind,
                                      request_id=contract.request_id, contract_revision=revision)
    actions = make_action_postprocess_node(save, slack, False)

    state.update(actions(state))

    retained = state["runtime"].pending_action
    assert retained is not None
    assert retained.contract.answer == contract.answer
    assert retained.phase == "awaiting_body"
    assert retained.response is None
    assert not retained.body_prepared
    assert delivered == []
    assert not output.exists()

    _prepare(state, DocumentModel("Slack API를 통해 앱에서 Slack 기능을 이용할 수 있습니다."))
    state.update(actions(state))

    assert state["runtime"].pending_action is None
    assert state["response"].kind == "answer"


@pytest.mark.parametrize("blocked_action,slot", [("save_text", "save_intent"), ("slack_notify", "slack_intent")])
def test_requested_action_with_an_unresolved_slot_keeps_the_prepared_body_and_asks_about_that_action(
    delivery_tools, blocked_action, slot,
):
    """requested 사실이 있어도 미해결 intent slot을 질문으로 보존하며 다른 준비된 행동만 실행한다."""
    save, slack, delivered, output = delivery_tools
    question = "파일로 저장할까요?" if blocked_action == "save_text" else "Slack으로 전송할까요?"
    contract = _contract(ComposeBody(instruction="본문을 설명한다."), save="requested", slack="requested",
                         slack_destination={"channel_id": "C123"},
                         missing_info=(MissingInformation(slot=slot, reason="unclear", question=question),))
    state = _state(contract)

    _prepare(state, DocumentModel("준비된 설명 본문"))
    state.update(make_action_postprocess_node(save, slack, False)(state))

    pending = state["runtime"].pending_action
    assert pending is not None
    assert pending.phase == "awaiting_input"
    assert pending.body_prepared
    assert export_answer_text(pending.response) == "준비된 설명 본문"
    assert getattr(pending.contract.actions, blocked_action).intent == "requested"
    acknowledgement = next(receipt for receipt in state["response"].result.actions if receipt.kind == blocked_action)
    assert acknowledgement.status == "skipped"
    assert acknowledgement.message == question
    if blocked_action == "save_text":
        assert not output.exists()
        assert [item["payload"]["text"] for item in delivered] == ["준비된 설명 본문"]
        assert pending.completed_actions == ("slack_notify",)
    else:
        assert next(output.glob("*.txt")).read_bytes().decode("utf-8-sig") == "준비된 설명 본문"
        assert delivered == []
        assert pending.completed_actions == ("save_text",)
