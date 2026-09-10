import pytest

from src.core.answer_schema import AnswerResponse, export_answer_text, finalize_answer, text_document
from src.core.contracts.graph_state import PendingAction
from src.core.request_contracts import AcknowledgeBody, ComposeBody, RequestContract
from tests.core.test_pending_action_delivery import _ConsumerGraph, _manager, delivery_tools
from tests.core.test_request_slots_delivery import _contract


@pytest.mark.parametrize("save_intent,slack_intent,confirmation", [
    ("forbidden", "not_requested", "알겠습니다. 파일로 저장하지 않겠습니다."),
    ("not_requested", "forbidden", "알겠습니다. Slack으로 전송하지 않겠습니다."),
    ("forbidden", "forbidden", "알겠습니다. 파일로 저장하거나 Slack으로 전송하지 않겠습니다."),
])
def test_prohibition_acknowledges_without_a_new_deliverable_or_pending_request(
    delivery_tools, save_intent, slack_intent, confirmation,
):
    """순수 금지는 모델 호출·보충 질문 없이 확인하고 이전 본문을 보존하며 어떤 전달도 수행하지 않는다."""
    save, slack, delivered, output = delivery_tools
    previous = finalize_answer(text_document("보존할 이전 답변"), [])
    contract = _contract(AcknowledgeBody(), save=save_intent, slack=slack_intent)
    manager = _manager(_ConsumerGraph([contract], [], save, slack))
    manager._ensure_session().previous_response = previous

    result = manager.run_agent_flow("저장하거나 전송하지 마")

    answer = AnswerResponse.model_validate(result["response"])
    assert export_answer_text(answer) == confirmation
    assert answer.actions == []
    assert manager._ensure_session().previous_response == previous
    assert manager._ensure_session().pending_action is None
    assert delivered == []
    assert not output.exists()


@pytest.mark.parametrize("relation", ["new", "correction"])
def test_untargeted_acknowledgement_preserves_the_existing_pending_body(delivery_tools, relation):
    """대상을 지정하지 않은 단순 확인은 기존 보류 요청의 원본문과 완료 정보를 바꾸지 않는다."""
    save, slack, delivered, output = delivery_tools
    previous = finalize_answer(text_document("원래 전달할 본문"), [])
    original = _contract(ComposeBody(instruction="기존 본문"), slack="requested", request_id="original-request")
    pending = PendingAction(contract=original, response=previous, body_prepared=True, completed_actions=("save_text",))
    acknowledgement = _contract(AcknowledgeBody(), request_id="acknowledgement", relation=relation)
    manager = _manager(_ConsumerGraph([acknowledgement], [], save, slack))
    manager._ensure_session().previous_response = previous
    manager._ensure_session().pending_action = pending

    result = manager.run_agent_flow("알겠어")

    assert export_answer_text(AnswerResponse.model_validate(result["response"])) == "알겠습니다."
    assert manager._ensure_session().pending_action == pending
    assert manager._ensure_session().previous_response == previous
    assert delivered == []
    assert not output.exists()


def test_targeted_prohibition_acknowledges_and_closes_the_finished_delivery_request(delivery_tools):
    """준비된 본문의 남은 전달을 금지하면 확인만 반환하며 이후 목적지 언급이 전송을 되살리지 않는다."""
    save, slack, delivered, output = delivery_tools
    original = _contract(ComposeBody(instruction="원래 본문"), slack="requested", request_id="original-request")

    def forbid(runtime):
        pending = runtime.pending_action
        return RequestContract.model_validate({
            **pending.contract.model_dump(mode="python"), "revision": 2, "relation": "correction",
            "target_request_id": pending.contract.request_id, "body": AcknowledgeBody(evidence_ids=("ban",)),
            "actions": {"slack_notify": {"intent": "forbidden", "evidence_ids": ("ban",)}},
            "evidence": (*pending.contract.evidence, {
                "id": "ban", "turn_id": "correction", "quote": "그 전송은 하지 마", "scope": "actions.slack_notify", "interpretation": "negation",
            }), "missing_info": (), "body_request": None,
        })

    destination_only = _contract(ComposeBody(instruction="대상만으로 전송 요청을 만들지 않는다."), request_id="independent-request")
    manager = _manager(_ConsumerGraph([original, forbid, destination_only], ["준비된 원래 본문", "전달할 새 요청을 알려 주세요."], save, slack))
    first = AnswerResponse.model_validate(manager.run_agent_flow("이 내용을 Slack으로 보내줘")["response"])
    assert manager._ensure_session().pending_action.body_prepared

    second = AnswerResponse.model_validate(manager.run_agent_flow("그 전송은 하지 마")["response"])

    assert export_answer_text(second) == "알겠습니다. Slack으로 전송하지 않겠습니다."
    assert manager._ensure_session().previous_response.content == first.content
    assert manager._ensure_session().pending_action is None
    manager.run_agent_flow("C123")
    assert manager._ensure_session().pending_action is None
    assert delivered == []
    assert not output.exists()
