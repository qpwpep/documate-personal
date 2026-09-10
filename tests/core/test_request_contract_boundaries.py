from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import PendingAction, ResponseState
from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update
from src.core.contracts.boundary.runtime import parse_runtime_state
from src.core.request_contracts import RequestContract


def test_runtime_roundtrip_keeps_the_contract_and_pending_body_revision():
    """상태 경계 왕복에서도 보류된 요청과 전달 본문의 원래 버전을 유지한다."""
    contract = RequestContract()
    pending = PendingAction(contract=contract, response=finalize_answer(text_document("전달할 본문"), []))
    state = build_graph_state_input(user_input="C123", request_contract=contract, pending_action=pending)
    runtime = parse_runtime_state(state["runtime"].model_dump())
    assert runtime.request_contract == contract
    assert runtime.pending_action == pending


def test_malformed_request_contract_is_invalid_without_keyword_recovery():
    """파싱 실패 뒤 원문의 저장 단어로 액션을 복구하지 않는다."""
    runtime = parse_runtime_state({"user_input": "저장해줘", "request_contract": {"actions": {"save_text": True}}})
    assert runtime.request_contract.status == "invalid"
    assert runtime.request_contract.failure == "contract_invalid"
    assert runtime.request_contract.actions.save_text.intent == "not_requested"
    assert runtime.request_contract.actions.slack_notify.intent == "not_requested"


def test_corrupt_pending_body_is_discarded_and_cannot_be_delivered():
    """원문 hash가 맞지 않는 보류 본문은 다른 답변으로 대체하지 않고 실행을 차단한다."""
    runtime = parse_runtime_state({"user_input": "C123", "request_contract": RequestContract().model_dump(),
                                   "pending_action": {"contract": RequestContract().model_dump(), "response": {"content_hash": "wrong"}}})
    assert runtime.pending_action is None
    assert runtime.request_contract.status == "invalid"


def test_response_kind_survives_graph_normalization():
    """안내문과 실행 가능한 답변의 구분을 상태 정규화에서도 유지한다."""
    response = ResponseState(result=finalize_answer(text_document("대상이 필요합니다."), []), kind="clarification")
    assert normalize_graph_update({"response": response.model_dump()})["response"] == response
