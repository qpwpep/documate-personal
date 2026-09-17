from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RuntimeState
from src.core.contracts.graph_state import PendingAction
from src.core.request_contracts import RequestContract
from src.runtime.nodes.actions import make_action_postprocess_node
from src.infra.tools.save_text import build_save_text_tool


def _contract(*, save="not_requested", slack="not_requested", **updates):
    missing_info = [
        {"slot": slot, "reason": "unclear", "question": "실행 여부를 알려 주세요."}
        for slot, intent in (("save_intent", save), ("slack_intent", slack)) if intent == "unresolved"
    ]
    return RequestContract.model_validate({
        "actions": {
            "save_text": {"intent": save, "evidence_ids": ["request"]},
            "slack_notify": {"intent": slack, "evidence_ids": ["request"]},
        },
        "evidence": [{"id": "request", "turn_id": "current", "quote": "사용자 요청", "scope": "current_request", "interpretation": "instruction"}],
        "missing_info": missing_info,
        **updates,
    })


def _state(text: str, request: str = "결과를 txt로 저장해줘", *, contract=None) -> GraphState:
    contract = contract or _contract(save="requested")
    return {
        "runtime": RuntimeState(user_input=request, request_contract=contract),
        "planner": PlannerState(),
        "response": ResponseState(result=finalize_answer(text_document(text), []), kind="answer",
                                  request_id=contract.request_id, contract_revision=contract.revision),
        "messages": [HumanMessage(content=request)],
    }


def test_save_exports_the_document_and_adds_a_separate_receipt(tmp_path: Path, monkeypatch):
    """저장 결과를 본문에 추가하지 않고 실제 저장한 내용과 별도 영수증을 반환한다."""
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    state = _state("실제 답변 본문")
    original = state["response"].result.model_dump()

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state)
    result = updates["response"].result

    destination = Path(result.actions[0].file_path)
    assert destination.read_text(encoding="utf-8-sig") == export_answer_text(state["response"].result)
    assert result.content.model_dump() == original["content"]
    assert result.content_hash == original["content_hash"]
    assert result.actions[0].status == "success"
    assert result.actions[0].verification == "verified"
    assert result.actions[0].operation.answer_hash == original["content_hash"]
    assert all(isinstance(message, ToolMessage) for message in updates["messages"])


def test_slack_exports_current_document_without_appending_receipts():
    """전송 대상을 명시한 요청은 현재 문서 본문 그대로 전달한다."""
    delivered = []
    state = _state("현재 답변", "결과를 슬랙으로 보내줘", contract=_contract(slack="requested"))

    def notify(**kwargs):
        delivered.append(kwargs)
        return {"status": "ok", "channel_id": "C123"}

    updates = make_action_postprocess_node(
        lambda **kwargs: {}, notify, False, has_default_slack_destination=True,
    )(state)

    assert [item["text"] for item in delivered] == [export_answer_text(state["response"].result)]
    assert updates["response"].result.content == state["response"].result.content
    assert updates["response"].result.actions[0].status == "success"
    assert updates["response"].result.actions[0].target == "C123"


def test_failed_action_keeps_the_original_document(tmp_path: Path):
    """저장 실패가 원래 답변을 실패 문구로 덮어쓰지 않는다."""
    state = _state("보존할 답변")

    def save_text(**kwargs):
        raise OSError("disk unavailable")

    updates = make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert updates["response"].result.content == state["response"].result.content
    assert updates["response"].result.actions[0].status == "error"
    assert updates["response"].result.actions[0].error == "disk unavailable"
    assert list(tmp_path.iterdir()) == []


def test_missing_destination_records_a_skipped_receipt_without_delivery():
    """목적지가 없으면 전송을 실행하지 않고 보류 상태를 기록한다."""
    delivered = []
    state = _state("본문", "슬랙으로 보내줘", contract=_contract(slack="requested"))

    def notify(**kwargs):
        delivered.append(kwargs)
        return {"status": "ok"}

    updates = make_action_postprocess_node(lambda **kwargs: {}, notify, False)(state)

    assert delivered == []
    assert updates["response"].result.actions[0].status == "skipped"
    assert updates["response"].result.content == state["response"].result.content
    assert updates["runtime"].pending_action.response.content == state["response"].result.content
    assert "알려주세요" in updates["response"].result.actions[0].message


def test_guided_followup_prevents_action_delivery(tmp_path: Path):
    """계획 단계의 후속 질문이 있으면 저장이나 전송을 실행하지 않는다."""
    state = _state("파일을 올려 주세요.")
    state["planner"] = PlannerState(guided_followup="파일을 올려 주세요.")
    state["response"] = state["response"].model_copy(update={"kind": "clarification"})
    destination = tmp_path / "answer.txt"

    def save_text(content, **kwargs):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    updates = make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert "response" not in updates
    assert not destination.exists()


def test_empty_document_does_not_create_an_artificial_delivery_body(tmp_path: Path):
    """본문이 없으면 형식적인 공유 문구를 만들어 저장하지 않는다."""
    state = _state("")
    destination = tmp_path / "answer.txt"

    def save_text(content, **kwargs):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    updates = make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert not destination.exists()
    assert updates["runtime"].pending_action.phase == "awaiting_body"
    assert updates["runtime"].pending_action.response is None


@pytest.mark.parametrize("intent", ["forbidden", "not_requested", "unresolved"])
def test_keywords_cannot_execute_actions_without_requested_contract(tmp_path: Path, intent):
    """원문의 저장·전송 키워드는 미요청·금지·불명확 계약을 실행으로 바꾸지 않는다."""
    destination = tmp_path / "answer.txt"
    state = _state("본문", "save this and send to Slack", contract=_contract(save=intent, slack=intent))

    def unexpected_action(**kwargs):
        destination.write_text(str(kwargs), encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    updates = make_action_postprocess_node(unexpected_action, unexpected_action, False, True)(state)

    assert not destination.exists()
    if intent == "unresolved":
        assert all(receipt.status == "skipped" for receipt in updates["response"].result.actions)
    else:
        assert "response" not in updates


@pytest.mark.parametrize("status", ["invalid", "unresolved"])
def test_contract_errors_cannot_recover_execution_from_keywords(tmp_path: Path, status):
    """해석에 실패한 요청은 액션 키워드가 있어도 파일이나 메시지를 만들지 않는다."""
    destination = tmp_path / "answer.txt"
    details = ({"failure": "invalid_contract"} if status == "invalid" else {
        "body": {"kind": "unresolved", "question": "본문을 알려 주세요."},
        "missing_info": [{"slot": "subject", "reason": "not_provided", "question": "본문을 알려 주세요."}],
    })
    state = _state("본문", contract=_contract(save="requested", slack="requested", **details))

    def unexpected_action(**kwargs):
        destination.write_text(str(kwargs), encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    make_action_postprocess_node(unexpected_action, unexpected_action, False, True)(state)

    assert not destination.exists()


@pytest.mark.parametrize("kind", ["draft", "clarification", "failure"])
def test_only_final_answer_can_be_delivered(tmp_path: Path, kind):
    """초안·보충 질문·실패 안내는 저장 대상 본문으로 취급하지 않는다."""
    destination = tmp_path / "answer.txt"
    state = _state("본문")
    state["response"] = state["response"].model_copy(update={"kind": kind})

    def save_text(content, **kwargs):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert not destination.exists()


def test_absent_contract_does_not_authorize_an_action(tmp_path: Path):
    """계약이 없는 요청은 저장 키워드나 설정된 Slack 대상만으로 실행하지 않는다."""
    state = _state("본문")
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": None})
    destination = tmp_path / "answer.txt"

    def unexpected_action(**kwargs):
        destination.write_text(str(kwargs), encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    assert make_action_postprocess_node(unexpected_action, unexpected_action, False, True)(state) == {}
    assert not destination.exists()


@pytest.mark.parametrize("attempt", [1, 2, 3])
def test_forbidden_output_is_never_saved_on_any_synthesis_attempt(tmp_path: Path, attempt):
    """정상 생성과 재합성 횟수에 관계없이 금지된 코드가 든 본문은 저장하지 않는다."""
    contract = _contract(save="requested", answer={
        "content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["request"]}],
    })
    state = _state("```python\nprint('forbidden')\n```", contract=contract)
    state["response"] = state["response"].model_copy(update={"synthesis_attempt": attempt})
    destination = tmp_path / "answer.txt"

    def save_text(content, **kwargs):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert not destination.exists()


def test_mismatched_previous_answer_cannot_be_delivered(tmp_path: Path):
    """고정한 이전 답변의 hash와 다른 본문은 재사용 요청으로 전달하지 않는다."""
    original = finalize_answer(text_document("원래 답변"), [])
    state = _state("다른 답변", contract=_contract(save="requested", body={
        "kind": "copy_answer", "source": {"ref": "previous", "response_hash": original.content_hash},
    }))
    state["runtime"] = state["runtime"].model_copy(update={"previous_response": original})
    destination = tmp_path / "answer.txt"

    def save_text(content, **kwargs):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert not destination.exists()


@pytest.mark.parametrize("kind", ["clarification", "failure"])
def test_forbidding_pending_actions_preserves_an_unfinished_body_request(kind):
    """모든 행동이 금지되어도 미완성 본문 요청은 금지 조건을 유지한 채 보존한다."""
    original = finalize_answer(text_document("원래 답변"), [])
    pending = PendingAction(contract=_contract(slack="requested"), response=original, body_prepared=True)
    contract = _contract(slack="forbidden", relation="supplement", request_id=pending.contract.request_id, target_request_id=pending.contract.request_id, body={
        "kind": "copy_answer", "source": {"ref": "pending", "response_hash": original.content_hash},
    })
    state = _state("새 조건을 충족하지 못했습니다.", contract=contract)
    state["runtime"] = state["runtime"].model_copy(update={"pending_action": pending})
    state["response"] = state["response"].model_copy(update={"kind": kind})

    updates = make_action_postprocess_node(lambda **kwargs: {}, lambda **kwargs: {}, False)(state)

    assert updates["runtime"].pending_action.contract.actions.slack_notify.intent == "forbidden"
    assert updates["runtime"].pending_action.phase == "awaiting_body"
    assert not updates["runtime"].pending_action.body_prepared
    assert "response" not in updates
