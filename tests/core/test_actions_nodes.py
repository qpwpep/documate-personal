from pathlib import Path

from langchain_core.messages import HumanMessage, ToolMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RuntimeState
from src.runtime.nodes.actions import make_action_postprocess_node, should_short_circuit_action_only


def _state(text: str, request: str = "결과를 txt로 저장해줘") -> GraphState:
    return {
        "runtime": RuntimeState(user_input=request),
        "planner": PlannerState(),
        "response": ResponseState(result=finalize_answer(text_document(text), [])),
        "messages": [HumanMessage(content=request)],
    }


def test_save_exports_the_document_and_adds_a_separate_receipt(tmp_path: Path):
    """저장 결과를 본문에 추가하지 않고 실제 저장한 내용과 별도 영수증을 반환한다."""
    destination = tmp_path / "answer.txt"
    state = _state("실제 답변 본문")
    original = state["response"].result.model_dump()

    def save_text(content: str, filename_prefix: str):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    updates = make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)
    result = updates["response"].result

    assert destination.read_text(encoding="utf-8") == export_answer_text(state["response"].result)
    assert result.content.model_dump() == original["content"]
    assert result.content_hash == original["content_hash"]
    assert result.actions[0].model_dump(exclude_none=True) == {
        "kind": "save_text", "status": "success", "file_path": str(destination),
    }
    assert all(isinstance(message, ToolMessage) for message in updates["messages"])


def test_slack_exports_current_document_without_appending_receipts():
    """전송 대상을 명시한 요청은 현재 문서 본문 그대로 전달한다."""
    delivered = []
    state = _state("현재 답변", "결과를 슬랙으로 보내줘")

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
    state = _state("본문", "슬랙으로 보내줘")

    def notify(**kwargs):
        delivered.append(kwargs)
        return {"status": "ok"}

    updates = make_action_postprocess_node(lambda **kwargs: {}, notify, False)(state)

    assert delivered == []
    assert updates["response"].result.actions[0].status == "skipped"
    assert updates["response"].result.content == state["response"].result.content


def test_guided_followup_prevents_action_delivery(tmp_path: Path):
    """계획 단계의 후속 질문이 있으면 저장이나 전송을 실행하지 않는다."""
    state = _state("파일을 올려 주세요.")
    state["planner"] = PlannerState(guided_followup="파일을 올려 주세요.")
    destination = tmp_path / "answer.txt"

    def save_text(content, **kwargs):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    updates = make_action_postprocess_node(save_text, lambda **kwargs: {}, False)(state)

    assert updates == {}
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
    assert updates["response"].result.actions[0].status == "skipped"
    assert updates["response"].result.content == state["response"].result.content


def test_short_circuit_requires_a_slack_destination_only():
    """저장은 바로 진행하고 목적지가 없는 Slack 요청만 후속 질문을 요구한다."""
    assert not should_short_circuit_action_only(
        user_input="방금 답변을 txt로 저장해줘", messages=[], slack_target_available=False,
    )
    assert should_short_circuit_action_only(
        user_input="send this to slack", messages=[], slack_target_available=False,
    )
