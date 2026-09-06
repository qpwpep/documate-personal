from pathlib import Path

from langchain_core.messages import HumanMessage

from src.core.answer_schema import (
    AnswerDocument,
    CodeBlock,
    ContentUnit,
    ResponseIssue,
    export_answer_text,
    finalize_answer,
)
from src.core.contracts import GraphState, PlannerState, ResponseState, RuntimeState
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence
from src.runtime.nodes.actions import make_action_postprocess_node


def test_save_and_slack_retain_source_selection_and_limitations_without_mutating_answer(tmp_path: Path):
    """저장·전송은 원문 위치와 예제의 제한을 보존하고 확정된 답변을 바꾸지 않는다."""
    original_file = tmp_path / "settings.py"
    original_file.write_text("# Config\nRETRIES = 3\n", encoding="utf-8")
    snapshot = build_snapshot(
        source_uri=original_file.as_uri(), title=original_file.name,
        media_type="text/x-python", source_type="upload",
        content=original_file.read_bytes(), parser="python", parser_version="1",
        quality_issues=["이 파일에는 설정의 예외 조건이 없습니다."],
    )
    element = DocumentElement(
        element_id="settings-code", kind="code", text="RETRIES = 3\n", language="python",
        anchors=[SourceAnchor(kind="code", start=9, end=21, line_start=2, line_end=2, precision="exact")],
    )
    evidence = build_evidence(snapshot=snapshot, element=element, start=10, end=11)
    code = "def retries():\n    return 3\n"
    answer = finalize_answer(
        AnswerDocument(blocks=[CodeBlock(language="python", content=ContentUnit(text=code, basis="example", refs=[evidence.id]))]),
        [evidence], retrieval_required=True,
        issues=[ResponseIssue(code="answer_incomplete", message="요청한 전체 조건을 확인하지 못한 예시입니다.")],
    )
    original_content = answer.content.model_dump(mode="json")
    original_hash = answer.content_hash
    request = "결과를 txt로 저장하고 슬랙으로 보내줘"
    state: GraphState = {
        "runtime": RuntimeState(user_input=request),
        "planner": PlannerState(),
        "response": ResponseState(result=answer, evidence_packet=[evidence]),
        "messages": [HumanMessage(content=request)],
    }
    destination = tmp_path / "answer.txt"
    slack_messages = []

    def save_text(content: str, filename_prefix: str):
        destination.write_text(content, encoding="utf-8")
        return {"status": "success", "file_path": str(destination)}

    def slack_notify(**message):
        slack_messages.append(message)
        return {"status": "ok", "channel_id": "C123"}

    updates = make_action_postprocess_node(
        save_text, slack_notify, False, has_default_slack_destination=True,
    )(state)

    saved = destination.read_text(encoding="utf-8")
    assert [message["text"] for message in slack_messages] == [saved]
    assert saved == export_answer_text(answer, include_sources=True)
    assert snapshot.snapshot_id in saved
    assert original_file.as_uri() in saved
    assert "settings-code" in saved
    assert "range [10, 11)" in saved
    assert "lines 2-2" in saved
    assert "코드 예시 · 실행 확인 안 됨" in saved
    assert "요청한 전체 조건을 확인하지 못한 예시입니다." in saved
    assert "이 파일에는 설정의 예외 조건이 없습니다." in saved
    assert "의미적 지지는 별도로 검증하지 않았습니다" in saved
    assert f"```python\n{code}```" in saved
    final = updates["response"].result
    assert final.content.model_dump(mode="json") == original_content
    assert answer.content.model_dump(mode="json") == original_content
    assert final.content_hash == answer.content_hash == original_hash
    assert [(receipt.kind, receipt.status) for receipt in final.actions] == [
        ("save_text", "success"), ("slack_notify", "success"),
    ]
