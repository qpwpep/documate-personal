from pathlib import Path

from langchain_core.messages import HumanMessage

from src.core.answer_schema import (
    AnswerDocument,
    CodeBlock,
    ContentUnit,
    ResponseIssue,
    export_answer_text,
    finalize_answer,
    text_document,
)
from src.core.contracts import GraphState, PlannerState, ResponseState, RuntimeState
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from .test_actions_nodes import _contract


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
    contract = _contract(save="requested", slack="requested")
    state: GraphState = {
        "runtime": RuntimeState(user_input=request, request_contract=contract),
        "planner": PlannerState(),
        "response": ResponseState(result=answer, evidence_packet=[evidence], kind="answer",
                                  request_id=contract.request_id, contract_revision=contract.revision),
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


def _code_source_answer(text: str, *, basis: str):
    code = "def retry():\n    return 3\n"
    snapshot = build_snapshot(
        source_uri="https://docs.example.com/retry.py", title="Retry configuration",
        media_type="text/x-python", source_type="official", content=code,
        parser="python", parser_version="1",
    )
    evidence = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(element_id="retry-code", kind="code", text=code, language="python"),
    )
    return finalize_answer(text_document(text, basis=basis, refs=[evidence.id]), [evidence])


def _no_code_save_state(answer):
    contract = _contract(save="requested", answer={
        "content": [{"kind": "code_example", "mode": "forbidden", "evidence_ids": ["request"]}],
    })
    return {
        "runtime": RuntimeState(request_contract=contract),
        "planner": PlannerState(),
        # Delivery must use the checked citations even if transient retrieval
        # state is unavailable after a previous-answer reuse.
        "response": ResponseState(result=answer, kind="answer", request_id=contract.request_id, contract_revision=contract.revision),
    }


def test_forbidden_source_code_wrapped_as_paragraph_is_not_saved(tmp_path: Path, monkeypatch):
    """코드 원문을 일반 문단으로 감싸도 코드 금지 조건을 우회하여 저장할 수 없다."""
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    answer = _code_source_answer("def retry():\n    return 3\n", basis="excerpt")

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(
        _no_code_save_state(answer)
    )

    assert not output.exists()
    assert "response" not in updates


def test_explanation_citing_code_is_saved_when_code_is_forbidden(tmp_path: Path, monkeypatch):
    """코드 출처를 근거로 설명한 일반 문장은 코드 자체로 오인하여 저장을 막지 않는다."""
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    answer = _code_source_answer("이 함수는 재시도 횟수로 세 번을 반환합니다.", basis="source")

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(
        _no_code_save_state(answer)
    )

    files = list(output.glob("*.txt"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8-sig") == export_answer_text(answer, include_sources=True)
    assert updates["response"].result.content == answer.content
    assert updates["response"].result.citations == answer.citations
    assert updates["response"].result.actions[0].status == "success"
