from pathlib import Path

import pytest
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
from src.core.contracts import GraphState, PlannerState, ResponseState, RetryState, RuntimeState
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import make_decisions_client
from tests.core.test_pending_request_delivery import _contract, delivery_tools


def test_save_and_slack_retain_source_selection_and_limitations_without_mutating_answer(tmp_path: Path, delivery_tools):
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
    contract = _contract(save="requested", slack="requested", slack_recipient={
        "state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["request"],
    })
    state: GraphState = {
        "runtime": RuntimeState(user_input=request, request_contract=contract),
        "planner": PlannerState(),
        "response": ResponseState(result=answer, evidence_packet=[evidence], kind="answer",
                                  request_id=contract.request_id, contract_revision=contract.revision),
        "messages": [HumanMessage(content=request)],
    }
    save, slack, slack_messages, _output = delivery_tools

    updates = make_action_postprocess_node(
        save, slack, False,
    )(state)

    saved = Path(updates["response"].result.actions[0].file_path).read_text(encoding="utf-8-sig")
    assert [message["payload"]["text"] for message in slack_messages if message["path"] == "/chat.postMessage"] == [saved]
    assert snapshot.snapshot_id in saved
    assert original_file.as_uri() in saved
    assert "settings-code" in saved
    assert "range [10, 11)" in saved
    assert "lines 2-2" in saved
    assert "예시 · 코드 블록 · 실행 확인 안 됨" in saved
    assert "요청한 전체 조건을 확인하지 못한 예시입니다." in saved
    assert "이 파일에는 설정의 예외 조건이 없습니다." in saved
    assert "출처 연결은 의미적 정확성을 보증하지 않습니다." in saved
    assert f"```python\n{code}```" in saved
    final = updates["response"].result
    assert final.content.model_dump(mode="json") == original_content
    assert answer.content.model_dump(mode="json") == original_content
    assert final.content_hash == answer.content_hash == original_hash
    assert [(receipt.kind, receipt.status) for receipt in final.actions] == [
        ("save_text", "success"), ("slack_notify", "success"),
    ]


@pytest.mark.parametrize(("basis", "label"), [
    ("interaction", "코드 블록 · 실행 확인 안 됨"),
    ("source", "근거 기반 내용 · 코드 블록 · 실행 확인 안 됨"),
    ("inference", "해석 · 코드 블록 · 실행 확인 안 됨"),
    ("example", "예시 · 코드 블록 · 실행 확인 안 됨"),
    ("excerpt", "원문 코드 발췌"),
])
def test_export_log_block_preserves_format_and_basis_without_claiming_program_code(basis, label):
    logs = "INFO worker started\nWARN worker retry scheduled"
    snapshot = build_snapshot(source_uri="upload:///worker.log", title="worker.log",
                              media_type="text/plain", source_type="upload", content=logs,
                              parser="text", parser_version="1")
    evidence = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="logs", kind="paragraph", text=logs))
    answer = finalize_answer(AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text=logs, basis=basis,
                                             refs=[evidence.id] if basis in {"source", "inference", "excerpt"} else []),
    )]), [evidence])
    before = answer.model_dump(mode="json")

    exported = export_answer_text(answer, include_sources=True)

    assert exported.startswith(label + "\n\n```text\n" + logs + "\n```")
    assert "작성한 코드" not in exported and "코드 예시" not in exported
    assert answer.model_dump(mode="json") == before


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
        "content": [{"kind": "code", "mode": "forbidden", "evidence_ids": ["request"]}],
    })
    return {
        "runtime": RuntimeState(request_contract=contract),
        "planner": PlannerState(),
        "retry": RetryState(max_retries=0, max_content_repairs=0),
        "response": ResponseState(result=answer, kind="answer", request_id=contract.request_id,
                                  contract_revision=contract.revision,
                                  evidence_packet=[citation.evidence for citation in answer.citations]),
    }


def test_forbidden_source_code_wrapped_as_paragraph_is_not_saved(tmp_path: Path, monkeypatch):
    """코드 원문을 일반 문단으로 감싸도 코드 금지 조건을 우회하여 저장할 수 없다."""
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    answer = _code_source_answer("def retry():\n    return 3\n", basis="excerpt")

    state = _no_code_save_state(answer)
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))
    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state)

    assert state["response"].kind == "failure"
    assert not output.exists()
    assert "response" not in updates
