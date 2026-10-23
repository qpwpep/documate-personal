"""Rendered answer checks describe body scope independently of delivery results."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.core.answer_export import build_answer_export_operation
from src.core.answer_schema import (
    AnswerDocument,
    AnswerResponse,
    SemanticCheck,
    SemanticValidation,
    finalize_answer,
    export_answer_text,
    semantic_evidence_fingerprint,
    text_document,
)
from src.core.contracts.outcome import TurnResult
from src.runtime.nodes.actions.receipts import build_save_receipt
from tests.core.test_pending_action_delivery import delivery_tools
from tests.core.test_answer_validation import _evidence
from tests.web.test_partial_answer_followup import _partial_manager


def _render(result):
    app = AppTest.from_string("""
import streamlit as st
from src.app.web.streamlit_chat import render_turn_result
from src.core.contracts.outcome import TurnResult
render_turn_result(TurnResult.model_validate(st.session_state["result"]), "http://localhost:8000")
""", default_timeout=10)
    app.session_state["result"] = result.model_dump(mode="json")
    app.run()
    assert not app.exception
    return app


def _checked_answer():
    answer = finalize_answer(text_document("보존할 답변 본문"), [])
    return answer.model_copy(update={"semantic_validation": SemanticValidation(
        content_hash=answer.content_hash,
        evidence_fingerprint=semantic_evidence_fingerprint(answer.citations),
        threshold=0.8,
        checks=[SemanticCheck(name="request_fulfilled", probability=0.95)],
    )})


def test_body_check_does_not_imply_completion_when_file_save_fails():
    answer = _checked_answer()
    operation = build_answer_export_operation(answer, session_id="scope-test")
    receipt = build_save_receipt({"status": "error", "error_code": "write_failed", "error": "disk unavailable"}, operation=operation)
    result = TurnResult(
        status="partial",
        response=answer.model_copy(update={"actions": [receipt]}),
        problem={"code": "save_failed", "stage": "action_postprocess",
                 "message": "파일을 만들지 못했습니다.", "next_action": "retry_later"},
    )

    app = _render(result)

    captions = [item.value for item in app.caption]
    assert any("답변 본문 조건 충족: 통과" in text for text in captions)
    assert any("저장·전송" in text and "별도" in text for text in captions)
    assert any("이번 요청 처리는 완료되지 않았습니다" in text for text in captions)
    assert not any(text.startswith("요청 충족:") for text in captions)
    assert any("파일 저장 실패" in item.value for item in app.error)
    assert any(item.value == "보존할 답변 본문" for item in app.markdown)


def test_saving_a_partial_answer_keeps_body_limits_without_marking_new_request_incomplete(delivery_tools):
    save, slack, deliveries, output = delivery_tools
    manager = _partial_manager(save, slack)
    original_result = manager.run_agent_flow("공식 문서와 첨부 코드를 비교하고 저장해줘")
    original = AnswerResponse.model_validate(original_result["response"])
    original_app = _render(TurnResult.model_validate({key: value for key, value in original_result.items() if key != "debug"}))
    assert any("이번 요청 처리는 완료되지 않았습니다" in item.value for item in original_app.caption)

    saved_result = manager.run_agent_flow("방금 답변을 그대로 저장해줘")
    saved = AnswerResponse.model_validate(saved_result["response"])
    app = _render(TurnResult.model_validate({key: value for key, value in saved_result.items() if key != "debug"}))

    assert saved_result["status"] == "completed"
    assert saved.content == original.content
    assert saved.issues == original.issues
    assert saved.semantic_validation == original.semantic_validation
    assert any("파일 저장 완료" in item.value for item in app.success)
    assert any("파일 다운로드" in item.value for item in app.markdown)
    for issue in original.issues:
        assert any(item.value == f"답변에 기록된 한계: {issue.message}" for item in app.warning)
    captions = [item.value for item in app.caption]
    assert not any("요청 전체는 미완료" in text or "요청 전체를 완료하지 못한" in text
                   or "이번 요청 처리는 완료되지 않았습니다" in text for text in captions)
    assert any("이 답변에 기록된" in text and "한계" in text for text in captions)
    saved_files = list(output.glob("*.txt"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8-sig") == export_answer_text(original, include_sources=True)
    assert deliveries == []


@pytest.mark.parametrize(("basis", "label"), [
    ("interaction", "코드 블록 · 실행 확인 안 됨"),
    ("source", "근거 기반 내용 · 코드 블록 · 실행 확인 안 됨"),
    ("inference", "해석 · 코드 블록 · 실행 확인 안 됨"),
    ("example", "예시 · 코드 블록 · 실행 확인 안 됨"),
    ("excerpt", "원문 발췌"),
])
def test_log_block_is_labeled_as_format_without_claiming_program_code(basis, label):
    logs = "2026-10-23T10:00:01Z INFO worker started\n2026-10-23T10:00:02Z WARN worker retry scheduled"
    evidence = _evidence(logs)
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "code", "language": "text",
        "content": {"text": logs, "basis": basis, "refs": [evidence.id] if basis in {"source", "inference", "excerpt"} else []},
    }]}), [evidence])

    app = _render(TurnResult(status="completed", response=answer))

    captions = [item.value for item in app.caption]
    assert label in captions
    assert "코드 · 실행 확인 안 됨" not in captions
    assert all("코드 예시" not in text and "작성한 코드" not in text for text in captions)
    assert app.code[0].value == logs


def test_inline_format_is_not_mislabeled_as_a_block():
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "code", "inline": True, "content": {"text": "INFO worker started", "basis": "interaction"},
    }]}), [])

    app = _render(TurnResult(status="completed", response=answer))

    assert "인라인 코드 · 실행 확인 안 됨" in [item.value for item in app.caption]
    assert any(item.value == "`INFO worker started`" for item in app.markdown)
    assert not app.code
    assert export_answer_text(answer, include_sources=True) == "인라인 코드 · 실행 확인 안 됨\n\n`INFO worker started`"
