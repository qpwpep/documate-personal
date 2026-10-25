"""Rendered answer checks describe body scope independently of delivery results."""
from __future__ import annotations

import pytest
from streamlit.testing.v1 import AppTest

from src.core.answer_export import build_answer_export_operation
from src.core.answer_schema import (
    ActionReceipt,
    AnswerDocument,
    AnswerResponse,
    ResponseIssue,
    SemanticCheck,
    SemanticValidation,
    finalize_answer,
    export_answer_text,
    semantic_evidence_fingerprint,
    text_document,
)
from src.core.contracts.outcome import TurnResult
from src.core.llm_errors import make_problem
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


def test_removed_content_explains_known_scope_and_next_step_before_technical_details():
    evidence = _evidence("확인한 문서 내용", source_type="official")
    answer = finalize_answer(text_document("확인한 문서 내용", basis="source", refs=[evidence.id]), [evidence], issues=[
        ResponseIssue(code="invalid_content_removed",
                      message="근거로 충분히 뒷받침되지 않거나 원문과 일치하지 않는 내용을 제외했습니다."),
    ])
    answer = answer.model_copy(update={"semantic_validation": SemanticValidation(
        content_hash=answer.content_hash,
        evidence_fingerprint=semantic_evidence_fingerprint(answer.citations), threshold=0.8,
        checks=[SemanticCheck(name="request_fulfilled", probability=0.95),
                SemanticCheck(name="grounded", unit_id="b0.content.0", probability=0.9)],
    )})

    app = _render(TurnResult(response=answer))

    visible = [item.value for item in app.main.children.values() if item.type in {"caption", "markdown", "warning"}]
    assert any("확인한 범위" in text and "원문에 연결된 내용 1개" in text for text in visible)
    assert any("확인하지 못한 범위" in text and "제외된 항목의 상세 기록은 없습니다" in text for text in visible)
    assert any("다음 행동" in text and "필요한 항목이 빠졌다면" in text for text in visible)
    assert not any("Decisions API" in text or "충족 확률" in text or "임곗값" in text for text in visible)
    details = next(item for item in app.expander if item.label == "검증 상세")
    assert not details.proto.expanded
    assert any("충족 확률 0.95" in item.value for item in details.caption)
    assert any("답변 전체의 정확도" in item.value and "보증" in item.value for item in details.caption)
    assert answer.issues[0].message in [item.value.removeprefix("답변에 기록된 한계: ") for item in app.warning]


@pytest.mark.parametrize("with_held_delivery", [False, True])
def test_partial_answer_keeps_one_summary_visible_and_preserves_limits_in_collapsed_details(with_held_delivery):
    evidence = _evidence("확인한 문서 내용", source_type="official")
    answer = finalize_answer(text_document("확인한 문서 내용", basis="source", refs=[evidence.id]), [evidence], issues=[
        ResponseIssue(code="answer_incomplete", message="비교 요청 전체를 완성하지 못했습니다."),
        ResponseIssue(code="retrieved_evidence_incomplete", message="첨부 코드의 동작은 확인하지 못했습니다."),
        ResponseIssue(code="scope_limited", unit_id="b0.content.0", message="이 내용은 공식 문서 범위에 한정됩니다."),
    ])
    if with_held_delivery:
        answer = answer.model_copy(update={"actions": [ActionReceipt(
            kind="slack_notify", status="skipped", message="본문을 완성하지 못해 전송하지 않았습니다.",
        )]})
    result = TurnResult(status="partial", response=answer,
                        problem=make_problem("evidence_insufficient", "validation"), request_id="partial-scope")

    app = _render(result)

    visible_warnings = [item for item in app.main.children.values() if item.type == "warning"]
    assert len(visible_warnings) == 1
    assert "부분 답변" in visible_warnings[0].value
    assert "자료나 조건을 추가" in visible_warnings[0].value
    assert any("확인한 문서 내용" in item.value for item in app.markdown)
    details = next(item for item in app.expander if item.label == "답변의 한계와 처리 정보")
    assert not details.proto.expanded
    for issue in answer.issues[:2]:
        assert any(issue.message in item.value for item in details.warning)
    assert any(result.problem.message in item.value for item in details.markdown)
    assert any("partial-scope" in item.value for item in details.caption)
    unit_details = next(item for item in app.expander if item.label == "이 내용의 한계 (1개)")
    assert not unit_details.proto.expanded
    assert any(answer.issues[2].message in item.value for item in unit_details.warning)
    assert any("확인한 문서 내용" in item.value for item in unit_details.caption)


@pytest.mark.parametrize(("basis", "refs", "warning"), [
    ("source", ["missing-source"], "이 내용에 연결된 근거를 찾지 못했습니다."),
    ("excerpt", None, "이 인용문은 보관한 원문과 일치하지 않습니다."),
])
def test_source_integrity_warning_stays_visible_beside_affected_content(basis, refs, warning):
    evidence = _evidence("자료의 실제 내용", source_type="official")
    answer = finalize_answer(
        text_document("주의가 필요한 내용", basis=basis, refs=refs or [evidence.id]), [evidence],
        issues=[ResponseIssue(code="scope_limited", unit_id="b0.content.0", message="범위를 확인해 주세요.")],
    )

    app = _render(TurnResult(response=answer))

    visible_warnings = [item.value for item in app.main.children.values() if item.type == "warning"]
    assert warning in visible_warnings
    assert any("주의가 필요한 내용" in item.value for item in app.markdown)
    assert any(item.label == "이 내용의 한계 (1개)" for item in app.expander)


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
    assert not any("부분 답변" in item.value for item in app.warning)


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
