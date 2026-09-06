from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_document_renders_content_once_and_keeps_code_layout():
    """The displayed document preserves block order and code without a duplicate answer."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_chat_history
from src.core.answer_schema import AnswerDocument, ContentUnit, ParagraphBlock, CodeBlock, finalize_answer

document = AnswerDocument(blocks=[
    ParagraphBlock(content=[ContentUnit(text="한 번만 표시할 설명", basis="interaction", refs=[])]),
    CodeBlock(language="python", content=ContentUnit(text="def value():\\n    return 2\\n", basis="example", refs=[])),
])
render_chat_history([{"role":"assistant", "response":finalize_answer(document, [])}], "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.value for item in app.markdown].count("한 번만 표시할 설명") == 1
    assert [item.value for item in app.code] == ["def value():\n    return 2\n"]
    assert any(item.value == "코드 예시 · 실행 확인 안 됨" for item in app.caption)


def test_chat_history_displays_typed_user_and_assistant_messages():
    """User text and the assistant document survive the same history render."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_chat_history
from src.core.answer_schema import finalize_answer, text_document

render_chat_history([
    {"role":"user", "content":"질문입니다"},
    {"role":"assistant", "response":finalize_answer(text_document("답변입니다"), [])},
], "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.value for item in app.markdown] == ["질문입니다", "답변입니다"]


def test_stream_completion_preserves_full_response_in_history():
    """The final stream response reaches history without flattening its typed content."""
    app = AppTest.from_string('''
import streamlit as st
from src.app.web.streamlit_chat import process_chat_prompt, render_chat_history
from src.app.web.streamlit_api_client import AgentCallResult, AgentStreamEvent
from src.core.answer_schema import finalize_answer, text_document

if "messages" not in st.session_state:
    st.session_state.messages = []

def stream_agent(prompt):
    yield AgentStreamEvent(event="stage_started", data={"stage":"synthesis"})
    response = finalize_answer(text_document("완료된 답변"), [])
    yield AgentStreamEvent(event="final_response", result=AgentCallResult(response=response))

if not st.session_state.messages:
    process_chat_prompt("질문", st.session_state.messages.append, st.session_state.messages.append, stream_agent)
else:
    render_chat_history(st.session_state.messages, "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.value for item in app.markdown] == ["질문", "완료된 답변"]
    assert list(app.session_state.messages[1]) == ["role", "response"]


def test_action_failure_is_separate_from_answer_content():
    """A delivery failure is visible without modifying the canonical answer."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_chat_history
from src.core.answer_schema import ActionReceipt, finalize_answer, text_document

response = finalize_answer(text_document("본문입니다"), [], actions=[ActionReceipt(kind="slack_notify", status="error", error="채널을 찾을 수 없습니다")])
render_chat_history([{"role":"assistant", "response":response}], "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.value for item in app.markdown] == ["본문입니다"]
    assert any("채널을 찾을 수 없습니다" in item.value for item in app.error)


def test_numbered_citations_open_the_matching_snapshot_and_explain_check_limits():
    """The source control at each block exposes its source and the actual check scope."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_chat_history
from tests.web.answer_fixtures import cited_response
render_chat_history([{"role":"assistant", "response":cited_response()}], "http://localhost:8000")
''').run()

    assert not app.exception
    popovers = app.get("popover")
    assert [item.proto.popover.label for item in popovers] == ["근거 [1]", "근거 [1]"]
    for item in popovers:
        assert item.code[0].value == "def value():\n    return 3\n"
        assert any("7–8행" in caption.value for caption in item.caption)
        assert item.expander[0].label == "당시 원문과 위치 보기"
    assert any("별도로 평가하지 않았습니다" in item.value for item in app.caption)
    assert any("파일 다운로드" in item.value and "/download/result.txt" in item.value for item in app.markdown)


def test_missing_reference_is_visible_next_to_affected_content():
    """An unresolved reference is displayed as a limitation rather than a fake citation."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_chat_history
from src.core.answer_schema import finalize_answer, text_document
response = finalize_answer(text_document("확인이 필요한 내용", basis="source", refs=["missing"]), [])
render_chat_history([{"role":"assistant", "response":response}], "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.value for item in app.markdown] == ["확인이 필요한 내용"]
    assert [item.value for item in app.warning] == ["이 내용에 연결된 근거를 찾지 못했습니다."]
    assert not app.get("popover")


def test_heading_list_and_table_keep_document_order_and_cell_boundaries():
    """Non-prose blocks retain their structure, including pipes inside a table cell."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_chat_history
from src.core.answer_schema import AnswerDocument, ContentUnit, HeadingBlock, ListBlock, TableBlock, finalize_answer

def unit(text):
    return ContentUnit(text=text)
document = AnswerDocument(blocks=[
    HeadingBlock(level=2, content=unit("설정")),
    ListBlock(ordered=True, items=[unit("첫 단계"), unit("다음 단계")]),
    TableBlock(columns=[unit("옵션"), unit("설명")], rows=[[unit("a | b"), unit("첫 줄\\n둘째 줄")]]),
])
render_chat_history([{"role":"assistant", "response":finalize_answer(document, [])}], "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.value for item in app.markdown] == [
        "## 설정",
        "1. 첫 단계\n2. 다음 단계",
        "| 옵션 | 설명 |\n| --- | --- |\n| a \\| b | 첫 줄<br>둘째 줄 |",
    ]


def test_unit_limitation_is_placed_after_its_block():
    """A limitation follows the content it qualifies instead of an unrelated opening."""
    app = AppTest.from_string('''
from src.app.web.streamlit_chat import render_answer_response
from src.core.answer_schema import AnswerDocument, ContentUnit, ParagraphBlock, ResponseIssue, finalize_answer
document = AnswerDocument(blocks=[ParagraphBlock(content=[ContentUnit(text="먼저 설명")]), ParagraphBlock(content=[ContentUnit(text="범위가 제한된 설명")])])
response = finalize_answer(document, [], issues=[ResponseIssue(code="limited", message="두 번째 내용의 제한", unit_id="b1.content.0")])
render_answer_response(response, "http://localhost:8000")
''').run()

    assert not app.exception
    assert [item.type for item in app.main.children.values()] == ["markdown", "markdown", "warning"]
    assert app.warning[0].value == "두 번째 내용의 제한"


def test_mutated_checked_body_is_rejected_before_any_content_is_displayed():
    """The UI cannot display changed text under a previous revision's checks."""
    app = AppTest.from_string('''
import streamlit as st
from pydantic import ValidationError
from src.app.web.streamlit_chat import render_answer_response
from src.core.answer_schema import finalize_answer, text_document

response = finalize_answer(text_document("검사 당시 내용"), [])
response.content.blocks[0].content[0].text = "검사 후 변조된 내용"
try:
    render_answer_response(response, "http://localhost:8000")
except ValidationError:
    st.error("답변 내용의 변경이 감지되었습니다.")
''').run()

    assert not app.exception
    assert not app.markdown
    assert [item.value for item in app.error] == ["답변 내용의 변경이 감지되었습니다."]
