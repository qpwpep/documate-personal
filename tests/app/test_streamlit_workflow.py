"""Observable chat behavior with only the HTTP transport replaced."""
from __future__ import annotations

import json
import threading
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.answer_export import build_answer_export_operation
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions.receipts import build_save_receipt
from tests.app.streamlit_support import finish_chat_response


APP = Path(__file__).resolve().parents[2] / "src/app/web/streamlit_app.py"


class HTTPBytes:
    def __init__(self, body: bytes):
        self.body = body

    def stream(self, *_args, **_kwargs):
        yield self.body

    def close(self):
        pass

    def release_conn(self):
        pass


class RequestsObserved(list):
    def __init__(self):
        super().__init__()
        self.exports = []
        self.export_failure = None
        self.natural_save_failure = False
        self.failure_code = None
        self.upload_files = []
        self.upload_revision = 0
        self.upload_failure = None
        self.upload_manifest_failure = None
        self.before_manifest_failure = None
        self.upload_syncs = []
        self.missing_slots = []
        self.active_snapshot = None
        self.manifest_reads = 0
        self.manifest_reads_at_submission = 0
        self.response_release = None
        self.response_started = threading.Event()
        self.response_gates = []


@pytest.fixture
def chat_http(monkeypatch, tmp_path, browser_storage):
    from src.app.web import streamlit_state

    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path / "uploads")
    submissions = RequestsObserved()
    answers = {}
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path / "saved")

    def send(_adapter, request, **_kwargs):
        path = urlparse(request.url).path
        result = requests.Response()
        result.status_code = 200
        result.request = request
        result.url = request.url
        if request.method == "GET" and path.endswith("/uploads"):
            submissions.manifest_reads += 1
            if submissions.upload_manifest_failure:
                result.status_code = 503
                data = {"detail": {"code": "UPLOAD_REQUEST_FAILED", "message": submissions.upload_manifest_failure}}
                if submissions.before_manifest_failure is not None:
                    submissions.before_manifest_failure()
            else:
                data = {"epoch": "epoch-" + path.split("/")[2], "revision": submissions.upload_revision, "files": submissions.upload_files}
            result._content = json.dumps(data).encode()
        elif request.method == "POST" and path.endswith("/uploads/sync"):
            payload = json.loads(request.body)
            submissions.upload_syncs.append(payload)
            if submissions.upload_failure:
                result.status_code = 422
                data = {"detail": {"code": "UPLOAD_VALIDATION_FAILED", "message": submissions.upload_failure}}
            else:
                submissions.upload_revision += 1
                for addition in payload["add"]:
                    content = Path(addition["path"]).read_bytes()
                    submissions.upload_files.append({
                        "file_id": "corrected-file", "name": addition["name"],
                        "size_bytes": len(content), "content_hash": addition["content_hash"],
                        "source_uri": "upload:///corrected-file",
                    })
                data = {"epoch": "epoch-" + path.split("/")[2], "revision": submissions.upload_revision,
                        "files": submissions.upload_files}
            result._content = json.dumps(data).encode()
        elif request.method == "POST" and path == "/agent/stream":
            payload = json.loads(request.body)
            submissions.append(payload)
            submissions.active_snapshot = browser_storage.snapshot
            submissions.manifest_reads_at_submission = submissions.manifest_reads
            answer = finalize_answer(text_document("답변: " + payload["query"]), [])
            answer_id = f"answer-{len(submissions)}"
            answers[answer_id] = answer
            final = {
                "status": "completed", "response": answer.model_dump(mode="json"),
                "answer_id": answer_id,
                "upload_manifest": {"epoch": "epoch-" + payload["session_id"], "revision": submissions.upload_revision, "files": submissions.upload_files},
                "request_id": f"test-request-{len(submissions)}",
            }
            if submissions.failure_code:
                from src.core.llm_errors import make_problem

                final.update(
                    status="failed", response=None, answer_id=None,
                    problem=make_problem(submissions.failure_code, "pre_synthesis_validation").model_dump(mode="json"),
                )
            if submissions.missing_slots and not submissions.upload_files:
                final.update(status="needs_input", response=None, answer_id=None,
                             message="확인할 파일을 먼저 업로드한 뒤 다시 질문해 주세요.",
                             missing_slots=submissions.missing_slots)
            if submissions.natural_save_failure:
                source = answers["answer-1"]
                operation = build_answer_export_operation(source, session_id=payload["session_id"])
                receipt = build_save_receipt({"status": "error", "error_code": "write_failed", "error": "Permission denied: server-private-path/.save.lock"}, operation=operation)
                final.update(status="partial", answer_id="answer-1",
                             response=source.model_copy(update={"actions": [receipt]}).model_dump(mode="json"),
                             problem={"code": "save_failed", "stage": "action_postprocess", "message": "파일을 만들지 못했습니다.", "next_action": "retry_later"})
            result.headers["Content-Type"] = "text/event-stream; charset=utf-8"
            result.raw = HTTPBytes(("event: final_response\ndata: " + json.dumps(final, ensure_ascii=False) + "\n\n").encode())
            if submissions.response_release is not None:
                body = result.raw.body
                release = submissions.response_release
                submissions.response_gates.append(release)

                class PendingHTTPBytes(HTTPBytes):
                    def stream(self, *_args, **_kwargs):
                        submissions.response_started.set()
                        assert release.wait(10), "test response was not released"
                        yield body

                result.raw = PendingHTTPBytes(body)
        elif request.method == "POST" and path.endswith("/save"):
            submissions.exports.append(path)
            if submissions.export_failure == "network":
                raise requests.ConnectionError("save boundary offline")
            if submissions.export_failure == "http":
                result.status_code = 503
                data = {"detail": {"message": "저장 서버가 잠시 중단되었습니다."}}
            else:
                answer_id = path.split("/")[4]
                answer = answers[answer_id]
                operation = build_answer_export_operation(answer, session_id=path.split("/")[2])
                saved = build_save_text_tool()(export_answer_text(answer, include_sources=True), operation=operation)
                receipt = build_save_receipt(saved, operation=operation)
                data = {"status": "completed", "answer_id": answer_id, "request_id": "save-request",
                        "response": answer.model_copy(update={"actions": [receipt]}).model_dump(mode="json"),
                        "message": "저장했습니다."}
            result._content = json.dumps(data).encode()
        else:
            raise AssertionError(f"Unexpected HTTP request: {request.method} {path}")
        return result

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    yield submissions
    for gate in submissions.response_gates:
        gate.set()


def test_pending_question_allows_draft_and_navigation_without_replaying(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    chat_http.response_release = threading.Event()
    try:
        app.chat_input[0].set_value("답변을 기다릴 질문").run()
        first = app.session_state["active_conversation_id"]
        conversation = app.session_state["conversations"][first]
        assert chat_http.response_started.wait(1)
        assert conversation.is_busy
        assert not app.chat_input[0].disabled
        assert not app.button(key="documate_new_chat").disabled
        app.chat_input[0].set_value("기다리면서 쓴 다음 질문").run()
        assert len(chat_http) == 1
        app.button(key="documate_new_chat").click().run()
        second = app.session_state["active_conversation_id"]
        assert first != second
        assert conversation.is_busy
        assert not any("답변: 답변을 기다릴 질문" in item.value for item in app.markdown)
        release = chat_http.response_release
        chat_http.response_release = None
        app.chat_input[0].set_value("다른 대화에서 독립적으로 보낸 질문").run()
        second_task = app.session_state["conversations"][second].chat_task
        if second_task is not None:
            assert second_task.finished.wait(2)
            app.run()
        assert any("답변: 다른 대화에서 독립적으로 보낸 질문" in item.value for item in app.markdown)
        next(item for item in app.selectbox if item.label == "대화 선택").set_value(first).run()
        assert app.chat_input[0].proto.value == "기다리면서 쓴 다음 질문"
        assert len(chat_http) == 2
        task = conversation.chat_task
        release.set()
        assert task.finished.wait(2)
        app.run()
        assert not app.exception
        assert not conversation.is_busy
        assert conversation.draft.text == "기다리면서 쓴 다음 질문"
        assert any("답변: 답변을 기다릴 질문" in item.value for item in app.markdown)
        assert not any("답변: 다른 대화에서 독립적으로 보낸 질문" in item.value for item in app.markdown)
        assert len(chat_http) == 2
    finally:
        for gate in chat_http.response_gates:
            gate.set()


def test_completion_keeps_the_mounted_composer_without_prefilling_stale_python_text(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    release = threading.Event()
    chat_http.response_release = release
    app.chat_input[0].set_value("대기 중 브라우저에서 초안을 작성할 질문").run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    task = conversation.chat_task
    assert chat_http.response_started.wait(1)
    assert not conversation.draft.needs_restore
    # Until blur or navigation, browser typing is newer than Python's draft.
    # A response update must not send an empty prefill to that mounted widget.
    assert conversation.draft.text == ""
    release.set()
    assert task.finished.wait(2)
    app.run()
    assert not app.exception
    assert not app.chat_input[0].proto.set_value
    assert not conversation.draft.needs_restore
    assert any("답변: 대기 중 브라우저에서 초안을 작성할 질문" in item.value for item in app.markdown)


def test_interrupted_response_restores_navigation_and_retains_question_without_replay(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    prompt = "  수신 중단 뒤에도 남길 질문\n두 번째 줄  "
    chat_http.response_release = threading.Event()

    app.chat_input[0].set_value(prompt).run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    task = conversation.chat_task
    assert chat_http.response_started.wait(1)
    app.button(key=conversation.widget_key("stop_response")).click().run()
    chat_http.response_release.set()
    assert task.finished.wait(2)
    app.run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == [prompt]
    assert not app.chat_input[0].disabled
    assert not app.button(key="documate_new_chat").disabled
    # Manifest reads share the server request lock. A recovery render must not
    # wait for the very server operation whose response the user stopped.
    assert chat_http.manifest_reads == chat_http.manifest_reads_at_submission
    assert any("응답 수신이 중단" in item.value for item in app.warning)
    assert any(prompt.strip() in item.value for item in app.markdown)
    assert not any(item.label == "같은 질문 다시 보내기" for item in app.button)
    next(item for item in app.button if item.label == "질문 수정").click().run()
    assert app.chat_input[0].proto.value == prompt
    app.run()
    assert len(chat_http) == 1
    chat_http.response_release = None
    reads_before_next_question = chat_http.manifest_reads
    finish_chat_response(app.chat_input[0].set_value("명시적으로 보낸 다음 질문").run())
    assert [item["query"] for item in chat_http] == [prompt, "명시적으로 보낸 다음 질문"]
    assert chat_http.manifest_reads > reads_before_next_question
    assert any("답변: 명시적으로 보낸 다음 질문" in item.value for item in app.markdown)


@pytest.mark.parametrize("snapshot_stage", ["queued", "receiving"])
def test_reload_during_response_retains_submitted_question_without_replay(chat_http, browser_storage, snapshot_stage):
    app = AppTest.from_file(APP, default_timeout=10).run()
    prompt = "새로고침 직전에 보낸 질문"
    chat_http.response_release = threading.Event()
    app.chat_input[0].set_value(prompt).run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    task = conversation.chat_task
    assert chat_http.response_started.wait(1)
    if snapshot_stage == "queued":
        browser_storage.snapshot = chat_http.active_snapshot

    restored = AppTest.from_file(APP, default_timeout=10).run()

    assert not restored.exception
    assert any(prompt in item.value for item in restored.markdown)
    assert any("응답 수신이 중단" in item.value for item in restored.warning)
    assert not restored.chat_input[0].disabled
    assert chat_http.manifest_reads == chat_http.manifest_reads_at_submission
    assert [item["query"] for item in chat_http] == [prompt]
    next(item for item in restored.button if item.label == "질문 수정").click().run()
    assert restored.chat_input[0].proto.value == prompt
    assert len(chat_http) == 1
    task.stopped.set()
    chat_http.response_release.set()
    assert task.finished.wait(2)


def test_late_response_after_stopping_cannot_replace_the_next_answer(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    release = threading.Event()
    chat_http.response_release = release
    app.chat_input[0].set_value("이전 스트림 종료 전에 복구할 질문").run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    task = conversation.chat_task
    assert chat_http.response_started.wait(1)
    app.button(key=conversation.widget_key("stop_response")).click().run()
    assert not conversation.is_busy
    assert not task.finished.is_set()
    chat_http.response_release = None
    finish_chat_response(app.chat_input[0].set_value("명시적으로 보낸 다음 질문").run())
    release.set()
    assert task.finished.wait(2)
    app.run()
    assert not app.exception
    assert len([item for item in app.warning if "응답 수신이 중단" in item.value]) == 1
    assert not app.button(key="documate_new_chat").disabled
    assert not app.chat_input[0].disabled
    assert len(chat_http) == 2
    assert not any("답변: 이전 스트림 종료 전에 복구할 질문" in item.value for item in app.markdown)
    assert any("답변: 명시적으로 보낸 다음 질문" in item.value for item in app.markdown)


def test_interrupted_validation_cannot_overwrite_the_next_question(chat_http, browser_storage):
    import streamlit as st

    from src.app.web.streamlit_requests import queue_chat

    app = AppTest.from_file(APP, default_timeout=10).run()

    def interrupt_and_submit_next_question():
        # The old validation response arrives after the browser has stopped
        # receiving it and explicitly submitted the next question.
        conversation_id = browser_storage.data["conversation_id"]
        st.session_state["documate_browser_state"] = {"interrupt": {
            "conversation_id": conversation_id,
            "operation_id": browser_storage.data["operation_id"],
            "nonce": 1,
        }}
        browser_storage.callbacks["on_interrupt_change"]()
        conversation = st.session_state["conversations"][conversation_id]
        queue_chat(conversation, "중단 후 명시적으로 보낸 새 질문")
        chat_http.upload_manifest_failure = None
        chat_http.before_manifest_failure = None

    chat_http.upload_manifest_failure = "이전 요청의 연결 확인 실패"
    chat_http.before_manifest_failure = interrupt_and_submit_next_question
    finish_chat_response(app.chat_input[0].set_value("연결 확인 중 중단한 이전 질문").run())

    assert not app.exception
    assert [item["query"] for item in chat_http] == ["중단 후 명시적으로 보낸 새 질문"]
    assert not app.chat_input[0].disabled
    assert any("답변: 중단 후 명시적으로 보낸 새 질문" in item.value for item in app.markdown)
    assert any("응답 수신이 중단" in item.value for item in app.warning)
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    assert conversation.saved_prompt is None
    assert conversation.validation_error is None


def test_new_conversation_guides_input_without_presenting_a_verified_answer(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()

    assert not app.exception
    assert not app.chat_message
    assert not any(item.label == "검증 상세" for item in app.expander)
    assert any("파일을 첨부" in item.value for item in app.markdown)
    assert not chat_http


def test_suggested_question_prefills_then_sends_only_on_explicit_submission(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    suggestion = next(item for item in app.button if "quick_prompt_" in (item.key or ""))
    prompt = suggestion.label
    suggestion.click().run()

    assert not app.exception
    assert not chat_http
    assert not app.chat_message
    assert app.chat_input[0].proto.value == prompt
    finish_chat_response(app.chat_input[0].set_value(prompt).run())
    app.run()
    assert [item["query"] for item in chat_http] == [prompt]
    assert any("답변: " + prompt in item.value for item in app.markdown)


def test_submitted_input_preserves_boundary_whitespace_for_exact_copy(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    prompt = "  아래 원문을 공백과 개행까지 그대로 복사해줘:\n    value = 1\n\n"

    finish_chat_response(app.chat_input[0].set_value(prompt).run())
    app.run()

    assert not app.exception
    assert [submission["query"] for submission in chat_http] == [prompt]


def test_examples_are_stable_and_require_confirmed_attachments(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    initial = [item.label for item in app.button if any((item.key or "").endswith(f"quick_prompt_{index}") for index in range(4))]
    assert len(initial) == 4
    assert not any(word in prompt for prompt in initial for word in ("첨부", "업로드", "방금", "Slack", "저장"))
    app.button(key="documate_new_chat").click().run()
    assert [item.label for item in app.button if any((item.key or "").endswith(f"quick_prompt_{index}") for index in range(4))] == initial

    chat_http.upload_files = [{
        "file_id": "sample", "name": "sample.py", "size_bytes": 10,
        "content_hash": "sha256:" + "a" * 64, "source_uri": "upload:///sample.py",
    }]
    next(item for item in app.button if item.label == "첨부 목록 새로고침").click().run()
    with_files = [item.label for item in app.button if "quick_prompt_" in (item.key or "")]
    assert any("첨부한 자료" in prompt for prompt in with_files)
    assert not any(word in prompt for prompt in with_files for word in ("방금", "Slack", "저장"))
    assert not app.exception
    assert not chat_http


def test_current_conversation_and_searchable_files_are_visible_in_main_header(chat_http):
    chat_http.upload_files = [{
        "file_id": "sample", "name": "sample.py", "size_bytes": 10,
        "content_hash": "sha256:" + "a" * 64, "source_uri": "upload:///sample.py",
    }]
    app = AppTest.from_file(APP, default_timeout=10).run()
    prompt = "첨부한 함수의 반환값을 설명해줘"
    finish_chat_response(app.chat_input[0].set_value(prompt).run())
    app.run()

    assert not app.exception
    header = next(item.value for item in app.markdown if 'class="dm-topbar"' in item.value)
    assert prompt in header
    assert "첨부 자료 1개" in header
    assert not any("다음 질문부터" in item.value for item in app.caption)
    assert any("실제 사용한 자료는 답변의 근거에서 확인" in item.value for item in app.caption)


def test_suggestion_does_not_replace_a_draft_received_during_the_click(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    suggestion = next(item for item in app.button if "quick_prompt_" in (item.key or ""))
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    text = "  작성하던 질문\n두 번째 줄  "
    conversation.draft.record(text, revision=conversation.draft.revision)
    suggestion.click().run()

    assert not app.exception
    assert conversation.draft.text == text
    assert not chat_http
    assert all(item.disabled for item in app.button if "quick_prompt_" in (item.key or ""))
    assert any("작성 중인 질문" in item.value for item in app.caption)


def test_examples_are_hidden_while_attachment_changes_need_confirmation(chat_http):
    from src.app.web.streamlit_state import PendingUpload

    app = AppTest.from_file(APP, default_timeout=10).run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    conversation.pending_upload = PendingUpload(
        base_manifest=conversation.client.manifest, clear=True, requires_confirmation=True,
    )
    app.run()
    assert not app.exception
    assert not any("quick_prompt_" in (item.key or "") for item in app.button)
    assert not chat_http


def test_incomplete_slack_setting_does_not_block_regular_question(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    assert not app.exception
    next(item for item in app.radio if item.label == "수신자 유형").set_value("채널").run()
    finish_chat_response(app.chat_input[0].set_value("NumPy reshape를 설명해줘").run())

    assert not app.exception
    assert [item["query"] for item in chat_http] == ["NumPy reshape를 설명해줘"]
    assert len([item for item in app.chat_message if item.name == "user"]) == 1
    assert any("답변: NumPy reshape를 설명해줘" in item.value for item in app.markdown)
    assert not any("아직 보내지 않은 질문" in item.value for item in app.info)
    app.run()
    assert len(chat_http) == 1


def test_sending_held_question_clears_recovery_controls_and_does_not_replay(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.session_state["conversations"][app.session_state["active_conversation_id"]].saved_prompt = "보류했던 질문"
    app.run()
    finish_chat_response(app.button(key="documate_send_saved_upload_prompt").click().run())

    assert not app.exception
    assert [item["query"] for item in chat_http] == ["보류했던 질문"]
    assert not any("아직 보내지 않은 질문" in item.value for item in app.info)
    assert not any(item.label == "보류한 질문 보내기" for item in app.button)
    app.run()
    assert len(chat_http) == 1


class _UploadedSample:
    def __init__(self, name: str, content: bytes) -> None:
        self.name = name
        self.content = content

    def getbuffer(self) -> bytes:
        return self.content


@pytest.mark.parametrize("missing_slots", [["upload"], ["subject"]], ids=["upload", "other-clarification"])
def test_file_only_upload_confirms_readiness_and_explicitly_continues_only_missing_upload_question(chat_http, tmp_path, missing_slots):
    from src.app.web.streamlit_state import PendingUpload

    chat_http.missing_slots = missing_slots
    app = AppTest.from_file(APP, default_timeout=10).run()
    prompt = "  첨부한 코드를 요약해줘.\n함수도 설명해줘.  "
    finish_chat_response(app.chat_input[0].set_value(prompt).run())
    assert not any(item.label == "이전 질문 계속하기" for item in app.button)
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    staged = conversation.client.stage_files(
        [_UploadedSample("sample.py", b"VALUE = 42\n")], tmp_path / "uploads" / conversation.id,
        max_files=10, max_file_mib=1, max_total_mib=10,
    )
    conversation.pending_upload = PendingUpload(base_manifest=conversation.client.manifest, files=staged.files)
    app.run()

    assert not app.exception
    attachments = next(item for item in app.expander if item.label == "첨부 1개 · 검색 가능")
    assert not attachments.proto.expanded
    assert any("sample.py" in item.value for item in app.caption)
    assert not any("다음 질문부터" in item.value for item in app.caption)
    assert [item["query"] for item in chat_http] == [prompt]
    app.run()
    assert len(chat_http) == 1
    if missing_slots != ["upload"]:
        assert not any(item.label == "이전 질문 계속하기" for item in app.button)
        return

    draft = "따로 작성 중인 다음 질문"
    conversation.draft.edit(draft)
    finish_chat_response(app.button(key="documate_continue_upload_question").click().run())
    assert not app.exception
    assert [item["query"] for item in chat_http] == [prompt, prompt]
    assert chat_http[-1]["uploads"] == {"epoch": "epoch-" + conversation.id, "revision": 1}
    assert app.chat_input[0].proto.value == draft
    assert not any(item.label == "이전 질문 계속하기" for item in app.button)
    app.run()
    assert len(chat_http) == 2


@pytest.mark.parametrize("recovery", ["파일 다시 선택", "기존 자료로 계속"])
@pytest.mark.parametrize("prompt", ["  이 노트북을 요약해줘.\n근거도 보여줘.  ", None], ids=["with-question", "file-only"])
def test_failed_attachment_can_be_replaced_without_losing_question_or_existing_files(chat_http, tmp_path, prompt, recovery):
    from src.app.web.streamlit_state import PendingUpload

    existing = {
        "file_id": "existing-file", "name": "existing.py", "size_bytes": 10,
        "content_hash": "sha256:" + "a" * 64, "source_uri": "upload:///existing-file",
    }
    chat_http.upload_files = [existing]
    chat_http.upload_failure = "sample.ipynb: Expecting value: line 1 column 1 (char 0)"
    app = AppTest.from_file(APP, default_timeout=10).run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    session_path = tmp_path / "uploads" / conversation.id
    invalid = conversation.client.stage_files(
        [_UploadedSample("sample.ipynb", b"not a notebook")], session_path,
        max_files=10, max_file_mib=1, max_total_mib=10,
    )
    # AppTest cannot choose native files. Enter at the prepared-upload boundary;
    # staging, HTTP handling, recovery controls and question submission are real.
    conversation.pending_upload = PendingUpload(
        base_manifest=conversation.client.manifest, files=invalid.files, prompt=prompt,
    )
    app.run()

    assert not app.exception
    assert not chat_http
    assert len(chat_http.upload_syncs) == 1
    assert conversation.pending_upload.had_failure
    assert any(item.label == recovery for item in app.button)
    assert any("첨부 준비 실패" in item.value for item in app.markdown)
    assert not any("첨부 연결 확인 중" in item.value for item in app.markdown)
    assert app.chat_input[0].disabled
    assert any("첨부 변경을 완료하지 못했습니다" in item.value for item in app.markdown)
    assert not any("반영 확인 대기" in item.value for item in app.markdown)
    assert not any("Expecting value" in item.value for item in app.error)
    details = next(item for item in app.expander if "오류 상세" in item.label)
    assert not details.proto.expanded
    next(item for item in app.button if item.label == recovery).click().run()

    assert not app.exception
    assert conversation.pending_upload is None
    assert conversation.saved_prompt is None
    assert app.chat_input[0].proto.value == (prompt or "")
    assert not app.chat_input[0].disabled
    assert app.chat_input[0].proto.accept_file
    assert [item.file_id for item in conversation.client.manifest.files] == ["existing-file"]
    assert not chat_http
    restored_prompt = app.chat_input[0].proto.value or None
    app.run()
    assert len(chat_http.upload_syncs) == 1

    chat_http.upload_failure = None
    corrected_content = b'{"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}'
    corrected = conversation.client.stage_files(
        [_UploadedSample("sample.ipynb", corrected_content)], session_path,
        max_files=10, max_file_mib=1, max_total_mib=10,
    )
    assert Path(corrected.files[0].path).read_bytes() == corrected_content
    conversation.draft.consume()
    conversation.pending_upload = PendingUpload(
        base_manifest=conversation.client.manifest, files=corrected.files, prompt=restored_prompt,
    )
    app.run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == ([prompt] if prompt else [])
    assert [item.file_id for item in conversation.client.manifest.files] == ["existing-file", "corrected-file"]
    assert len(chat_http.upload_syncs) == 2
    assert chat_http.upload_syncs[0]["operation_id"] != chat_http.upload_syncs[1]["operation_id"]
    if prompt:
        assert chat_http[0]["uploads"] == {"epoch": "epoch-" + conversation.id, "revision": 1}
    app.run()
    assert len(chat_http) == (1 if prompt else 0)
    assert len(chat_http.upload_syncs) == 2


def test_failed_upload_review_keeps_question_and_hides_latest_technical_error(chat_http, tmp_path):
    from src.app.web.streamlit_state import PendingUpload

    app = AppTest.from_file(APP, default_timeout=10).run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    staged = conversation.client.stage_files(
        [_UploadedSample("sample.py", b"VALUE = 42\n")], tmp_path / "uploads" / conversation.id,
        max_files=10, max_file_mib=1, max_total_mib=10,
    )
    prompt = "  첨부를 확인한 뒤 답해줘.\n질문을 유지해줘.  "
    pending = PendingUpload(
        base_manifest=conversation.client.manifest, files=staged.files, prompt=prompt,
        attempted=True, had_failure=True, needs_refresh_review=True,
        error="첨부 목록이 변경되었습니다.", error_detail="previous revision conflict",
    )
    conversation.pending_upload = pending
    app.run()
    raw_error = "manifest store unavailable: server-private-path/uploads"
    chat_http.upload_manifest_failure = raw_error

    app.button(key="documate_review_uploads").click().run()

    assert not app.exception
    assert conversation.pending_upload is pending
    assert pending.prompt == prompt
    assert pending.files == staged.files
    assert pending.needs_refresh_review
    assert conversation.saved_prompt is None
    assert not chat_http
    assert not chat_http.upload_syncs
    expected = "최신 첨부 목록을 확인하지 못했습니다. 잠시 후 다시 시도하거나 첨부를 수정해 주세요."
    assert [item.value for item in app.error] == [expected]
    assert pending.error_detail == raw_error
    details = next(item for item in app.expander if item.label == "오류 상세")
    assert not details.proto.expanded
    assert [item.value for item in details.text] == [raw_error]
    assert any(prompt.strip() in item.value for item in app.info)
    assert any(item.label == "파일 다시 선택" for item in app.button)
    app.run()
    assert conversation.pending_upload is pending
    assert not chat_http
    assert not chat_http.upload_syncs


def test_held_upload_question_can_return_to_editable_composer_without_sending(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    prompt = "  보존할 질문\n두 번째 줄  "
    conversation.saved_prompt = prompt
    app.run()

    assert any(item.label == "질문과 첨부 수정" for item in app.button)
    next(item for item in app.button if item.label == "질문과 첨부 수정").click().run()

    assert not app.exception
    assert conversation.saved_prompt is None
    assert app.chat_input[0].proto.value == prompt
    assert not app.chat_input[0].disabled
    assert app.chat_input[0].proto.accept_file
    assert not chat_http
    finish_chat_response(app.chat_input[0].set_value("수정한 질문").run())
    app.run()
    assert [item["query"] for item in chat_http] == ["수정한 질문"]


def test_completed_answer_offers_direct_txt_save(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("NumPy reshape를 설명해줘").run())

    assert not app.exception
    assert any(item.label == "TXT 파일 만들기" for item in app.button)


def test_prepared_txt_offers_download_instead_of_saving_the_same_answer_again(chat_http, tmp_path):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("내려받을 답변").run())
    next(item for item in app.button if item.label == "TXT 파일 만들기").click().run()

    assert not app.exception
    assert any("파일 준비 완료" in item.value and "내려받으세요" in item.value for item in app.success)
    downloads = app.get("link_button")
    assert len(downloads) == 1
    assert downloads[0].proto.label == "TXT 다운로드"
    assert downloads[0].proto.type == "primary"
    saved = list((tmp_path / "saved").glob("*.txt"))
    assert len(saved) == 1
    assert downloads[0].proto.url.endswith("/download/" + saved[0].name)
    assert not any(item.label in {"TXT 파일 만들기", "저장 다시 시도"} for item in app.button)
    assert "답변: 내려받을 답변" in saved[0].read_text(encoding="utf-8-sig")
    app.run()
    assert len(chat_http) == 1
    assert len(chat_http.exports) == 1


@pytest.mark.parametrize("verified", [True, False], ids=["expired", "unverified"])
def test_unavailable_download_keeps_answer_and_offers_file_preparation(verified):
    from src.core.answer_schema import ActionReceipt
    from src.core.contracts.outcome import TurnResult
    from src.core.save_contract import SaveArtifact

    answer = finalize_answer(text_document("계속 읽을 수 있는 원래 답변"), [])
    operation = build_answer_export_operation(answer, session_id="download-test")
    artifact = SaveArtifact(
        artifact_id=operation.binding_sha256, filename="retained-answer.txt",
        sha256=operation.payload_sha256, byte_count=operation.byte_count,
        created_at=1, expires_at=2,
    ) if verified else None
    receipt = ActionReceipt(
        kind="save_text", status="success", file_path="retained-answer.txt",
        operation=operation if verified else None, artifact=artifact,
        verification="verified" if verified else None,
    )
    app = AppTest.from_string('''
import streamlit as st
from src.app.web.streamlit_chat import render_chat_history

def save(answer_id):
    st.session_state["save_requests"].append(answer_id)

render_chat_history(st.session_state["messages"], "http://localhost:8000", on_save=save)
''', default_timeout=10)
    app.session_state["messages"] = [{"role": "assistant", "result": TurnResult(
        status="completed", answer_id="original-answer",
        response=answer.model_copy(update={"actions": [receipt]}),
    )}]
    app.session_state["save_requests"] = []
    app.run()

    assert not app.exception
    assert any(item.value == "계속 읽을 수 있는 원래 답변" for item in app.markdown)
    assert not app.get("link_button")
    assert not app.success
    warning = "보관 기한이 지났습니다" if verified else "저장 결과를 검증할 수 없습니다"
    assert any(warning in item.value for item in app.warning)
    label = "TXT 다시 만들기" if verified else "TXT 파일 만들기"
    next(item for item in app.button if item.label == label).click().run()
    assert app.session_state["save_requests"] == ["original-answer"]


def test_reloaded_answer_can_be_saved_with_the_original_answer_id_without_regeneration(chat_http, tmp_path):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("새로고침 뒤 저장할 답변").run())
    session_id = chat_http[0]["session_id"]

    reloaded = AppTest.from_file(APP, default_timeout=10).run()
    assert not reloaded.exception
    assert len(chat_http) == 1
    assert not chat_http.exports
    next(item for item in reloaded.button if item.label == "TXT 파일 만들기").click().run()

    assert not reloaded.exception
    assert any("파일 준비 완료" in item.value for item in reloaded.success)
    assert any(item.proto.label == "TXT 다운로드" for item in reloaded.get("link_button"))
    assert chat_http.exports == [f"/sessions/{session_id}/answers/answer-1/save"]
    assert len(chat_http) == 1
    saved = list((tmp_path / "saved").glob("*.txt"))
    assert len(saved) == 1
    assert "답변: 새로고침 뒤 저장할 답변" in saved[0].read_text(encoding="utf-8-sig")

    after_save = AppTest.from_file(APP, default_timeout=10).run()
    assert not after_save.exception
    assert any("파일 준비 완료" in item.value for item in after_save.success)
    assert any(item.proto.label == "TXT 다운로드" for item in after_save.get("link_button"))
    assert not any(item.label == "TXT 파일 만들기" for item in after_save.button)
    assert len(chat_http.exports) == 1
    assert len(chat_http) == 1


@pytest.mark.parametrize("failure_code", ["evidence_insufficient", "model_output_invalid"])
def test_failed_question_can_be_retried_once_without_retyping(chat_http, failure_code):
    chat_http.failure_code = failure_code
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("공식 문서에서 append의 반환값을 설명해줘").run())
    assert not app.exception
    assert app.error

    retry = next(item for item in app.button if item.label == "같은 질문 다시 보내기")
    chat_http.failure_code = None
    finish_chat_response(retry.click().run())

    assert not app.exception
    assert [item["query"] for item in chat_http] == ["공식 문서에서 append의 반환값을 설명해줘"] * 2
    assert any("답변: 공식 문서에서 append의 반환값을 설명해줘" in item.value for item in app.markdown)
    assert not any(item.label == "같은 질문 다시 보내기" for item in app.button)
    app.run()
    assert len(chat_http) == 2


def test_only_latest_failed_question_offers_recovery(chat_http):
    chat_http.failure_code = "evidence_insufficient"
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("첫 번째 실패 질문").run())
    finish_chat_response(app.chat_input[0].set_value("두 번째 실패 질문").run())

    retries = [item for item in app.button if item.label == "같은 질문 다시 보내기"]
    assert len(retries) == 1
    retries[0].click().run()
    assert not app.exception
    assert [item["query"] for item in chat_http] == ["첫 번째 실패 질문", "두 번째 실패 질문", "두 번째 실패 질문"]


def test_editing_failed_question_prefills_without_sending_then_submits_the_revision_once(chat_http):
    chat_http.failure_code = "evidence_insufficient"
    app = AppTest.from_file(APP, default_timeout=10).run()
    original = "Python append와 extend를 설명해줘"
    finish_chat_response(app.chat_input[0].set_value(original).run())

    next(item for item in app.button if item.label == "질문 수정").click().run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == [original]
    assert app.chat_input[0].proto.value == original
    chat_http.failure_code = None
    revised = "Python 공식 문서에서 list.append만 설명해줘"
    finish_chat_response(app.chat_input[0].set_value(revised).run())

    assert not app.exception
    assert [item["query"] for item in chat_http] == [original, revised]
    assert any("답변: " + revised in item.value for item in app.markdown)
    assert not any(item.label == "질문 수정" for item in app.button)
    app.run()
    assert len(chat_http) == 2


def _recovery_view(messages, *, allow_edit=True, actions_disabled=False):
    app = AppTest.from_string('''
import streamlit as st
from src.app.web.streamlit_chat import render_chat_history

def retry(prompt):
    st.session_state["events"].append(("retry", prompt))

def edit(prompt):
    st.session_state["events"].append(("edit", prompt))

render_chat_history(st.session_state["messages"], "http://localhost:8000", on_retry=retry,
                    on_edit=edit if st.session_state["allow_edit"] else None,
                    actions_disabled=st.session_state["actions_disabled"])
''', default_timeout=10)
    app.session_state["messages"] = messages
    app.session_state["events"] = []
    app.session_state["allow_edit"] = allow_edit
    app.session_state["actions_disabled"] = actions_disabled
    app.run()
    assert not app.exception
    return app


def test_interrupted_response_offers_edit_without_replaying_uncertain_request():
    prompt = "정리한 내용을 Slack으로 보내줘"
    app = _recovery_view([
        {"role": "user", "content": prompt},
        {"role": "assistant", "result": None, "transport_errors": ["연결이 끊겼습니다."]},
    ])

    assert not any(item.label == "같은 질문 다시 보내기" for item in app.button)
    assert any("완료 여부를 확인" in item.value for item in app.caption)
    next(item for item in app.button if item.label == "질문 수정").click().run()
    assert app.session_state["events"] == [("edit", prompt)]
    app.run()
    assert app.session_state["events"] == [("edit", prompt)]


@pytest.mark.parametrize(("scenario", "expected_guidance"), [
    ("latest", True),
    ("followed_by_answer", False),
    ("read_only", False),
    ("busy", False),
])
def test_interrupted_card_only_guides_to_an_available_edit_action(scenario, expected_guidance):
    from src.core.contracts.outcome import TurnResult

    messages = [
        {"role": "user", "content": "중단한 질문"},
        {"role": "assistant", "result": None, "interrupted": True},
    ]
    if scenario == "followed_by_answer":
        messages.extend([
            {"role": "user", "content": "이어서 보낸 질문"},
            {"role": "assistant", "result": TurnResult(
                status="completed", response=finalize_answer(text_document("완료한 답변"), []),
            )},
        ])

    app = _recovery_view(messages, allow_edit=scenario != "read_only", actions_disabled=scenario == "busy")

    assert any("응답 수신이 중단" in item.value for item in app.warning)
    guidance = [item.value for item in app.caption if "아래 ‘질문 수정’" in item.value]
    assert bool(guidance) is expected_guidance
    enabled_edits = [item for item in app.button if item.label == "질문 수정" and not item.disabled]
    assert bool(enabled_edits) is expected_guidance
    if expected_guidance:
        enabled_edits[0].click().run()
        assert app.session_state["events"] == [("edit", "중단한 질문")]


@pytest.mark.parametrize("delivery_status", ["sent", "unknown"])
def test_partial_answer_with_slack_receipt_does_not_offer_whole_request_replay(delivery_status):
    from src.core.contracts.outcome import TurnResult
    from src.core.llm_errors import make_problem
    from src.core.slack_contract import SlackDelivery
    from src.runtime.nodes.actions.receipts import build_slack_receipt

    delivery = SlackDelivery.model_validate({
        "intent": {"state": "explicit", "selector": {"kind": "channel", "value": "C12345678"}},
        "selection": {"request_id": "request-1", "source": "user_text",
                      "selector": {"kind": "channel", "value": "C12345678"}},
        "target": {"channel_id": "C12345678"}, "status": delivery_status,
        "message_ts": "1234.5678" if delivery_status == "sent" else None,
        "failure": None if delivery_status == "sent" else {
            "stage": "send", "code": "delivery_unknown", "message": "전달 여부를 확인해 주세요.",
            "next_action": "verify_delivery",
        },
    })
    answer = finalize_answer(text_document("확인된 답변"), [])
    result = TurnResult(
        status="partial", problem=make_problem("evidence_insufficient", "validation"),
        response=answer.model_copy(update={"actions": [build_slack_receipt(delivery)]}),
    )
    app = _recovery_view([
        {"role": "user", "content": "공식 문서를 정리하고 Slack으로 보내줘"},
        {"role": "assistant", "result": result},
    ])

    assert not any(item.label in {"같은 질문 다시 보내기", "질문 수정"} for item in app.button)
    assert app.session_state["events"] == []


@pytest.mark.parametrize("failure", ["network", "http"])
def test_save_failure_preserves_answer_and_retries_same_target_once(chat_http, failure):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("원래 저장할 답변").run())
    before = app.session_state["conversations"][app.session_state["active_conversation_id"]].messages[-1]["result"]
    chat_http.export_failure = failure
    next(item for item in app.button if item.label == "TXT 파일 만들기").click().run()
    assert not app.exception
    assert app.error
    assert any("답변: 원래 저장할 답변" in item.value for item in app.markdown)
    assert len(chat_http) == 1
    assert len(chat_http.exports) == 1
    chat_http.export_failure = None
    next(item for item in app.button if item.label == "저장 다시 시도").click().run()
    assert not app.exception
    assert not app.error
    assert any(item.proto.label == "TXT 다운로드" for item in app.get("link_button"))
    conversation = app.session_state["conversations"][app.session_state["active_conversation_id"]]
    assert conversation.messages[-1]["result"] == before
    assert chat_http.exports == [chat_http.exports[0]] * 2
    app.run()
    assert len(chat_http.exports) == 2
    assert len(chat_http) == 1


def test_natural_save_failure_shows_only_latest_success_after_direct_retry(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("원래 답변").run())
    chat_http.natural_save_failure = True
    finish_chat_response(app.chat_input[0].set_value("방금 답변을 출처와 함께 txt 파일로 저장해줘").run())
    assert not app.exception
    assert any("파일 저장 실패" in item.value for item in app.error)
    assert not any("server-private-path" in item.value for item in app.error)
    assert not any(item.label in {"같은 질문 다시 보내기", "질문 수정"} for item in app.button)
    next(item for item in app.button if item.label == "저장 다시 시도").click().run()
    assert not app.exception
    assert not app.error
    assert not any("파일을 만들지 못했습니다" in item.value for item in app.warning)
    assert any("파일 준비 완료" in item.value for item in app.success)
    assert len(chat_http.exports) == 1
