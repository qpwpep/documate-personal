"""Observable chat behavior with only the HTTP transport replaced."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.answer_export import build_answer_export_operation
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions.receipts import build_save_receipt


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
            manifest = {"epoch": "epoch-" + path.split("/")[2], "revision": 0, "files": submissions.upload_files}
            result._content = json.dumps(manifest).encode()
        elif request.method == "POST" and path == "/agent/stream":
            payload = json.loads(request.body)
            submissions.append(payload)
            answer = finalize_answer(text_document("답변: " + payload["query"]), [])
            answer_id = f"answer-{len(submissions)}"
            answers[answer_id] = answer
            final = {
                "status": "completed", "response": answer.model_dump(mode="json"),
                "answer_id": answer_id,
                "upload_manifest": {"epoch": "epoch-" + payload["session_id"], "revision": 0, "files": []},
                "request_id": f"test-request-{len(submissions)}",
            }
            if submissions.failure_code:
                from src.core.llm_errors import make_problem

                final.update(
                    status="failed", response=None, answer_id=None,
                    problem=make_problem(submissions.failure_code, "pre_synthesis_validation").model_dump(mode="json"),
                )
            if submissions.natural_save_failure:
                source = answers["answer-1"]
                operation = build_answer_export_operation(source, session_id=payload["session_id"])
                receipt = build_save_receipt({"status": "error", "error_code": "write_failed", "error": "Permission denied: server-private-path/.save.lock"}, operation=operation)
                final.update(status="partial", answer_id="answer-1",
                             response=source.model_copy(update={"actions": [receipt]}).model_dump(mode="json"),
                             problem={"code": "save_failed", "stage": "action_postprocess", "message": "파일을 만들지 못했습니다.", "next_action": "retry_later"})
            result.headers["Content-Type"] = "text/event-stream; charset=utf-8"
            result.raw = HTTPBytes(("event: final_response\ndata: " + json.dumps(final, ensure_ascii=False) + "\n\n").encode())
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
    return submissions


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
    app.chat_input[0].set_value(prompt).run()
    app.run()
    assert [item["query"] for item in chat_http] == [prompt]
    assert any("답변: " + prompt in item.value for item in app.markdown)


def test_submitted_input_preserves_boundary_whitespace_for_exact_copy(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    prompt = "  아래 원문을 공백과 개행까지 그대로 복사해줘:\n    value = 1\n\n"

    app.chat_input[0].set_value(prompt).run()
    app.run()

    assert not app.exception
    assert [submission["query"] for submission in chat_http] == [prompt]


def test_examples_are_stable_and_require_confirmed_attachments(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    initial = [item.label for item in app.button if "quick_prompt_" in (item.key or "")]
    assert len(initial) == 4
    assert not any(word in prompt for prompt in initial for word in ("첨부", "업로드", "방금", "Slack", "저장"))
    app.button(key="documate_new_chat").click().run()
    assert [item.label for item in app.button if "quick_prompt_" in (item.key or "")] == initial

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
    app.chat_input[0].set_value("NumPy reshape를 설명해줘").run()

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
    app.button(key="documate_send_saved_upload_prompt").click().run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == ["보류했던 질문"]
    assert not any("아직 보내지 않은 질문" in item.value for item in app.info)
    assert not any(item.label == "보류한 질문 보내기" for item in app.button)
    app.run()
    assert len(chat_http) == 1


def test_completed_answer_offers_direct_txt_save(chat_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.chat_input[0].set_value("NumPy reshape를 설명해줘").run()

    assert not app.exception
    assert any(item.label == "TXT 저장" for item in app.button)


def test_prepared_txt_offers_download_instead_of_saving_the_same_answer_again(chat_http, tmp_path):
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.chat_input[0].set_value("내려받을 답변").run()
    next(item for item in app.button if item.label == "TXT 저장").click().run()

    assert not app.exception
    assert any("파일 준비 완료" in item.value and "내려받으세요" in item.value for item in app.success)
    downloads = app.get("link_button")
    assert len(downloads) == 1
    assert downloads[0].proto.label == "TXT 다운로드"
    assert downloads[0].proto.type == "primary"
    saved = list((tmp_path / "saved").glob("*.txt"))
    assert len(saved) == 1
    assert downloads[0].proto.url.endswith("/download/" + saved[0].name)
    assert not any(item.label in {"TXT 저장", "저장 다시 시도"} for item in app.button)
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
    label = "TXT 다시 만들기" if verified else "TXT 저장"
    next(item for item in app.button if item.label == label).click().run()
    assert app.session_state["save_requests"] == ["original-answer"]


def test_reloaded_answer_can_be_saved_with_the_original_answer_id_without_regeneration(chat_http, tmp_path):
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.chat_input[0].set_value("새로고침 뒤 저장할 답변").run()
    session_id = chat_http[0]["session_id"]

    reloaded = AppTest.from_file(APP, default_timeout=10).run()
    assert not reloaded.exception
    assert len(chat_http) == 1
    assert not chat_http.exports
    next(item for item in reloaded.button if item.label == "TXT 저장").click().run()

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
    assert not any(item.label == "TXT 저장" for item in after_save.button)
    assert len(chat_http.exports) == 1
    assert len(chat_http) == 1


@pytest.mark.parametrize("failure_code", ["evidence_insufficient", "model_output_invalid"])
def test_failed_question_can_be_retried_once_without_retyping(chat_http, failure_code):
    chat_http.failure_code = failure_code
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.chat_input[0].set_value("공식 문서에서 append의 반환값을 설명해줘").run()
    assert not app.exception
    assert app.error

    retry = next(item for item in app.button if item.label == "같은 질문 다시 보내기")
    chat_http.failure_code = None
    retry.click().run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == ["공식 문서에서 append의 반환값을 설명해줘"] * 2
    assert any("답변: 공식 문서에서 append의 반환값을 설명해줘" in item.value for item in app.markdown)
    assert not any(item.label == "같은 질문 다시 보내기" for item in app.button)
    app.run()
    assert len(chat_http) == 2


def test_only_latest_failed_question_offers_recovery(chat_http):
    chat_http.failure_code = "evidence_insufficient"
    app = AppTest.from_file(APP, default_timeout=10).run()
    app.chat_input[0].set_value("첫 번째 실패 질문").run()
    app.chat_input[0].set_value("두 번째 실패 질문").run()

    retries = [item for item in app.button if item.label == "같은 질문 다시 보내기"]
    assert len(retries) == 1
    retries[0].click().run()
    assert not app.exception
    assert [item["query"] for item in chat_http] == ["첫 번째 실패 질문", "두 번째 실패 질문", "두 번째 실패 질문"]


def test_editing_failed_question_prefills_without_sending_then_submits_the_revision_once(chat_http):
    chat_http.failure_code = "evidence_insufficient"
    app = AppTest.from_file(APP, default_timeout=10).run()
    original = "Python append와 extend를 설명해줘"
    app.chat_input[0].set_value(original).run()

    next(item for item in app.button if item.label == "질문 수정").click().run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == [original]
    assert app.chat_input[0].proto.value == original
    chat_http.failure_code = None
    revised = "Python 공식 문서에서 list.append만 설명해줘"
    app.chat_input[0].set_value(revised).run()

    assert not app.exception
    assert [item["query"] for item in chat_http] == [original, revised]
    assert any("답변: " + revised in item.value for item in app.markdown)
    assert not any(item.label == "질문 수정" for item in app.button)
    app.run()
    assert len(chat_http) == 2


def _recovery_view(messages):
    app = AppTest.from_string('''
import streamlit as st
from src.app.web.streamlit_chat import render_chat_history

def retry(prompt):
    st.session_state["events"].append(("retry", prompt))

def edit(prompt):
    st.session_state["events"].append(("edit", prompt))

render_chat_history(st.session_state["messages"], "http://localhost:8000", on_retry=retry, on_edit=edit)
''', default_timeout=10)
    app.session_state["messages"] = messages
    app.session_state["events"] = []
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
    app.chat_input[0].set_value("원래 저장할 답변").run()
    before = app.session_state["conversations"][app.session_state["active_conversation_id"]].messages[-1]["result"]
    chat_http.export_failure = failure
    next(item for item in app.button if item.label == "TXT 저장").click().run()
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
    app.chat_input[0].set_value("원래 답변").run()
    chat_http.natural_save_failure = True
    app.chat_input[0].set_value("방금 답변을 출처와 함께 txt 파일로 저장해줘").run()
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
