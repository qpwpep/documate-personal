"""Conversation navigation through the real Streamlit UI and HTTP boundary."""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from streamlit.testing.v1 import AppTest

from src.core.answer_schema import finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence
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


@pytest.fixture
def conversation_http(monkeypatch, tmp_path, browser_storage):
    from src.app.web import streamlit_state

    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path / "uploads")
    observed = {"requests": [], "manifests": {}, "unavailable": set(), "reads": [], "syncs": []}

    def send(_adapter, request, **_kwargs):
        path = urlparse(request.url).path
        response = requests.Response()
        response.status_code = 200
        response.request = request
        response.url = request.url
        if request.method == "GET" and path == "/slack/default-recipient":
            response._content = json.dumps({"selector": None, "failure": None}).encode()
        elif request.method == "GET" and path.endswith("/uploads"):
            session_id = path.split("/")[2]
            observed["reads"].append(session_id)
            if session_id in observed["unavailable"]:
                raise requests.ConnectionError("test boundary offline")
            manifest = observed["manifests"].setdefault(session_id, {
                "epoch": "epoch-" + session_id, "revision": 0, "files": [],
            })
            response._content = json.dumps(manifest).encode()
        elif request.method == "POST" and path.endswith("/uploads/sync"):
            session_id = path.split("/")[2]
            payload = json.loads(request.body)
            observed["syncs"].append((session_id, payload))
            manifest = observed["manifests"][session_id]
            assert payload["clear"] is True
            assert payload["epoch"] == manifest["epoch"]
            assert payload["expected_revision"] == manifest["revision"]
            manifest.update({"revision": manifest["revision"] + 1, "files": []})
            response._content = json.dumps(manifest).encode()
        elif request.method == "POST" and path == "/agent/stream":
            payload = json.loads(request.body)
            observed["requests"].append(payload)
            source_text = "retained source for " + payload["query"]
            snapshot = build_snapshot(
                source_uri="upload:///retained.py", title="retained.py", media_type="text/x-python",
                source_type="upload", content=source_text, parser="test", parser_version="1",
            )
            evidence = build_evidence(
                snapshot=snapshot,
                element=DocumentElement(
                    element_id="line", kind="code", text=source_text, language="python",
                    anchors=[SourceAnchor(kind="code", line_start=1, line_end=1)],
                ),
            )
            answer = finalize_answer(text_document("답변: " + payload["query"], basis="source", refs=[evidence.id]), [evidence])
            final = {
                "status": "completed", "response": answer.model_dump(mode="json"),
                "answer_id": f"answer-{len(observed['requests'])}",
                "upload_manifest": observed["manifests"][payload["session_id"]],
                "request_id": f"conversation-request-{len(observed['requests'])}",
            }
            response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
            response.raw = HTTPBytes(("event: final_response\ndata: " + json.dumps(final, ensure_ascii=False) + "\n\n").encode())
        else:
            raise AssertionError(f"Unexpected HTTP request: {request.method} {path}")
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
    return observed


def _picker(app):
    return next(item for item in app.selectbox if item.label == "대화 선택")


def _slack_kind(app):
    return next(item for item in app.radio if item.label == "수신자 유형")


def _first_id(observed):
    return next(iter(observed["manifests"]))


def test_recent_conversation_buttons_switch_without_sending_or_losing_drafts(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("첫 대화 질문").run())
    first = _picker(app).value
    conversation = app.session_state["conversations"][first]
    conversation.draft.record("첫 대화의 작성 초안", revision=conversation.draft.revision)
    app.button(key="documate_new_chat").click().run()
    second = _picker(app).value
    finish_chat_response(app.chat_input[0].set_value("두 번째 대화 질문").run())

    current = next(item for item in app.button if item.label == "현재 · 2. 두 번째 대화 질문")
    assert current.disabled
    next(item for item in app.button if item.label == "1. 첫 대화 질문").click().run()

    assert not app.exception
    assert _picker(app).value == first
    assert app.chat_input[0].proto.value == "첫 대화의 작성 초안"
    assert any("답변: 첫 대화 질문" in item.value for item in app.markdown)
    assert not any("답변: 두 번째 대화 질문" in item.value for item in app.markdown)
    assert next(item for item in app.button if item.label == "현재 · 1. 첫 대화 질문").disabled
    next(item for item in app.button if item.label == "2. 두 번째 대화 질문").click().run()
    assert _picker(app).value == second
    assert [item["query"] for item in conversation_http["requests"]] == ["첫 대화 질문", "두 번째 대화 질문"]


def test_reload_restores_stored_answer_without_optional_source_location(conversation_http, browser_storage):
    conversation_id = "c3fe22c2-ef6b-47a5-a352-2996ab2a0c05"
    # Fixed body and hash in the schema before optional source positions were added.
    # Keep both fixed so the current serializer cannot conceal a revision change.
    content_hash = "df8ee3a420f2177db595b2dea63fc628814c375b0fadd6040041e45d9e30ea1c"
    browser_storage.snapshot = json.dumps({
        "version": 1, "active_conversation_id": conversation_id,
        "conversations": [{
            "id": conversation_id,
            "messages": [
                {"role": "user", "content": "Stored question"},
                {"role": "assistant", "result": {
                    "status": "completed", "answer_id": "stored-answer", "response": {
                        "content": {"blocks": [{"type": "paragraph", "content": [{
                            "text": "Previously saved answer", "basis": "interaction", "refs": [],
                        }]}]},
                        "citations": [], "checks": [{
                            "unit_id": "b0.content.0", "reference_status": "not_required",
                            "support_status": "not_evaluated", "issues": [],
                        }],
                        "issues": [], "actions": [], "content_hash": content_hash,
                        "retrieval_required": False, "semantic_validation": None,
                    },
                }},
            ],
            "draft": {"text": "아직 보내지 않은 질문", "revision": 1},
            "confirmed_epoch": "epoch-" + conversation_id,
        }],
    }, ensure_ascii=False)

    restored = AppTest.from_file(APP, default_timeout=10).run()

    assert not restored.exception
    assert any("Previously saved answer" in item.value for item in restored.markdown)
    assert _picker(restored).value == conversation_id
    assert restored.chat_input[0].proto.value == "아직 보내지 않은 질문"
    saved = json.loads(browser_storage.snapshot)
    response = saved["conversations"][0]["messages"][1]["result"]["response"]
    assert response["content_hash"] == content_hash
    assert not conversation_http["requests"]


def test_reload_restores_answer_sources_attachments_and_unsent_draft(conversation_http, browser_storage):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("새로고침 후에도 확인할 답변").run())
    first = _first_id(conversation_http)
    conversation_http["manifests"][first].update({"revision": 1, "files": [{
        "file_id": "retained-file", "name": "retained-after-refresh.py", "size_bytes": 20,
        "content_hash": "sha256:" + "a" * 64, "source_uri": "upload:///retained-after-refresh.py",
    }]})
    app.button(key="documate_refresh_uploads").click().run()
    draft = "  아직 보내지 않은 질문\n두 번째 줄  "
    browser_storage.type_draft(first, draft)

    reloaded = AppTest.from_file(APP, default_timeout=10).run()

    assert not reloaded.exception
    assert any("답변: 새로고침 후에도 확인할 답변" in item.value for item in reloaded.markdown)
    assert any("retained source for 새로고침 후에도 확인할 답변" in item.value for item in reloaded.code)
    assert any("retained-after-refresh.py" in item.value for item in reloaded.markdown)
    assert reloaded.chat_input[0].proto.value == draft
    assert _picker(reloaded).value == first
    assert [item["query"] for item in conversation_http["requests"]] == ["새로고침 후에도 확인할 답변"]
    assert not conversation_http["syncs"]

    finish_chat_response(reloaded.chat_input[0].set_value("기존 첨부로 계속 질문").run())
    assert not reloaded.exception
    assert conversation_http["requests"][-1]["session_id"] == first
    assert conversation_http["requests"][-1]["uploads"] == {"epoch": "epoch-" + first, "revision": 1}
    assert len(conversation_http["requests"]) == 2


def test_reload_preserves_selected_conversation_and_separate_drafts_without_replaying_sent_text(
    conversation_http, browser_storage,
):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("첫 대화 답변").run())
    first = _picker(app).value
    app.button(key="documate_new_chat").click().run()
    finish_chat_response(app.chat_input[0].set_value("두 번째 대화 답변").run())
    second = _picker(app).value
    _picker(app).set_value(first).run()
    browser_storage.type_draft(first, "  첫 대화의 초안\n다음 줄  ")
    browser_storage.type_draft(second, "두 번째 대화의 초안")

    reloaded = AppTest.from_file(APP, default_timeout=10).run()

    assert not reloaded.exception
    assert _picker(reloaded).value == first
    assert any("답변: 첫 대화 답변" in item.value for item in reloaded.markdown)
    assert not any("두 번째 대화 답변" in item.value for item in reloaded.markdown)
    assert reloaded.chat_input[0].proto.value == "  첫 대화의 초안\n다음 줄  "
    _picker(reloaded).set_value(second).run()
    assert not reloaded.exception
    assert any("답변: 두 번째 대화 답변" in item.value for item in reloaded.markdown)
    assert not any("첫 대화 답변" in item.value for item in reloaded.markdown)
    assert reloaded.chat_input[0].proto.value == "두 번째 대화의 초안"
    assert len(conversation_http["requests"]) == 2

    finish_chat_response(reloaded.chat_input[0].set_value("두 번째 대화의 초안").run())
    after_send = AppTest.from_file(APP, default_timeout=10).run()
    assert not after_send.exception
    assert _picker(after_send).value == second
    assert after_send.chat_input[0].proto.value == ""
    assert any("답변: 두 번째 대화의 초안" in item.value for item in after_send.markdown)
    assert [item["query"] for item in conversation_http["requests"]] == [
        "첫 대화 답변", "두 번째 대화 답변", "두 번째 대화의 초안",
    ]


@pytest.mark.parametrize("server_state", ["expired", "offline"])
def test_reload_keeps_records_and_draft_when_server_session_cannot_be_resumed(
    conversation_http, browser_storage, server_state,
):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("서버 연결이 끊겨도 보존할 답변").run())
    first = _picker(app).value
    conversation_http["manifests"][first].update({"revision": 1, "files": [{
        "file_id": "retained-file", "name": "keep-visible.py", "size_bytes": 20,
        "content_hash": "sha256:" + "b" * 64, "source_uri": "upload:///keep-visible.py",
    }]})
    app.button(key="documate_refresh_uploads").click().run()
    browser_storage.type_draft(first, "복원 후 계속 작성할 질문")
    if server_state == "expired":
        conversation_http["manifests"][first] = {"epoch": "replacement-session", "revision": 0, "files": []}
    else:
        conversation_http["unavailable"].add(first)

    reloaded = AppTest.from_file(APP, default_timeout=10).run()

    assert not reloaded.exception
    assert any("답변: 서버 연결이 끊겨도 보존할 답변" in item.value for item in reloaded.markdown)
    assert any("retained source for 서버 연결이 끊겨도 보존할 답변" in item.value for item in reloaded.code)
    assert any("keep-visible.py" in item.value for item in reloaded.markdown)
    assert any("복원 후 계속 작성할 질문" in item.value for item in reloaded.info)
    assert not reloaded.chat_input or reloaded.chat_input[0].disabled
    assert any(item.label == "TXT 파일 만들기" and item.disabled for item in reloaded.button)
    assert not conversation_http["syncs"]
    assert len(conversation_http["requests"]) == 1

    if server_state == "expired":
        assert any("읽기 전용" in item.value for item in reloaded.warning)
    else:
        assert any("대화의 서버 상태를 확인하지 못했습니다" in item.value for item in reloaded.warning)
        conversation_http["unavailable"].remove(first)
        next(item for item in reloaded.button if item.label == "대화 연결 다시 확인").click().run()
        assert not reloaded.exception
        assert reloaded.chat_input[0].proto.value == "복원 후 계속 작성할 질문"
        assert not reloaded.chat_input[0].disabled
        assert any("keep-visible.py" in item.value for item in reloaded.markdown)
        assert len(conversation_http["requests"]) == 1


def test_unreadable_browser_storage_explains_recovery_and_allows_a_new_question(conversation_http, browser_storage):
    browser_storage.snapshot = "{interrupted browser storage"

    app = AppTest.from_file(APP, default_timeout=10).run()

    assert not app.exception
    assert any("저장된 대화를 복원하지 못했습니다" in item.value for item in app.warning)
    assert app.chat_input and not app.chat_input[0].disabled
    assert not conversation_http["requests"]
    finish_chat_response(app.chat_input[0].set_value("새 대화에서 이어가기").run())
    assert not app.exception
    assert any("답변: 새 대화에서 이어가기" in item.value for item in app.markdown)
    assert [item["query"] for item in conversation_http["requests"]] == ["새 대화에서 이어가기"]


def test_new_chat_preserves_previous_messages_and_isolates_slack_drafts(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    _slack_kind(app).set_value("채널").run()
    next(item for item in app.text_input if item.label == "Slack 수신자").set_value("C12345678").run()
    finish_chat_response(app.chat_input[0].set_value("첫 대화 질문").run())
    assert not app.exception
    first = _first_id(conversation_http)
    app.button(key="documate_new_chat").click().run()

    assert not app.exception
    assert not any("답변: 첫 대화 질문" in item.value for item in app.markdown)
    assert _slack_kind(app).value == "미지정"
    finish_chat_response(app.chat_input[0].set_value("두 번째 대화 질문").run())
    _picker(app).set_value(first).run()

    assert not app.exception
    assert any("답변: 첫 대화 질문" in item.value for item in app.markdown)
    assert any("retained source for 첫 대화 질문" in item.value for item in app.code)
    assert not any("두 번째 대화 질문" in item.value for item in app.markdown)
    assert _slack_kind(app).value == "채널"
    assert next(item for item in app.text_input if item.label == "Slack 수신자").value == "C12345678"
    assert [item["query"] for item in conversation_http["requests"]] == ["첫 대화 질문", "두 번째 대화 질문"]
    assert len({item["session_id"] for item in conversation_http["requests"]}) == 2


def test_expired_conversation_keeps_visible_history_without_replaying_requests(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("보존할 답변").run())
    first = _first_id(conversation_http)
    app.button(key="documate_new_chat").click().run()
    conversation_http["manifests"][first]["epoch"] = "replacement-server-session"
    _picker(app).set_value(first).run()

    assert not app.exception
    assert any("답변: 보존할 답변" in item.value for item in app.markdown)
    assert any("retained source for 보존할 답변" in item.value for item in app.code)
    assert any("읽기 전용" in item.value for item in app.warning)
    assert not app.chat_input or app.chat_input[0].disabled
    assert any(item.label == "TXT 파일 만들기" and item.disabled for item in app.button)
    app.run()
    assert [item["query"] for item in conversation_http["requests"]] == ["보존할 답변"]


def test_failed_session_validation_keeps_history_and_offers_retry(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("연결 실패에도 남길 답변").run())
    first = _first_id(conversation_http)
    app.button(key="documate_new_chat").click().run()
    conversation_http["unavailable"].add(first)
    _picker(app).set_value(first).run()

    assert not app.exception
    assert any("답변: 연결 실패에도 남길 답변" in item.value for item in app.markdown)
    assert not app.chat_input or app.chat_input[0].disabled
    retry = next(item for item in app.button if item.label == "대화 연결 다시 확인")
    conversation_http["unavailable"].remove(first)
    retry.click().run()
    assert not app.exception
    assert app.chat_input and not app.chat_input[0].disabled
    assert len(conversation_http["requests"]) == 1


def test_reconnecting_restores_unsent_draft_without_sending_it(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    first = _first_id(conversation_http)
    conversation = app.session_state["conversations"][first]
    conversation.draft.record("연결 실패 전 작성한 질문", revision=0)
    conversation_http["unavailable"].add(first)
    next(item for item in app.button if item.label == "첨부 목록 새로고침").click().run()
    assert not app.exception
    assert not app.chat_input

    conversation_http["unavailable"].remove(first)
    next(item for item in app.button if item.label == "대화 연결 다시 확인").click().run()
    assert not app.exception
    assert app.chat_input[0].proto.value == "연결 실패 전 작성한 질문"
    assert not conversation_http["requests"]


def test_held_question_stays_in_its_conversation_until_explicit_send(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    first = _first_id(conversation_http)
    # Setup the recoverable draft left by an upload failure; subsequent actions use the UI.
    app.session_state["conversations"][first].saved_prompt = "이 대화에만 보류한 질문"
    app.run()
    app.button(key="documate_new_chat").click().run()

    assert not app.exception
    assert not any("이 대화에만 보류한 질문" in item.value for item in app.info)
    _picker(app).set_value(first).run()
    assert any("이 대화에만 보류한 질문" in item.value for item in app.info)
    assert not conversation_http["requests"]
    finish_chat_response(next(item for item in app.button if item.label == "보류한 질문 보내기").click().run())
    assert not app.exception
    assert [item["query"] for item in conversation_http["requests"]] == ["이 대화에만 보류한 질문"]
    assert not any("아직 보내지 않은 질문" in item.value for item in app.info)
    app.run()
    assert len(conversation_http["requests"]) == 1


def test_returning_to_failed_upload_requires_confirmation_without_replay(conversation_http):
    from src.app.web.streamlit_state import PendingUpload

    app = AppTest.from_file(APP, default_timeout=10).run()
    first = _first_id(conversation_http)
    conversation = app.session_state["conversations"][first]
    conversation.pending_upload = PendingUpload(
        base_manifest=conversation.client.manifest, clear=True, prompt="첨부 후 질문",
        attempted=True, had_failure=True, error="첨부 서버 연결 실패",
    )
    app.button(key="documate_new_chat").click().run()
    assert not app.exception
    assert not any("첨부 변경을 완료하지 못했습니다" in item.value for item in app.markdown)
    _picker(app).set_value(first).run()

    assert not app.exception
    assert any("첨부 변경을 완료하지 못했습니다" in item.value for item in app.markdown)
    assert any(item.label == "첨부 변경 다시 시도" for item in app.button)
    assert not conversation_http["syncs"]
    assert not conversation_http["requests"]

    next(item for item in app.button if item.label == "파일 다시 선택").click().run()

    assert not app.exception
    assert conversation.pending_upload is None
    assert conversation.saved_prompt is None
    assert app.chat_input[0].proto.value == "첨부 후 질문"
    assert not app.chat_input[0].disabled
    assert app.chat_input[0].proto.accept_file
    assert not conversation_http["syncs"]
    assert not conversation_http["requests"]


def test_confirmed_attachments_follow_the_selected_conversation(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    first = _first_id(conversation_http)
    conversation_http["manifests"][first].update({"revision": 1, "files": [{
        "file_id": "retained-file", "name": "first-conversation.py", "size_bytes": 20,
        "content_hash": "sha256:" + "a" * 64, "source_uri": "upload:///first-conversation.py",
    }]})
    app.button(key="documate_refresh_uploads").click().run()
    assert any("first-conversation.py" in item.value for item in app.markdown)
    app.button(key="documate_new_chat").click().run()
    assert not any("first-conversation.py" in item.value for item in app.markdown)
    _picker(app).set_value(first).run()
    assert not app.exception
    assert any("first-conversation.py" in item.value for item in app.markdown)
    finish_chat_response(app.chat_input[0].set_value("이 첨부를 사용해줘").run())
    assert conversation_http["requests"][-1]["session_id"] == first
    assert conversation_http["requests"][-1]["uploads"] == {"epoch": "epoch-" + first, "revision": 1}


def test_expiry_before_submission_keeps_draft_and_does_not_start_new_server_work(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    finish_chat_response(app.chat_input[0].set_value("만료 전 답변").run())
    first = _first_id(conversation_http)
    conversation_http["manifests"][first]["epoch"] = "new-server-incarnation"
    finish_chat_response(app.chat_input[0].set_value("만료 직후 새 질문").run())

    assert not app.exception
    assert any("답변: 만료 전 답변" in item.value for item in app.markdown)
    assert any("읽기 전용" in item.value for item in app.warning)
    assert any("전송하지 않은 질문: 만료 직후 새 질문" in item.value for item in app.info)
    assert [item["query"] for item in conversation_http["requests"]] == ["만료 전 답변"]


def test_upload_preflight_failure_allows_navigation_and_requires_explicit_resume(conversation_http):
    app = AppTest.from_file(APP, default_timeout=10).run()
    first = _first_id(conversation_http)
    conversation_http["manifests"][first].update({"revision": 1, "files": [{
        "file_id": "pending-clear-file", "name": "keep-until-confirmed.py", "size_bytes": 20,
        "content_hash": "sha256:" + "b" * 64, "source_uri": "upload:///keep-until-confirmed.py",
    }]})
    app.button(key="documate_refresh_uploads").click().run()
    conversation_http["unavailable"].add(first)
    app.button(key="documate_clear_uploads").click().run()

    assert not app.exception
    assert any("대화의 서버 상태를 확인하지 못했습니다" in item.value for item in app.warning)
    assert any("keep-until-confirmed.py" in item.value for item in app.markdown)
    assert not app.button(key="documate_new_chat").disabled
    assert not _picker(app).disabled
    assert not conversation_http["syncs"]
    assert not conversation_http["requests"]

    app.button(key="documate_new_chat").click().run()
    assert not app.exception
    assert not any("keep-until-confirmed.py" in item.value for item in app.markdown)
    _picker(app).set_value(first).run()
    assert not app.exception
    assert any("keep-until-confirmed.py" in item.value for item in app.markdown)

    conversation_http["unavailable"].remove(first)
    next(item for item in app.button if item.label == "대화 연결 다시 확인").click().run()
    assert not app.exception
    assert any("첨부 변경 준비 중" in item.value for item in app.markdown)
    assert any("keep-until-confirmed.py" in item.value for item in app.markdown)
    assert not conversation_http["syncs"]
    apply_upload = next(item for item in app.button if item.label == "첨부 변경 적용")
    assert not apply_upload.disabled
    apply_upload.click().run()

    assert not app.exception
    assert not any("keep-until-confirmed.py" in item.value for item in app.markdown)
    assert not any("첨부 변경 준비 중" in item.value for item in app.markdown)
    assert len(conversation_http["syncs"]) == 1
    assert conversation_http["syncs"][0][0] == first
    assert not conversation_http["requests"]
    app.run()
    assert len(conversation_http["syncs"]) == 1
