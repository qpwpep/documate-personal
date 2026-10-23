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


@pytest.fixture
def chat_http(monkeypatch, tmp_path):
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
            manifest = {"epoch": "epoch-" + path.split("/")[2], "revision": 0, "files": []}
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
    assert any("파일 다운로드" in item.value for item in app.markdown)
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
    next(item for item in app.button if item.label == "저장 다시 시도").click().run()
    assert not app.exception
    assert not app.error
    assert not any("파일을 만들지 못했습니다" in item.value for item in app.warning)
    assert any("파일 저장 완료" in item.value for item in app.success)
    assert len(chat_http.exports) == 1
