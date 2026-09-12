"""Observable attachment UI state across the HTTP sync boundary."""

import json
from pathlib import Path
from types import SimpleNamespace

import requests
import pytest

from src.app.web import streamlit_app, streamlit_state
from src.app.web.streamlit_upload_handler import PendingUploadOperation, StagedUpload
from src.core.uploads import UploadFileInfo, UploadManifest
from tests.web.answer_fixtures import answer_response


def _manifest(revision=1, files=None):
    return UploadManifest(epoch="epoch-one", revision=revision, files=files or [])


def _confirmed_file():
    return UploadFileInfo(file_id="file-a", name="a.py", size_bytes=3,
                          content_hash="sha256:" + "a" * 64, source_uri="upload://session-one/file-a")


def _install_ui(monkeypatch, tmp_path, pending):
    previous_answer = {"role": "user", "content": "earlier question"}
    fake_st = SimpleNamespace(session_state={
        "session_id": "session-one", "upload_manifest": _manifest(files=[_confirmed_file()]),
        "pending_upload": pending, "messages": [previous_answer],
    })
    monkeypatch.setattr(streamlit_app, "st", fake_st)
    monkeypatch.setattr(streamlit_state, "st", fake_st)
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    return fake_st


def _responses(monkeypatch, statuses_and_payloads):
    calls = []
    responses = iter(statuses_and_payloads)

    def send(_session, method, url, **kwargs):
        calls.append({"method": method, "url": url, "payload": kwargs.get("json")})
        status, payload = next(responses)
        response = requests.Response()
        response.status_code = status
        response._content = json.dumps(payload).encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    return calls


def test_failed_batch_keeps_confirmed_files_and_retry_does_not_send_held_question(monkeypatch, tmp_path):
    """A failed sync preserves ready files; a successful retry leaves its question unsent."""
    pending = PendingUploadOperation(epoch="epoch-one", expected_revision=1, prompt="compare the files")
    fake_st = _install_ui(monkeypatch, tmp_path, pending)
    confirmed = streamlit_state.get_upload_manifest()
    payload = pending.request_payload()
    calls = _responses(monkeypatch, [
        (400, {"detail": {"code": "UPLOAD_INVALID", "message": "failed", "files": [{"name": "bad.py", "message": "invalid"}]}}),
        (200, {"manifest": _manifest(revision=2, files=[_confirmed_file()]).model_dump(), "changed": True, "unchanged_names": []}),
    ])

    assert streamlit_app.commit_pending_upload() is False
    assert streamlit_state.get_upload_manifest() == confirmed
    assert "bad.py" in pending.error
    assert "upload_followup_prompt" not in fake_st.session_state
    assert streamlit_app.commit_pending_upload() is True
    assert fake_st.session_state["upload_saved_prompt"] == "compare the files"
    assert "upload_followup_prompt" not in fake_st.session_state
    assert [call["payload"] for call in calls] == [payload, payload]
    assert streamlit_state.get_messages() == [{"role": "user", "content": "earlier question"}]


def test_conflicting_name_never_syncs_before_explicit_replacement(monkeypatch, tmp_path):
    """Preparing a conflicting file cannot replace the confirmed file without the button decision."""
    pending = PendingUploadOperation(epoch="epoch-one", expected_revision=1, files=[
        StagedUpload(path=str(tmp_path / "staged.py"), name="a.py", size_bytes=3,
                     content_hash="sha256:" + "b" * 64, conflicting_file_id="file-a"),
    ])
    _install_ui(monkeypatch, tmp_path, pending)
    calls = _responses(monkeypatch, [])
    assert streamlit_app.commit_pending_upload() is False
    assert calls == []
    assert pending.attempted is False


def test_stale_batch_refreshes_manifest_without_replaying_the_mutation(monkeypatch, tmp_path):
    """A revision conflict loads current server state and waits for another explicit review."""
    pending = PendingUploadOperation(epoch="epoch-one", expected_revision=1, clear=True)
    _install_ui(monkeypatch, tmp_path, pending)
    fresh = _manifest(revision=3)
    calls = _responses(monkeypatch, [
        (409, {"detail": {"code": "UPLOAD_REVISION_CONFLICT", "message": "changed"}}),
        (200, fresh.model_dump()),
    ])
    assert streamlit_app.commit_pending_upload() is False
    assert streamlit_state.get_upload_manifest() == fresh
    assert pending.needs_refresh_review is True
    assert [call["method"] for call in calls] == ["post", "get"]


def test_sidebar_clear_button_removes_attachments_and_preserves_visible_conversation(monkeypatch, tmp_path):
    """The actual Streamlit clear action commits an empty set while keeping chat history."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    _responses(monkeypatch, [
        (200, _manifest(files=[_confirmed_file()]).model_dump()),
        (200, {"manifest": _manifest(revision=2).model_dump(), "changed": True, "unchanged_names": []}),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()
    assert not app.exception
    previous_messages = list(app.session_state["messages"])
    app.button(key="documate_clear_uploads").click().run()
    assert not app.exception
    assert app.session_state["upload_manifest"].files == []
    assert app.session_state["messages"] == previous_messages
    assert any("0개 파일" in item.value for item in app.markdown)


def test_stale_question_refreshes_attachment_context_without_replaying_question(monkeypatch, tmp_path):
    """A rejected question refreshes its attachment generation and waits for a new user request."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    calls = []

    def send(_session, method, url, **kwargs):
        calls.append(method)
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if method == "post":
            response.headers["Content-Type"] = "text/event-stream"
            response._content = b'event: error\ndata: {"message":"UPLOAD_REVISION_CONFLICT: attachment set changed"}\n\nevent: done\ndata: {}\n\n'
        else:
            response._content = _manifest(revision=1 if len(calls) == 1 else 4).model_dump_json().encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    app = AppTest.from_file(streamlit_app.__file__).run()
    app.session_state["upload_saved_prompt"] = "ask with the current attachments"
    app.run()
    app.button(key="documate_send_saved_upload_prompt").click().run()
    assert not app.exception
    assert app.session_state["upload_manifest"].revision == 4
    assert calls == ["get", "post", "get"]


def _stream_responses(monkeypatch, responses):
    calls = []
    responses = iter(responses)

    def send(_session, method, url, **kwargs):
        calls.append({"method": method, "url": url, "payload": kwargs.get("json")})
        expected_method, payload = next(responses)
        assert method == expected_method
        if isinstance(payload, Exception):
            raise payload
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if method == "get":
            response._content = json.dumps(payload).encode()
        else:
            response.headers["Content-Type"] = "text/event-stream"
            response._content = "".join(
                f"event: {event}\ndata: {json.dumps(data)}\n\n" for event, data in payload
            ).encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    return calls


def _send_saved_question(app, prompt):
    app.session_state["upload_saved_prompt"] = prompt
    app.run()
    app.button(key="documate_send_saved_upload_prompt").click().run()
    assert not app.exception


@pytest.mark.parametrize("preceding_error", [False, True])
def test_final_manifest_updates_sidebar_and_next_question_without_extra_get(monkeypatch, tmp_path, preceding_error):
    """A completed reset immediately clears the sidebar and supplies the next question's epoch."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    reset_manifest = UploadManifest(epoch="reset-epoch", revision=0, files=[])
    reset_answer = answer_response("Chat session has been reset. Start again.")
    final = {"response": reset_answer.model_dump(mode="json"),
             "upload_manifest": reset_manifest.model_dump(mode="json")}
    reset_events = [("error", {"message": "recoverable error"})] if preceding_error else []
    reset_events.append(("final_response", final))
    followup_answer = answer_response("New question accepted.")
    calls = _stream_responses(monkeypatch, [
        ("get", _manifest(files=[_confirmed_file()]).model_dump(mode="json")),
        ("post", reset_events),
        ("post", [("final_response", {
            "response": followup_answer.model_dump(mode="json"),
            "upload_manifest": reset_manifest.model_dump(mode="json"),
        })]),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()
    previous_messages = list(app.session_state["messages"])

    _send_saved_question(app, "exit")

    assert app.session_state["upload_manifest"] == reset_manifest
    assert app.session_state["messages"][:-2] == previous_messages
    assert app.session_state["messages"][-1]["response"] == reset_answer
    assert any("0개 파일" in item.value for item in app.markdown)
    assert not any(button.label == "a.py 삭제" for button in app.button)

    _send_saved_question(app, "next question")

    assert app.session_state["messages"][-1]["response"] == followup_answer
    assert [call["method"] for call in calls] == ["get", "post", "post"]
    assert calls[-1]["payload"]["uploads"] == reset_manifest.context().model_dump(mode="json")
    assert calls[1]["payload"]["session_id"] == calls[2]["payload"]["session_id"]


@pytest.mark.parametrize("completion", ["legacy_final", "timeout", "done"])
def test_unconfirmed_question_result_refreshes_before_next_input_without_replay(monkeypatch, tmp_path, completion):
    """Missing confirmation recovers through a read-only refresh without repeating the question."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    fresh = UploadManifest(epoch="fresh-epoch", revision=0, files=[])
    response = {
        "legacy_final": [("final_response", {"response": answer_response("Completed.").model_dump(mode="json")})],
        "timeout": requests.exceptions.Timeout("response was lost"),
        "done": [("done", {})],
    }[completion]
    calls = _stream_responses(monkeypatch, [
        ("get", _manifest(files=[_confirmed_file()]).model_dump(mode="json")),
        ("post", response),
        ("get", fresh.model_dump(mode="json")),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()

    _send_saved_question(app, "exit")

    assert app.session_state["upload_manifest"] == fresh
    assert [message["content"] for message in app.session_state["messages"] if message["role"] == "user"] == ["exit"]
    assert [call["method"] for call in calls] == ["get", "post", "get"]
    assert len(app.chat_input) == 1
    if completion == "legacy_final":
        assert app.session_state["messages"][-1]["response"] == answer_response("Completed.")


def test_failed_manifest_recovery_keeps_confirmation_unknown_without_replay(monkeypatch, tmp_path):
    """An unavailable confirmation endpoint preserves unknown state and offers a read-only reconnect."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    calls = _stream_responses(monkeypatch, [
        ("get", _manifest(files=[_confirmed_file()]).model_dump(mode="json")),
        ("post", requests.exceptions.Timeout("response was lost")),
        ("get", requests.exceptions.ConnectionError("server is unavailable")),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()

    _send_saved_question(app, "exit")

    assert app.session_state["upload_manifest"] is None
    assert any(button.key == "documate_reconnect_uploads" for button in app.button)
    assert [message["content"] for message in app.session_state["messages"] if message["role"] == "user"] == ["exit"]
    assert [call["method"] for call in calls] == ["get", "post", "get"]


def test_closing_uncertain_upload_requires_fresh_confirmation_before_next_question(monkeypatch, tmp_path):
    """Closing a failed mutation cannot expose a stale attachment set when reconnection also fails."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    calls = _stream_responses(monkeypatch, [
        ("get", _manifest(files=[_confirmed_file()]).model_dump(mode="json")),
        ("post", requests.exceptions.Timeout("the upload response was lost")),
        ("get", requests.exceptions.ConnectionError("server is unavailable")),
        ("get", requests.exceptions.ConnectionError("server is unavailable")),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()
    messages = list(app.session_state["messages"])
    app.button(key="documate_clear_uploads").click().run()

    app.button(key="documate_cancel_pending_upload").click().run()

    assert not app.exception
    assert app.session_state["upload_manifest"] is None
    assert app.session_state["pending_upload"] is None
    assert app.session_state["messages"] == messages
    assert len(app.chat_input) == 0
    assert any(button.key == "documate_reconnect_uploads" for button in app.button)
    assert [call["method"] for call in calls] == ["get", "post", "get", "get"]


def _submit_attachments(monkeypatch, app, files, prompt=""):
    """Supply one file input at the Streamlit boundary; AppTest has no file-upload setter."""
    chat_input = streamlit_app.st.chat_input
    submissions = [SimpleNamespace(text=prompt, files=[
        SimpleNamespace(name=name, getbuffer=lambda data=data: data)
        for name, data in files
    ])]

    def submit_once(*args, **kwargs):
        current = chat_input(*args, **kwargs)
        return submissions.pop() if submissions else current

    with monkeypatch.context() as input_patch:
        input_patch.setattr(streamlit_app.st, "chat_input", submit_once)
        app.run()
    assert not app.exception


def test_multiple_attachments_are_confirmed_before_the_comparison_question(monkeypatch, tmp_path):
    """One file submission adds both files and sends its question only against the committed set."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    files = [_confirmed_file(), _confirmed_file().model_copy(update={
        "file_id": "file-b", "name": "b.py", "source_uri": "upload://session-one/file-b",
    })]
    committed = _manifest(revision=1, files=files)
    calls = []

    def send(_session, method, url, **kwargs):
        calls.append({"method": method, "url": url, "payload": kwargs.get("json")})
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if method == "get":
            response._content = _manifest(revision=0).model_dump_json().encode()
        elif url.endswith("/uploads/sync"):
            assert [(item["name"], Path(item["path"]).read_bytes()) for item in kwargs["json"]["add"]] == [
                ("a.py", b"one"), ("b.py", b"two"),
            ]
            response._content = json.dumps({"manifest": committed.model_dump(), "changed": True}).encode()
        else:
            response.headers["Content-Type"] = "text/event-stream"
            final = {"response": answer_response("Both files compared.").model_dump(mode="json"),
                     "upload_manifest": committed.model_dump(mode="json")}
            response._content = f"event: final_response\ndata: {json.dumps(final)}\n\n".encode()
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", send)
    app = AppTest.from_file(streamlit_app.__file__).run()

    _submit_attachments(monkeypatch, app, [("a.py", b"one"), ("b.py", b"two")], "compare a.py and b.py")

    assert app.session_state["upload_manifest"] == committed
    assert [call["method"] for call in calls] == ["get", "post", "post"]
    assert calls[1]["payload"]["expected_revision"] == 0
    assert calls[2]["payload"]["uploads"] == committed.context().model_dump()
    assert calls[2]["payload"]["query"] == "compare a.py and b.py"
    assert [message["content"] for message in app.session_state["messages"] if message["role"] == "user"] == ["compare a.py and b.py"]
    assert {button.label for button in app.button} >= {"a.py 삭제", "b.py 삭제"}
    assert not list(tmp_path.rglob("*.py"))


def test_replacement_button_commits_only_after_reviewing_the_same_name(monkeypatch, tmp_path):
    """A changed filename waits for explicit replacement and then displays the new confirmed file."""
    import hashlib
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    replacement = _confirmed_file().model_copy(update={
        "content_hash": "sha256:" + hashlib.sha256(b"new").hexdigest(),
    })
    committed = _manifest(revision=2, files=[replacement])
    calls = _responses(monkeypatch, [
        (200, _manifest(files=[_confirmed_file()]).model_dump()),
        (200, {"manifest": committed.model_dump(), "changed": True}),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()
    messages = list(app.session_state["messages"])

    _submit_attachments(monkeypatch, app, [("a.py", b"new")])

    assert [call["method"] for call in calls] == ["get"]
    assert app.session_state["upload_manifest"].files == [_confirmed_file()]
    assert any("같은 이름의 다른 내용" in warning.value for warning in app.warning)

    app.button(key="documate_confirm_upload_replacements").click().run()

    assert not app.exception
    assert app.session_state["upload_manifest"] == committed
    assert app.session_state["messages"] == messages
    assert calls[1]["payload"]["add"][0]["replace_file_id"] == "file-a"
    assert not list(tmp_path.rglob("*.py"))


def test_individual_delete_preserves_other_attachments_and_conversation(monkeypatch, tmp_path):
    """The public delete button removes exactly its selected file while keeping the other file and answers."""
    from streamlit.testing.v1 import AppTest
    monkeypatch.setattr(streamlit_state, "get_uploads_dir", lambda: tmp_path)
    other = _confirmed_file().model_copy(update={
        "file_id": "file-b", "name": "b.py", "source_uri": "upload://session-one/file-b",
    })
    remaining = _manifest(revision=2, files=[other])
    calls = _responses(monkeypatch, [
        (200, _manifest(files=[_confirmed_file(), other]).model_dump()),
        (200, {"manifest": remaining.model_dump(), "changed": True}),
    ])
    app = AppTest.from_file(streamlit_app.__file__).run()
    messages = list(app.session_state["messages"])

    app.button(key="documate_remove_upload_file-a").click().run()

    assert not app.exception
    assert app.session_state["upload_manifest"] == remaining
    assert app.session_state["messages"] == messages
    assert calls[1]["payload"]["remove"] == ["file-a"]
    assert calls[1]["payload"]["clear"] is False
    assert any(button.label == "b.py 삭제" for button in app.button)
    assert not any(button.label == "a.py 삭제" for button in app.button)
