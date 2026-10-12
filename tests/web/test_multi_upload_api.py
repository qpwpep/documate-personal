"""Real HTTP attachment transactions and graph retrieval, with external model boundaries replaced."""

from __future__ import annotations

import json
import hashlib
import logging
import socket
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Barrier, Event
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage

from src.app.web.app import create_app
from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.settings import AppSettings


class LocalEmbeddings(Embeddings):
    def __init__(self, controls):
        self.controls = controls

    def embed_documents(self, texts):
        if self.controls.before_embedding is not None:
            self.controls.before_embedding()
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class LocalChatModel:
    """Deterministic OpenAI boundary; all planning, retrieval and validation remain real."""

    def __init__(self, controls, *, schema_name=""):
        self.controls = controls
        self.schema_name = schema_name

    def with_structured_output(self, schema, **kwargs):
        return LocalChatModel(self.controls, schema_name=schema["name"])

    def invoke(self, messages, **_kwargs):
        if self.schema_name == "PlannerOutput":
            raw = next(message.content for message in messages if message.name == "request_context")
            context = json.loads(raw.split("\n", 1)[1])
            files = context["upload_files"]
            file_ids = (self.controls.planned_file_ids if self.controls.planned_file_ids is not None
                        else [item["file_id"] for item in files])
            parsed = {
                "use_retrieval": bool(files),
                "tasks": [{"route": "upload", "query": "compare values", "k": 4,
                           "requirement": {"file_ids": file_ids}}] if files else [],
                "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
            }
            if files and self.controls.planned_symbols:
                parsed["tasks"] = [
                    {"route": "upload", "query": f"Compare {symbol} definitions", "k": 4,
                     "requirement": {"file_ids": file_ids,
                                     "symbols": [symbol], "match": "definition"}}
                    for symbol in self.controls.planned_symbols
                ]
            parsed = InitialPlannerOutput.model_validate(parsed).model_dump(mode="json")
        elif self.schema_name == "RetrievalRewriteOutput":
            raise AssertionError("Upload-only requests must not replan retrieval")
        elif self.schema_name == "AnswerDocument":
            if self.controls.before_synthesis is not None:
                self.controls.before_synthesis()
            if self.controls.fail_synthesis:
                raise TimeoutError("Simulated model outage")
            packet = json.loads(str(messages[-1].content).split("\n", 2)[2])
            blocks = [{"type": "code", "language": "python", "content": {
                "text": item["excerpt"], "basis": "excerpt", "refs": [item["id"]],
            }} for item in packet]
            parsed = {"blocks": blocks or [{"type": "paragraph", "content": [
                {"text": "No attached evidence.", "basis": "interaction", "refs": []},
            ]}]}
        else:
            return AIMessage(content="Conversation summary.")
        return {"parsed": parsed, "parsing_error": None, "raw": AIMessage(
            content="", response_metadata={"model_name": "local-test-model"},
            usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
        )}


@pytest.fixture
def api(tmp_path, monkeypatch):
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
        upload_max_files=10, upload_max_file_mib=1, upload_max_total_mib=2,
        session_ttl_seconds=5, session_cleanup_interval_seconds=1,
    )
    controls = SimpleNamespace(fail_synthesis=False, before_embedding=None, before_synthesis=None,
                               planned_symbols=(), planned_file_ids=None)
    monkeypatch.setattr("src.app.web.app.get_settings", lambda: settings)
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: LocalEmbeddings(controls))
    monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: LocalChatModel(controls))
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    connect = socket.socket.connect

    def local_connections_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network access is forbidden in upload API tests")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connections_only)
    with TestClient(create_app()) as client:
        yield SimpleNamespace(client=client, root=tmp_path, settings=settings, controls=controls)


def staged(api, name, content, *, session="session-a"):
    path = api.root / "uploads" / session / "staging" / uuid4().hex / name
    path.parent.mkdir(parents=True)
    path.write_bytes(content.encode("utf-8") if isinstance(content, str) else content)
    return {"path": str(path), "name": name,
            "content_hash": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()}


def manifest(api, session="session-a"):
    response = api.client.get(f"/sessions/{session}/uploads")
    assert response.status_code == 200, response.text
    return response.json()


def sync_body(current, *, add=(), remove=(), clear=False, operation_id=None):
    return {"epoch": current["epoch"], "expected_revision": current["revision"],
            "operation_id": operation_id or uuid4().hex, "add": list(add), "remove": list(remove), "clear": clear}


def sync(api, *, add=(), remove=(), clear=False, session="session-a"):
    response = api.client.post(f"/sessions/{session}/uploads/sync",
                               json=sync_body(manifest(api, session), add=add, remove=remove, clear=clear))
    assert response.status_code == 200, response.text
    return response.json()


def context(current):
    return {"epoch": current["epoch"], "revision": current["revision"]}


def stream(api, **kwargs):
    response = api.client.post("/agent/stream", json={"query": "Compare the attached files", "session_id": "session-a", **kwargs})
    assert response.status_code == 200, response.text
    events = []
    for frame in response.text.strip().split("\n\n"):
        lines = frame.splitlines()
        event = next(line[7:] for line in lines if line.startswith("event: "))
        data = json.loads(next(line[6:] for line in lines if line.startswith("data: ")))
        events.append((event, data))
    return events


def final_response(api, **kwargs):
    events = stream(api, **kwargs)
    assert not [data for event, data in events if event == "error"], events
    return next(data for event, data in events if event == "final_response")


def answer(api, **kwargs):
    final = final_response(api, **kwargs)
    return AnswerResponse.model_validate(final["response"])






def test_sync_rejects_bytes_changed_after_approval(api):
    current = manifest(api)
    addition = staged(api, "alpha.py", "alpha = 1\n")
    addition["content_hash"] = "sha256:" + hashlib.sha256(b"alpha = 1\n").hexdigest()
    Path(addition["path"]).write_bytes(b"alpha = 2\n")
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=[addition]))
    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UPLOAD_CONTENT_CHANGED"
    assert manifest(api) == current
    assert not list((api.root / "uploads" / "session-a" / "objects").glob("**/*.py"))










@contextmanager
def session_requests_arrived(count):
    """Observe actual server request arrivals through its existing log boundary, without replacing the store."""
    arrived = Event()

    class ArrivalObserver(logging.Handler):
        remaining = count

        def emit(self, record):
            if getattr(record, "event", None) == "session_cache_event":
                self.remaining -= 1
                if self.remaining <= 0:
                    arrived.set()

    logger = logging.getLogger("src.app.web.session_store")
    observer = ArrivalObserver()
    logger.addHandler(observer)
    try:
        yield arrived
    finally:
        logger.removeHandler(observer)






def test_http_comparison_only_uses_the_planned_files_from_a_larger_attachment_set(api):
    """A comparison scoped to two attached files never cites an unrelated third attachment."""
    current = sync(api, add=[
        staged(api, "alpha.py", "alpha = 1\n"),
        staged(api, "beta.py", "beta = 2\n"),
        staged(api, "unrelated.py", "unrelated = 99\n"),
    ])
    api.controls.planned_file_ids = [item["file_id"] for item in current["files"] if item["name"] != "unrelated.py"]

    result = answer(api, query="Compare only alpha.py and beta.py", uploads=context(current))

    assert result.issues == []
    assert {item.evidence.element.metadata["file_id"] for item in result.citations} == set(api.controls.planned_file_ids)
    assert {item.evidence.snapshot.title for item in result.citations} == {"alpha.py", "beta.py"}
    assert manifest(api) == current




@pytest.mark.parametrize("query", ["Compare the attached files", "exit"])
def test_stale_http_revision_rejects_mutations_and_question_without_changing_attachments(api, query):
    """A stale client cannot overwrite, reset or use a different attachment revision."""
    old = manifest(api)
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(old, clear=True))
    events = stream(api, query=query, uploads=context(old))

    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "UPLOAD_REVISION_CONFLICT"
    final = next(data for event, data in events if event == "final_response")
    assert final["status"] == "failed"
    assert final["problem"]["code"] == "upload_revision_conflict"
    assert final["response"] is None
    assert manifest(api) == current








def test_cross_session_staged_file_is_rejected_without_entering_the_index(api):
    """Session A cannot index source bytes staged inside session B's storage."""
    before = manifest(api)
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        before, add=[staged(api, "secret.py", "secret = 1\n", session="session-b")],
    ))

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "UPLOAD_PATH_INVALID"
    assert manifest(api) == before




def notebook_bytes(size):
    # Notebook metadata counts against the raw byte quota without creating thousands of search chunks.
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"padding": ""}, "cells": [
        {"id": "value", "cell_type": "code", "metadata": {}, "source": "value = 1\n", "outputs": [], "execution_count": None},
    ]}
    encode = lambda: json.dumps(notebook, separators=(",", ":")).encode("utf-8")
    notebook["metadata"]["padding"] = "x" * (size - len(encode()))
    return encode()














def test_failed_http_batch_preserves_previous_sources_for_later_questions(api):
    """One invalid notebook rolls back the batch while the prior attachment remains usable over HTTP."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])
    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=[
        staged(api, "beta.py", "beta = 2\n"), staged(api, "broken.ipynb", "not JSON"),
    ]))

    assert rejected.status_code == 422
    assert manifest(api) == current
    result = answer(api, uploads=context(current))
    assert {citation.evidence.snapshot.title for citation in result.citations} == {"alpha.py"}












@pytest.mark.parametrize("intervening_change", [False, True])
def test_session_recovers_committed_upload_after_lost_response(api, monkeypatch, intervening_change):
    """Lost HTTP delivery cannot duplicate a committed batch or replay a question."""
    from urllib.parse import urlsplit

    import requests

    from src.app.client import AgentRequestContext, AgentSessionClient, UploadAPIError
    from src.app.uploads import build_upload_sync_request, discard_staged_files

    deliveries = []
    lose_first_sync = True

    def deliver(_session, method, url, **kwargs):
        nonlocal lose_first_sync
        path = urlsplit(url).path
        deliveries.append((method, path, kwargs.get("json")))
        # Keep the app, session store, indexing, and response contracts real.
        received = api.client.request(method, path, json=kwargs.get("json"))
        if path.endswith("/uploads/sync") and lose_first_sync:
            lose_first_sync = False
            assert received.status_code == 200, received.text
            raise requests.Timeout("response lost after the server committed")
        response = requests.Response()
        response.status_code = received.status_code
        response.headers.update(received.headers)
        response._content = received.content
        response._content_consumed = True
        return response

    monkeypatch.setattr(requests.sessions.Session, "request", deliver)
    client = AgentSessionClient(AgentRequestContext(fastapi_url="http://local-test", session_id="session-a"))
    session_path = api.root / "uploads" / "session-a"
    prepared = client.stage_files(
        [SimpleNamespace(name="alpha.py", getbuffer=lambda: b"alpha = 1\n")], session_path,
        max_files=10, max_file_mib=1, max_total_mib=2,
    )
    assert not prepared.errors
    request = build_upload_sync_request(client.manifest, files=prepared.files)
    original_payload = request.model_dump(mode="json")

    with pytest.raises(UploadAPIError):
        client.sync_uploads(request)

    assert client.manifest is None
    committed = manifest(api)
    assert committed["revision"] == 1
    assert [file["name"] for file in committed["files"]] == ["alpha.py"]
    assert len(deliveries) == 2  # Initial GET and exactly one attempted mutation.
    assert Path(prepared.files[0].path).read_bytes() == b"alpha = 1\n"
    current = (sync(api, add=[staged(api, "beta.py", "beta = 2\n")])
               if intervening_change else committed)

    replayed = client.sync_uploads(request)

    assert replayed.model_dump(mode="json") == current
    assert manifest(api) == current
    sync_payloads = [payload for _, path, payload in deliveries if path.endswith("/uploads/sync")]
    assert sync_payloads == [original_payload, original_payload]
    assert all(path != "/agent/stream" for _, path, _ in deliveries)
    assert client.manifest.model_dump(mode="json") == current
    events = list(client.stream("Compare the attached files"))
    assert not [event for event in events if event.event == "error"]
    final = next(event.result for event in events if event.event == "final_response")
    assert final.upload_manifest.model_dump(mode="json") == current
    assert {citation.evidence.snapshot.title for citation in final.response.citations} == {
        file["name"] for file in current["files"]
    }
    assert len([path for _, path, _ in deliveries if path == "/agent/stream"]) == 1
    discard_staged_files(prepared.files, session_path)


def test_loopback_attachment_changes_reach_streamlit_client_and_versioned_answers(api, monkeypatch):
    """Real TCP requests carry additions, replacements and deletions into the next answer's exact sources."""
    from threading import Thread

    import uvicorn

    from src.app.client import (
        AgentRequestContext, UploadAPIError, fetch_upload_manifest, stream_agent_response, sync_uploads,
    )

    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    # TestClient owns app lifespan; Uvicorn serves that same app on an ephemeral local port.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        server = uvicorn.Server(uvicorn.Config(api.client.app, lifespan="off", ws="none", log_config=None, access_log=False))
        worker = Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
        worker.start()
        try:
            def update(current, **changes):
                return sync_uploads(url, "session-a", sync_body(current.model_dump(mode="json"), **changes))

            def ask(current):
                events = list(stream_agent_response("Compare the attached files", AgentRequestContext(
                    fastapi_url=url, session_id="session-a",
                ), uploads=current.context()))
                assert not [event for event in events if event.event == "error"]
                result = next(event.result for event in events if event.event == "final_response")
                assert result.upload_manifest == current
                return result.response

            empty = fetch_upload_manifest(url, "session-a")
            first = update(empty, add=[staged(api, "alpha.py", "alpha = 1\n")])
            attached = update(first, add=[staged(api, "beta.py", "beta = 2\n")])
            original = ask(attached)
            assert {item.evidence.snapshot.title for item in original.citations} == {"alpha.py", "beta.py"}
            assert update(attached, add=[staged(api, "alpha.py", "alpha = 1\n")]) == attached

            replacement = staged(api, "alpha.py", "alpha = 10\n")
            with pytest.raises(UploadAPIError) as conflict:
                update(attached, add=[replacement])
            assert conflict.value.code == "UPLOAD_NAME_CONFLICT"
            alpha = next(item for item in attached.files if item.name == "alpha.py")
            replacement["replace_file_id"] = alpha.file_id
            replaced = update(attached, add=[replacement])
            current = ask(replaced)
            current_alpha = next(item.evidence for item in current.citations if item.evidence.snapshot.title == "alpha.py")
            old_alpha = next(item.evidence for item in original.citations if item.evidence.snapshot.title == "alpha.py")
            assert current_alpha.element.metadata["file_id"] == alpha.file_id
            assert current_alpha.excerpt == "alpha = 10"
            assert old_alpha.excerpt == "alpha = 1"
            assert current_alpha.snapshot.snapshot_id != old_alpha.snapshot.snapshot_id

            beta = next(item for item in replaced.files if item.name == "beta.py")
            removed = update(replaced, remove=[beta.file_id])
            assert {item.evidence.snapshot.title for item in ask(removed).citations} == {"alpha.py"}
            cleared = update(removed, clear=True)
            assert cleared.files == []
            assert ask(cleared).citations == []
            assert fetch_upload_manifest(url, "session-a") == cleared
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            assert not worker.is_alive(), "loopback server did not shut down"
