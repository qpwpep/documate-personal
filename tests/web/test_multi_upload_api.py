"""Real HTTP attachment transactions and graph retrieval, with external model boundaries replaced."""

from __future__ import annotations

import json
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
from src.core.contracts import SessionMetadata
from src.core.request_contracts import WireRequestContract
from src.core.uploads import UploadContext
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

    def invoke(self, messages):
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
                "request_contract": WireRequestContract().model_dump(mode="json"),
            }
            if files and self.controls.planned_symbols:
                parsed["tasks"] = [
                    {"route": "upload", "query": f"Compare {symbol} definitions", "k": 4,
                     "requirement": {"file_ids": file_ids,
                                     "symbols": [symbol], "match": "definition"}}
                    for symbol in self.controls.planned_symbols
                ]
        elif self.schema_name == "AnswerDocument":
            if self.controls.before_synthesis is not None:
                self.controls.before_synthesis()
            if self.controls.fail_synthesis:
                raise RuntimeError("Simulated model outage")
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
    return {"path": str(path), "name": name}


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


def test_multi_upload_http_question_cites_both_real_sources_and_reuses_the_manifest(api):
    """A JSON attachment batch reaches real graph retrieval and returns both immutable source citations."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n"), staged(api, "beta.py", "beta = 2\n")])["manifest"]

    first = answer(api, uploads=context(current))
    second = answer(api, uploads=context(current))

    expected = {item["file_id"] for item in current["files"]}
    assert {citation.evidence.element.metadata["file_id"] for citation in first.citations} == expected
    assert {citation.evidence.snapshot.title for citation in first.citations} == {"alpha.py", "beta.py"}
    assert {citation.evidence.snapshot.snapshot_id for citation in second.citations} == {
        citation.evidence.snapshot.snapshot_id for citation in first.citations
    }
    assert manifest(api) == current


def test_http_independent_requirements_keep_each_files_source_citations(api):
    """Independent definitions across the same five uploads reach a complete validated HTTP answer."""
    api.controls.planned_symbols = ("setup", "cleanup")
    current = sync(api, add=[
        staged(api, f"file-{index}.py",
               f"def setup():\n    return {index}\n\ndef cleanup():\n    return {index + 10}\n")
        for index in range(5)
    ])["manifest"]

    result = answer(api, query="Compare setup and cleanup in every attached file",
                    uploads=context(current))

    assert result.issues == []
    assert {
        (citation.evidence.element.metadata["file_id"], symbol)
        for citation in result.citations
        for symbol in api.controls.planned_symbols
        if f"def {symbol}(" in citation.evidence.excerpt
    } == {(file["file_id"], symbol) for file in current["files"] for symbol in api.controls.planned_symbols}
    assert manifest(api) == current


def test_http_comparison_only_uses_the_planned_files_from_a_larger_attachment_set(api):
    """A comparison scoped to two attached files never cites an unrelated third attachment."""
    current = sync(api, add=[
        staged(api, "alpha.py", "alpha = 1\n"),
        staged(api, "beta.py", "beta = 2\n"),
        staged(api, "unrelated.py", "unrelated = 99\n"),
    ])["manifest"]
    api.controls.planned_file_ids = [item["file_id"] for item in current["files"] if item["name"] != "unrelated.py"]

    result = answer(api, query="Compare only alpha.py and beta.py", uploads=context(current))

    assert result.issues == []
    assert {item.evidence.element.metadata["file_id"] for item in result.citations} == set(api.controls.planned_file_ids)
    assert {item.evidence.snapshot.title for item in result.citations} == {"alpha.py", "beta.py"}
    assert manifest(api) == current


@pytest.mark.parametrize("clear_payload", [{}, {"upload_file_path": None}])
def test_legacy_http_path_replaces_the_set_and_absent_or_null_path_clears_it(api, clear_payload):
    """Legacy requests keep their replacement/clear semantics while using the new real source index."""
    sync(api, add=[staged(api, "alpha.py", "alpha = 1\n"), staged(api, "beta.py", "beta = 2\n")])
    legacy = staged(api, "legacy.py", "legacy = 3\n")

    replaced = answer(api, upload_file_path=legacy["path"])
    assert {citation.evidence.snapshot.title for citation in replaced.citations} == {"legacy.py"}
    assert [item["name"] for item in manifest(api)["files"]] == ["legacy.py"]

    cleared = answer(api, **clear_payload)
    assert cleared.citations == []
    assert manifest(api)["files"] == []


@pytest.mark.parametrize("legacy_path", [None, "uploads/session-a/alpha.py"])
def test_http_schema_rejects_mixed_attachment_protocols_before_creating_a_session(api, legacy_path):
    """Neither null nor non-null legacy fields may silently override a versioned upload context."""
    response = api.client.post("/agent/stream", json={
        "query": "Compare files", "session_id": "session-a", "upload_file_path": legacy_path,
        "uploads": {"epoch": "old", "revision": 0},
    })
    assert response.status_code == 422
    assert api.client.app.state.session_store.active_session_ids() == set()


@pytest.mark.parametrize("query", ["Compare the attached files", "exit"])
def test_stale_http_revision_rejects_mutations_and_question_without_changing_attachments(api, query):
    """A stale client cannot overwrite, reset or use a different attachment revision."""
    old = manifest(api)
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(old, clear=True))
    events = stream(api, query=query, uploads=context(old))

    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "UPLOAD_REVISION_CONFLICT"
    assert any(event == "error" and "UPLOAD_REVISION_CONFLICT" in data["message"] for event, data in events)
    assert not any(event == "final_response" for event, _ in events)
    assert manifest(api) == current


def test_exit_from_a_previous_epoch_preserves_the_current_session_at_the_same_revision(api):
    """A previous session's reset context cannot erase a new session even when revisions coincide."""
    old = sync(api, add=[staged(api, "old.py", "old = 1\n")])["manifest"]
    final_response(api, query="exit", uploads=context(old))
    current = sync(api, add=[staged(api, "current.py", "current = 2\n")])["manifest"]
    answer(api, uploads=context(current))
    session = api.client.app.state.session_store.get_or_create("session-a")._ensure_session()
    memory = session.snapshot_conversation_memory()
    owned = [Path(record.path) for record in session.upload_records]
    assert current["revision"] == old["revision"]
    assert current["epoch"] != old["epoch"]

    events = stream(api, query="exit", uploads=context(old))

    assert any(event == "error" and "UPLOAD_REVISION_CONFLICT" in data["message"] for event, data in events)
    assert not any(event == "final_response" for event, _ in events)
    assert session.upload_manifest().model_dump(mode="json") == current
    assert session.snapshot_conversation_memory() == memory
    assert all(path.is_file() for path in owned)


def test_expired_session_epoch_rejects_an_old_upload_operation(api, monkeypatch):
    """An expired session recreated under the same ID cannot accept the prior client's revision zero."""
    clock = SimpleNamespace(value=1000.0)
    monkeypatch.setattr("src.app.web.session_store.time", SimpleNamespace(monotonic=lambda: clock.value))
    old = manifest(api)
    clock.value += api.settings.session_ttl_seconds + api.settings.session_cleanup_interval_seconds + 1
    current = manifest(api)

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(old, clear=True))

    assert current["epoch"] != old["epoch"]
    assert current["revision"] == old["revision"] == 0
    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "UPLOAD_REVISION_CONFLICT"
    assert manifest(api) == current


def test_http_operation_replay_is_identical_and_changed_payload_conflicts(api):
    """A retried transaction is idempotent, but the same operation ID cannot authorize another change."""
    body = sync_body(manifest(api), add=[staged(api, "alpha.py", "alpha = 1\n")])
    first = api.client.post("/sessions/session-a/uploads/sync", json=body)
    replayed = api.client.post("/sessions/session-a/uploads/sync", json=body)
    conflict = api.client.post("/sessions/session-a/uploads/sync", json={**body, "add": [], "clear": True})

    assert first.status_code == replayed.status_code == 200
    assert replayed.json() == first.json()
    assert conflict.status_code == 409
    assert conflict.json()["detail"]["code"] == "UPLOAD_OPERATION_CONFLICT"
    assert manifest(api) == first.json()["manifest"]


def test_cross_session_staged_file_is_rejected_without_entering_the_index(api):
    """Session A cannot index source bytes staged inside session B's storage."""
    before = manifest(api)
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        before, add=[staged(api, "secret.py", "secret = 1\n", session="session-b")],
    ))

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "UPLOAD_PATH_INVALID"
    assert manifest(api) == before


def test_ten_files_are_accepted_and_an_eleventh_file_is_rejected_atomically(api):
    """The configured ten-file limit counts the whole active set across later additions."""
    current = sync(api, add=[staged(api, f"file-{index}.py", f"value = {index}\n") for index in range(10)])["manifest"]
    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        current, add=[staged(api, "eleventh.py", "extra = 11\n")],
    ))

    assert len(current["files"]) == 10
    assert response.status_code == 413
    assert response.json()["detail"]["code"] == "UPLOAD_TOO_MANY_FILES"
    assert manifest(api) == current


def notebook_bytes(size):
    # Notebook metadata counts against the raw byte quota without creating thousands of search chunks.
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"padding": ""}, "cells": [
        {"id": "value", "cell_type": "code", "metadata": {}, "source": "value = 1\n", "outputs": [], "execution_count": None},
    ]}
    encode = lambda: json.dumps(notebook, separators=(",", ":")).encode("utf-8")
    notebook["metadata"]["padding"] = "x" * (size - len(encode()))
    return encode()


def test_http_exit_releases_managed_files_and_reuses_the_full_upload_quota(api):
    """Resetting the same HTTP session releases its committed originals before the next manifest fetch."""
    previous_epoch = manifest(api)["epoch"]
    for _ in range(3):
        additions = [staged(api, name, notebook_bytes(1024 * 1024))
                     for name in ("first.ipynb", "second.ipynb")]
        current = sync(api, add=additions)["manifest"]
        for addition in additions:
            path = Path(addition["path"])
            path.unlink()
            path.parent.rmdir()

        reset = answer(api, query="exit", uploads=context(current))

        assert reset.issues == []
        assert not [path for path in (api.root / "uploads" / "session-a" / "objects").rglob("*")
                    if path.is_file()]
        empty = manifest(api)
        assert empty["files"] == []
        assert empty["revision"] == 0
        assert empty["epoch"] != previous_epoch
        previous_epoch = empty["epoch"]


@pytest.mark.parametrize("name, content", [
    pytest.param("broken.ipynb", b"not-json", id="invalid-notebook"),
    pytest.param("replacement.py", b"replacement = 2\n", id="embedding-outage"),
])
def test_legacy_exit_resets_the_session_without_preparing_the_supplied_file(api, name, content):
    """Legacy exit resets owned state despite unusable upload input, preserving borrowed bytes and prior citations."""
    original = staged(api, "source.py", "value = 1\n")
    current = sync(api, add=[original])["manifest"]
    previous = answer(api, uploads=context(current))
    saved_citations = [citation.model_dump(mode="json") for citation in previous.citations]
    agent = api.client.app.state.session_store.get_or_create("session-a")
    session = agent._ensure_session()
    owned = [Path(record.path) for record in session.upload_records]
    assert session.snapshot_conversation_memory().messages
    assert session.previous_response is not None
    supplied = staged(api, name, content)

    def fail_embedding():
        raise RuntimeError("Simulated embedding outage")

    api.controls.before_embedding = fail_embedding
    reset = final_response(api, query="exit", upload_file_path=supplied["path"], include_debug=True)

    empty = reset["upload_manifest"]
    assert empty["epoch"] != current["epoch"]
    assert empty == {"epoch": empty["epoch"], "revision": 0, "files": []}
    assert export_answer_text(AnswerResponse.model_validate(reset["response"])) == "Chat session has been reset. Start again."
    assert reset["debug"]["model_usage_status"] == "deterministic"
    assert reset["debug"]["errors"] == []
    # Check close itself before a manifest GET can reconcile orphaned originals.
    assert all(not path.exists() for path in owned)
    assert session.upload_retriever_handle is None
    memory = session.snapshot_conversation_memory()
    assert memory.messages == memory.user_turns == ()
    assert memory.memory_summary is None
    assert session.previous_response is session.pending_action is None
    assert not session.upload_operations
    assert Path(original["path"]).read_bytes() == b"value = 1\n"
    assert Path(supplied["path"]).read_bytes() == content
    assert [citation.model_dump(mode="json") for citation in previous.citations] == saved_citations

    following = final_response(api, query="hello", uploads=context(empty))
    assert following["upload_manifest"] == empty


@pytest.mark.parametrize("command", ["exit", "종료", "quit", "q"])
@pytest.mark.parametrize("attached", [False, True])
def test_exit_response_supplies_the_context_for_the_next_question(api, command, attached):
    """A successful reset lets the next explicit question use its new context without a manifest GET."""
    current = (sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]
               if attached else manifest(api))

    reset = final_response(api, query=command, uploads=context(current))

    empty = reset["upload_manifest"]
    assert empty["epoch"] != current["epoch"]
    assert empty == {"epoch": empty["epoch"], "revision": 0, "files": []}
    assert reset["debug"] is None
    following = final_response(api, query="hello", uploads=context(empty))
    assert following["upload_manifest"] == empty
    assert AnswerResponse.model_validate(following["response"]).content.blocks


@pytest.mark.parametrize("model_failure", [False, True])
def test_final_response_preserves_the_confirmed_manifest_even_when_the_model_fails(api, model_failure):
    """The completion envelope reports attachment state independently of answer quality or debug visibility."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]
    api.controls.fail_synthesis = model_failure

    final = final_response(api, uploads=context(current))

    assert final["upload_manifest"] == current
    assert bool(AnswerResponse.model_validate(final["response"]).issues) == model_failure


def test_session_request_manifest_is_a_detached_snapshot_of_its_completion(api):
    """A later reset cannot change the manifest returned with an earlier session request."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]
    store = api.client.app.state.session_store
    request = {"session_id": "session-a", "session_metadata": SessionMetadata(),
               "uploads": UploadContext.model_validate(context(current))}

    _, _, _, previous = store.run_session_request(user_input="Compare files", **request)
    _, _, _, reset = store.run_session_request(user_input="exit", **request)

    assert previous.model_dump(mode="json") == current
    assert reset.epoch != previous.epoch
    assert reset.model_dump(mode="json") == {"epoch": reset.epoch, "revision": 0, "files": []}


@pytest.mark.parametrize("limit", ["file", "total"])
def test_http_upload_size_accepts_exact_quota_and_rejects_one_byte_over(api, limit):
    """Per-file and aggregate raw-byte quotas accept their boundary and preserve the set on overflow."""
    one_mib = 1024 * 1024
    additions = [staged(api, "first.ipynb", notebook_bytes(one_mib))]
    if limit == "total":
        additions.append(staged(api, "second.ipynb", notebook_bytes(one_mib)))
    current = sync(api, add=additions)["manifest"]
    overflow = (staged(api, "too-large.ipynb", notebook_bytes(one_mib + 1)) if limit == "file"
                else staged(api, "extra.py", b"x"))

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=[overflow]))

    assert sum(item["size_bytes"] for item in current["files"]) == one_mib * len(additions)
    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == ("UPLOAD_FILE_TOO_LARGE" if limit == "file" else "UPLOAD_TOTAL_TOO_LARGE")
    assert manifest(api) == current


def test_failed_http_batch_preserves_previous_sources_for_later_questions(api):
    """One invalid notebook rolls back the batch while the prior attachment remains usable over HTTP."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]
    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=[
        staged(api, "beta.py", "beta = 2\n"), staged(api, "broken.ipynb", "not JSON"),
    ]))

    assert rejected.status_code == 422
    assert manifest(api) == current
    result = answer(api, uploads=context(current))
    assert {citation.evidence.snapshot.title for citation in result.citations} == {"alpha.py"}


def test_model_failure_leaves_the_attachment_set_searchable_for_the_next_question(api):
    """A model outage cannot destroy a successfully committed session search index."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n"), staged(api, "beta.py", "beta = 2\n")])["manifest"]
    api.controls.fail_synthesis = True
    failed = answer(api, uploads=context(current))
    assert failed.issues
    assert manifest(api) == current

    api.controls.fail_synthesis = False
    recovered = answer(api, uploads=context(current))
    assert {citation.evidence.snapshot.title for citation in recovered.citations} == {"alpha.py", "beta.py"}
    assert manifest(api) == current


def test_concurrent_additions_from_one_revision_commit_only_one_complete_set(api):
    """Two writers sharing a base revision preserve the existing source and reject the losing change."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]
    additions = [staged(api, "beta.py", "beta = 2\n"), staged(api, "gamma.py", "gamma = 3\n")]
    requests_ready = Barrier(3)
    index_started, release_index = Event(), Event()

    def hold_index():
        index_started.set()
        assert release_index.wait(10), "test did not release the embedding operation"

    def add_file(addition):
        requests_ready.wait(timeout=10)
        return api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=[addition]))

    api.controls.before_embedding = hold_index
    with session_requests_arrived(2) as arrived, ThreadPoolExecutor(max_workers=2) as workers:
        futures = [workers.submit(add_file, addition) for addition in additions]
        try:
            requests_ready.wait(timeout=10)
            assert index_started.wait(10), "neither concurrent upload reached indexing"
            assert arrived.wait(10), "both upload requests did not reach the session store"
        finally:
            release_index.set()
        responses = [future.result(timeout=10) for future in futures]
    api.controls.before_embedding = None

    assert sorted(response.status_code for response in responses) == [200, 409]
    winner = next(response.json()["manifest"] for response in responses if response.status_code == 200)
    loser = next(response for response in responses if response.status_code == 409)
    assert loser.json()["detail"]["code"] == "UPLOAD_REVISION_CONFLICT"
    assert winner["revision"] == current["revision"] + 1
    assert winner["files"][0] == current["files"][0]
    assert len(winner["files"]) == 2
    assert manifest(api) == winner
    result = answer(api, uploads=context(winner))
    assert {citation.evidence.element.metadata["file_id"] for citation in result.citations} == {
        item["file_id"] for item in winner["files"]
    }


def test_exit_waiting_for_an_upload_rechecks_its_revision_before_resetting(api):
    """A queued reset cannot erase attachments committed while it waited for the session lock."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])["manifest"]
    addition = staged(api, "beta.py", "beta = 2\n")
    index_started, release_index = Event(), Event()

    def hold_index():
        index_started.set()
        assert release_index.wait(10), "test did not release the embedding operation"

    api.controls.before_embedding = hold_index
    with ThreadPoolExecutor(max_workers=2) as workers:
        upload = workers.submit(api.client.post, "/sessions/session-a/uploads/sync",
                                json=sync_body(current, add=[addition]))
        try:
            assert index_started.wait(10), "upload did not reach indexing"
            with session_requests_arrived(1) as arrived:
                reset = workers.submit(stream, api, query="exit", uploads=context(current))
                assert arrived.wait(10), "reset did not reach the session store"
        finally:
            release_index.set()
        committed = upload.result(timeout=10)
        events = reset.result(timeout=10)
    api.controls.before_embedding = None

    assert committed.status_code == 200, committed.text
    updated = committed.json()["manifest"]
    assert updated["epoch"] == current["epoch"]
    assert updated["revision"] == current["revision"] + 1
    assert any(event == "error" and "UPLOAD_REVISION_CONFLICT" in data["message"] for event, data in events)
    assert not any(event == "final_response" for event, _ in events)
    assert manifest(api) == updated
    result = answer(api, uploads=context(updated))
    assert {citation.evidence.snapshot.title for citation in result.citations} == {"alpha.py", "beta.py"}


def test_question_and_file_removal_keep_each_answers_source_revision_consistent(api):
    """A question uses its starting attachment set while a queued deletion changes only later answers."""
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n"), staged(api, "beta.py", "beta = 2\n")])["manifest"]
    beta = next(item for item in current["files"] if item["name"] == "beta.py")
    synthesis_started, release_synthesis = Event(), Event()

    def hold_answer():
        synthesis_started.set()
        assert release_synthesis.wait(10), "test did not release the answer model"

    def delete_beta():
        return api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, remove=[beta["file_id"]]))

    api.controls.before_synthesis = hold_answer
    with ThreadPoolExecutor(max_workers=2) as workers:
        question = workers.submit(final_response, api, uploads=context(current))
        try:
            assert synthesis_started.wait(10), "question did not reach synthesis"
            with session_requests_arrived(1) as arrived:
                deletion = workers.submit(delete_beta)
                assert arrived.wait(10), "concurrent deletion did not reach the session store"
        finally:
            release_synthesis.set()
        old_result = question.result(timeout=10)
        removed = deletion.result(timeout=10)
    api.controls.before_synthesis = None

    assert removed.status_code == 200, removed.text
    new_manifest = removed.json()["manifest"]
    assert old_result["upload_manifest"] == current
    old_answer = AnswerResponse.model_validate(old_result["response"])
    assert {citation.evidence.element.metadata["file_id"] for citation in old_answer.citations} == {
        item["file_id"] for item in current["files"]
    }
    old_beta = next(citation.evidence for citation in old_answer.citations if citation.evidence.element.metadata["file_id"] == beta["file_id"])
    assert old_beta.snapshot.content_hash == beta["content_hash"]
    assert old_beta.element.text == "beta = 2\n"
    assert [item["name"] for item in new_manifest["files"]] == ["alpha.py"]
    latest = answer(api, uploads=context(new_manifest))
    assert {citation.evidence.snapshot.title for citation in latest.citations} == {"alpha.py"}
    assert manifest(api) == new_manifest


def test_new_session_question_completes_while_uploads_pin_the_lru_capacity(api):
    """An active upload session cannot make a new HTTP question evict itself before acquiring its lock."""
    api.settings.max_active_sessions = 1
    current = manifest(api, "busy-session")
    addition = staged(api, "alpha.py", "alpha = 1\n", session="busy-session")
    index_started, release_index = Event(), Event()

    def hold_index():
        index_started.set()
        assert release_index.wait(10), "test did not release the active upload"

    api.controls.before_embedding = hold_index
    with ThreadPoolExecutor(max_workers=1) as workers:
        upload = workers.submit(api.client.post, "/sessions/busy-session/uploads/sync",
                                json=sync_body(current, add=[addition]))
        try:
            assert index_started.wait(10), "upload did not reach indexing"
            result = final_response(api, query="hello", session_id="new-session")
            assert result["upload_manifest"]["files"] == []
        finally:
            release_index.set()
        committed = upload.result(timeout=10)
    assert committed.status_code == 200, committed.text
    assert committed.json()["manifest"]["epoch"] == current["epoch"]
    assert manifest(api, "busy-session") == committed.json()["manifest"]


def test_loopback_attachment_changes_reach_streamlit_client_and_versioned_answers(api, monkeypatch):
    """Real TCP requests carry additions, replacements and deletions into the next answer's exact sources."""
    from threading import Thread

    import uvicorn

    from src.app.web.streamlit_api_client import (
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
                    fastapi_url=url, session_id="session-a", uploads=current.context(),
                )))
                assert not [event for event in events if event.event == "error"]
                result = next(event.result for event in events if event.event == "final_response")
                assert result.upload_manifest == current
                return result.response

            empty = fetch_upload_manifest(url, "session-a")
            first = update(empty, add=[staged(api, "alpha.py", "alpha = 1\n")]).manifest
            attached = update(first, add=[staged(api, "beta.py", "beta = 2\n")]).manifest
            original = ask(attached)
            assert {item.evidence.snapshot.title for item in original.citations} == {"alpha.py", "beta.py"}
            assert update(attached, add=[staged(api, "alpha.py", "alpha = 1\n")]).manifest == attached

            replacement = staged(api, "alpha.py", "alpha = 10\n")
            with pytest.raises(UploadAPIError) as conflict:
                update(attached, add=[replacement])
            assert conflict.value.code == "UPLOAD_NAME_CONFLICT"
            alpha = next(item for item in attached.files if item.name == "alpha.py")
            replacement["replace_file_id"] = alpha.file_id
            replaced = update(attached, add=[replacement]).manifest
            current = ask(replaced)
            current_alpha = next(item.evidence for item in current.citations if item.evidence.snapshot.title == "alpha.py")
            old_alpha = next(item.evidence for item in original.citations if item.evidence.snapshot.title == "alpha.py")
            assert current_alpha.element.metadata["file_id"] == alpha.file_id
            assert current_alpha.excerpt == "alpha = 10"
            assert old_alpha.excerpt == "alpha = 1"
            assert current_alpha.snapshot.snapshot_id != old_alpha.snapshot.snapshot_id

            beta = next(item for item in replaced.files if item.name == "beta.py")
            removed = update(replaced, remove=[beta.file_id]).manifest
            assert {item.evidence.snapshot.title for item in ask(removed).citations} == {"alpha.py"}
            cleared = update(removed, clear=True).manifest
            assert cleared.files == []
            assert ask(cleared).citations == []
            assert fetch_upload_manifest(url, "session-a") == cleared
        finally:
            server.should_exit = True
            worker.join(timeout=10)
            assert not worker.is_alive(), "loopback server did not shut down"
