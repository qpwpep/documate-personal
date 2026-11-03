"""Real HTTP attachment transactions and graph retrieval, with external model boundaries replaced."""

from __future__ import annotations

import json
import hashlib
import asyncio
import socket
from collections.abc import AsyncIterator
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.embeddings import Embeddings
from langchain_core.messages import convert_to_messages
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.app.web.app import create_app
from src.core.answer_schema import AnswerDocument, AnswerResponse
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.settings import AppSettings
from src.runtime.agent_runtime.debug_collector import DebugCollector


class LocalEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0]


class LocalModelProvider:
    """Replace provider HTTP while keeping model parsing and wire validation real."""

    def __init__(self, controls):
        self.controls = controls

    def respond(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        messages = convert_to_messages(payload["messages"])
        schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
        if schema_name == "PlannerOutput":
            raw = next(message.content for message in messages if message.name == "request_context")
            context = json.loads(raw.split("\n", 1)[1])
            files = context["upload_files"]
            file_ids = (self.controls.planned_file_ids if self.controls.planned_file_ids is not None
                        else [item["file_id"] for item in files])
            parsed = {
                "use_retrieval": bool(files),
                "tasks": [{"route": "upload", "query": self.controls.search_query, "k": 4,
                           "requirement": {"file_ids": file_ids}}] if files else [],
                "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
            }
            parsed = InitialPlannerOutput.model_validate(parsed).model_dump(mode="json")
        elif schema_name == "RetrievalRewriteOutput":
            raise AssertionError("Upload-only requests must not replan retrieval")
        elif schema_name == "AnswerDocument":
            packet = json.loads(str(messages[-1].content).split("\n", 2)[2])
            blocks = [{"type": "code", "language": "python", "content": {
                "text": item["excerpt"], "basis": "excerpt", "refs": [item["id"]],
            }} for item in packet]
            parsed = {"blocks": blocks or [{"type": "paragraph", "content": [
                {"text": "No attached evidence.", "basis": "interaction", "refs": []},
            ]}]}
            parsed = AnswerDocument.model_validate(parsed).model_dump(mode="json")
        else:
            parsed = "Conversation summary."
        return httpx.Response(200, json={
            "id": "chatcmpl-upload-test", "object": "chat.completion", "created": 1,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": parsed if isinstance(parsed, str) else json.dumps(parsed),
            }}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        })


@pytest.fixture
def api(tmp_path, monkeypatch, request):
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
        upload_max_files=10, upload_max_file_mib=1, upload_max_total_mib=2,
        session_ttl_seconds=5, session_cleanup_interval_seconds=1,
        upload_search_mode=getattr(request, "param", "hybrid"),
    )
    controls = SimpleNamespace(planned_file_ids=None, search_query="compare values")
    monkeypatch.setattr("src.app.web.app.get_settings", lambda: settings)
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: LocalEmbeddings())
    model_provider = LocalModelProvider(controls)

    def decisions_provider(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("POST", "/v1/decisions")
        payload = json.loads(request.content)
        return httpx.Response(200, json={
            "model": payload["model"],
            "answers": [{"type": "predicate", "name": question["name"], "probability": 0.99}
                        for question in payload["questions"]],
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })

    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    connect = socket.socket.connect

    def local_connections_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network access is forbidden in upload API tests")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_connections_only)
    with (httpx.Client(transport=httpx.MockTransport(decisions_provider)) as provider_client,
          httpx.Client(transport=httpx.MockTransport(model_provider.respond)) as model_client):
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(
            model="gpt-4o-mini", api_key="test-key", base_url="https://provider.test/v1",
            http_client=model_client, max_retries=0, use_responses_api=False,
        ))
        monkeypatch.setattr("src.infra.llm.OpenAI", lambda **kwargs: OpenAI(
            **kwargs, http_client=provider_client, base_url="https://provider.test/v1",
        ))
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


def test_invalid_internal_debug_preserves_the_answer_and_attachment_manifest(api, monkeypatch):
    current = manifest(api)
    build = DebugCollector.build

    def corrupt_observation(self, **kwargs):
        debug = build(self, **kwargs)
        debug["observability_status"] = "unknown"
        return debug

    monkeypatch.setattr(DebugCollector, "build", corrupt_observation)
    final = final_response(api, uploads=context(current), include_debug=True)

    assert final["status"] == "completed"
    assert AnswerResponse.model_validate(final["response"]).content.blocks
    assert final["upload_manifest"] == current
    assert final["debug"]["observability_status"] == "failed"
    assert final["debug"]["missing_required_debug_fields"] == ["debug"]
    assert final["debug"]["error_codes"] == ["DEBUG_NORMALIZATION_FAILED"]


def test_http_session_boundary_preserves_case_identity_and_rejects_invalid_ids(api):
    current = manifest(api, "SESSION-A")

    assert manifest(api, "session-a") == current
    result = final_response(api, session_id="Session-A", uploads=context(current))
    assert result["status"] == "completed"
    assert result["upload_manifest"] == current
    assert api.client.get("/sessions/invalid.id/uploads").status_code == 422
    rejected = api.client.post("/agent/stream", json={
        "query": "Hello", "session_id": "invalid.id", "uploads": context(current),
    })
    assert rejected.status_code == 422
    assert manifest(api) == current


@pytest.mark.parametrize("transport", ["path", "browser"])
@pytest.mark.parametrize("conflict", ["duplicate_replacement", "replace_and_remove"])
def test_sync_rejects_conflicting_operations_without_changing_attachments(api, transport, conflict):
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])
    target = current["files"][0]["file_id"]
    replacement = (browser_addition(browser_stage(api, "alpha.py", b"alpha = 2\n"))
                   if transport == "browser" else staged(api, "alpha.py", "alpha = 2\n"))
    replacement["replace_file_id"] = target
    additions = [replacement, replacement] if conflict == "duplicate_replacement" else [replacement]
    removals = [target] if conflict == "replace_and_remove" else []

    response = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        current, add=additions, remove=removals,
    ))

    assert response.status_code == 422
    error = next(item for item in response.json()["detail"]
                 if "a file cannot be replaced or removed twice" in item["msg"])
    assert error["type"] == "value_error"
    if transport == "browser":
        assert error["loc"] == []
    else:
        assert error["loc"][0] == "body"
    assert manifest(api) == current


@pytest.mark.parametrize(("api", "expected_metric"), [
    ("baseline", "l2"), ("bm25", "bm25"), ("hybrid", "rrf"),
], indirect=["api"])
def test_configured_search_mode_reaches_http_answers_and_upload_index(api, expected_metric):
    """The configured search mode controls both the ready index and subsequent HTTP retrieval."""
    current = sync(api, add=[staged(api, "contract.py", "# 계약을 갱신한다.\nrenewal_days = 30\n")])
    api.controls.search_query = "계약"

    final = final_response(api, query="첨부의 계약 갱신을 설명해줘", uploads=context(current), include_debug=True)

    response = AnswerResponse.model_validate(final["response"])
    assert {citation.evidence.snapshot.title for citation in response.citations} == {"contract.py"}
    hits = final["debug"]["observed_hits"]
    assert hits
    assert {hit["score"]["metric"] for hit in hits} == {expected_metric}
    with api.client.app.state.session_store.locked_session("session-a") as (entry, _wait_ms):
        retriever = entry.agent.session.upload_retriever_handle.retriever
        if api.settings.upload_search_mode == "baseline":
            with pytest.raises(ValueError, match="built without BM25"):
                retriever.search("계약", mode="bm25")
        else:
            lexical_hits = retriever.search("계약", mode="bm25")
            assert [hit.evidence.snapshot.title for hit in lexical_hits] == ["contract.py"]
            assert all(hit.score.metric == "bm25" for hit in lexical_hits)


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


def test_stale_http_revision_rejects_mutations_and_question_without_changing_attachments(api):
    """A stale client cannot overwrite, reset or use a different attachment revision."""
    old = manifest(api)
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(old, clear=True))
    events = stream(api, query="Compare the attached files", uploads=context(old))

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


def test_session_recovers_committed_upload_after_lost_response(api, monkeypatch):
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
    current = sync(api, add=[staged(api, "beta.py", "beta = 2\n")])

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


def test_loopback_attachment_changes_reach_python_client_and_versioned_answers(api, monkeypatch):
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
            assert current_alpha.excerpt == "alpha = 10\n"
            assert old_alpha.excerpt == "alpha = 1\n"
            assert current_alpha.snapshot.snapshot_id != old_alpha.snapshot.snapshot_id
            assert current_alpha.excerpt == current_alpha.element.text
            assert old_alpha.excerpt == old_alpha.element.text
            assert "alpha = 10\n" in {block.content.text for block in current.content.blocks}
            assert "alpha = 1\n" in {block.content.text for block in original.content.blocks}
            assert all(check.support_status == "exact_match" for check in current.checks + original.checks)

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


def browser_stage(api, name: str, content: bytes, *, session: str = "session-a") -> dict[str, object]:
    response = api.client.post(f"/sessions/{session}/uploads/staging", files={"file": (name, content)})
    assert response.status_code == 200, response.text
    return response.json()


def browser_addition(upload: dict[str, object]) -> dict[str, object]:
    return {key: upload[key] for key in ("upload_id", "name", "content_hash")}


def test_browser_stage_keeps_manifest_unconfirmed_and_coexists_with_path_uploads(api):
    before = manifest(api)
    content = "# 한글 원문\nalpha = 7\n".encode("utf-8")

    upload = browser_stage(api, "alpha.py", content)

    assert set(upload) == {"upload_id", "name", "size_bytes", "content_hash"}
    assert len(upload["upload_id"]) == 32
    assert set(upload["upload_id"]) <= set("0123456789abcdef")
    assert upload["name"] == "alpha.py"
    assert upload["size_bytes"] == len(content)
    assert upload["content_hash"] == "sha256:" + hashlib.sha256(content).hexdigest()
    assert manifest(api) == before

    path_input = staged(api, "beta.py", "beta = 2\n")
    path_manifest = sync(api, add=[path_input])
    committed = sync(api, add=[browser_addition(upload)])

    assert Path(path_input["path"]).read_bytes() == b"beta = 2\n"
    assert committed["revision"] == path_manifest["revision"] + 1
    assert {item["name"] for item in committed["files"]} == {"alpha.py", "beta.py"}
    alpha = next(item for item in committed["files"] if item["name"] == "alpha.py")
    assert (alpha["size_bytes"], alpha["content_hash"]) == (len(content), upload["content_hash"])
    response = answer(api, uploads=context(committed))
    assert {item.evidence.snapshot.title for item in response.citations} == {"alpha.py", "beta.py"}
    assert content.decode() in {item.evidence.excerpt for item in response.citations}


def test_browser_upload_replacement_removal_and_clear_preserve_confirmed_identity(api):
    alpha = browser_stage(api, "alpha.py", b"alpha = 1\n")
    beta = browser_stage(api, "beta.py", b"beta = 2\n")
    current = sync(api, add=[browser_addition(alpha), browser_addition(beta)])
    alpha_file = next(item for item in current["files"] if item["name"] == "alpha.py")
    beta_file = next(item for item in current["files"] if item["name"] == "beta.py")
    replacement = browser_addition(browser_stage(api, "alpha.py", b"alpha = 10\n"))

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(current, add=[replacement]))

    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "UPLOAD_NAME_CONFLICT"
    assert manifest(api) == current
    retry_path = api.root / "uploads" / "session-a" / "staging" / replacement["upload_id"] / "alpha.py"
    assert retry_path.read_bytes() == b"alpha = 10\n"

    replacement["replace_file_id"] = alpha_file["file_id"]
    replaced = sync(api, add=[replacement])
    replaced_alpha = next(item for item in replaced["files"] if item["name"] == "alpha.py")
    assert replaced_alpha["file_id"] == alpha_file["file_id"]
    assert replaced_alpha["source_uri"] == alpha_file["source_uri"]
    assert replaced_alpha["content_hash"] == "sha256:" + hashlib.sha256(b"alpha = 10\n").hexdigest()
    removed = sync(api, remove=[beta_file["file_id"]])
    assert [item["name"] for item in removed["files"]] == ["alpha.py"]
    assert sync(api, clear=True)["files"] == []


def test_browser_upload_can_replace_a_full_attachment_without_exhausting_staging(api):
    api.settings.upload_max_total_mib = 1
    api.settings.session_ttl_seconds = 3600
    original_file_id = None
    for value in (1, 2, 3):
        notebook = json.dumps({
            "cells": [{"cell_type": "code", "execution_count": None, "id": "sample",
                       "metadata": {}, "outputs": [], "source": [f"alpha = {value}\n"]}],
            "metadata": {}, "nbformat": 4, "nbformat_minor": 5,
        }).encode("utf-8")
        # JSON padding reaches the byte limit without inflating the indexed code.
        content = notebook + b" " * (1_048_576 - len(notebook))
        upload = browser_stage(api, "sample.ipynb", content)
        addition = browser_addition(upload)
        if original_file_id is not None:
            addition["replace_file_id"] = original_file_id

        current = sync(api, add=[addition])

        assert current["revision"] == value
        assert len(current["files"]) == 1
        assert current["files"][0]["size_bytes"] == 1_048_576
        assert current["files"][0]["content_hash"] == "sha256:" + hashlib.sha256(content).hexdigest()
        if original_file_id is None:
            original_file_id = current["files"][0]["file_id"]
        assert current["files"][0]["file_id"] == original_file_id

    response = answer(api, uploads=context(current))
    assert any("alpha = 3" in citation.evidence.excerpt for citation in response.citations)
    assert not any("alpha = 1" in citation.evidence.excerpt for citation in response.citations)


@pytest.mark.parametrize("stale_field", ["epoch", "expected_revision"])
def test_browser_upload_rejects_stale_context_without_mutating_manifest(api, stale_field):
    old = manifest(api)
    current = sync(api, add=[staged(api, "alpha.py", "alpha = 1\n")])
    upload = browser_stage(api, "beta.py", b"beta = 2\n")
    body = sync_body(current, add=[browser_addition(upload)])
    body[stale_field] = "expired-epoch" if stale_field == "epoch" else old["revision"]

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=body)

    assert rejected.status_code == 409
    assert rejected.json()["detail"]["code"] == "UPLOAD_REVISION_CONFLICT"
    assert manifest(api) == current


def test_browser_upload_id_cannot_read_another_sessions_staged_file(api):
    before = manifest(api)
    upload = browser_stage(api, "secret.py", b"secret = 42\n", session="session-b")

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(
        before, add=[browser_addition(upload)],
    ))

    assert rejected.status_code == 422
    assert rejected.json()["detail"]["code"] == "UPLOAD_PATH_INVALID"
    assert manifest(api) == before
    assert manifest(api, "session-b")["files"] == []


@pytest.mark.parametrize("name,content,status,code", [
    ("../escape.py", b"value = 1\n", 422, "UPLOAD_NAME_INVALID"),
    ("CON.py", b"value = 1\n", 422, "UPLOAD_NAME_INVALID"),
    ("disabled.pdf", b"%PDF-1.4", 422, "UPLOAD_TYPE_INVALID"),
    ("empty.py", b"", 422, "UPLOAD_EMPTY"),
    ("large.py", b"x" * (1_048_576 + 1), 413, "UPLOAD_FILE_TOO_LARGE"),
], ids=["path", "reserved-name", "disabled-format", "empty", "too-large"])
def test_browser_stage_rejects_invalid_files_without_retaining_partial_bytes(api, name, content, status, code):
    before = manifest(api)

    rejected = api.client.post("/sessions/session-a/uploads/staging", files={"file": (name, content)})

    assert rejected.status_code == status
    assert rejected.json()["detail"]["code"] == code
    assert manifest(api) == before
    staging = api.root / "uploads" / "session-a" / "staging"
    assert not [path for path in staging.rglob("*") if path.is_file()]


@pytest.mark.parametrize("upload_id", ["../other", "F" * 32, "a" * 31])
def test_browser_sync_rejects_malformed_upload_identifiers(api, upload_id):
    before = manifest(api)
    addition = {"upload_id": upload_id, "name": "alpha.py", "content_hash": "sha256:" + "0" * 64}

    rejected = api.client.post("/sessions/session-a/uploads/sync", json=sync_body(before, add=[addition]))

    assert rejected.status_code == 422
    assert manifest(api) == before


def test_browser_stage_enforces_storage_cap_and_removes_incomplete_file(api):
    api.settings.session_ttl_seconds = 3600
    before = manifest(api)
    content = b"x" * 1_048_576
    for index in range(6):
        browser_stage(api, f"pending-{index}.py", content)
    staging = api.root / "uploads" / "session-a" / "staging"
    retained = {path: path.stat().st_size for path in staging.rglob("*") if path.is_file()}
    assert sum(retained.values()) == 6_291_456

    rejected = api.client.post("/sessions/session-a/uploads/staging", files={"file": ("overflow.py", b"value = 1\n")})

    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == "UPLOAD_STORAGE_FULL"
    assert {path: path.stat().st_size for path in staging.rglob("*") if path.is_file()} == retained
    assert manifest(api) == before


def test_browser_sync_retry_after_staging_cleanup_returns_current_manifest(api):
    api.settings.session_ttl_seconds = 3600
    before = manifest(api)
    upload = browser_stage(api, "alpha.py", b"alpha = 1\n")
    pending = browser_stage(api, "pending.py", b"pending = 1\n")
    body = sync_body(before, add=[browser_addition(upload)], operation_id="browser-operation")
    committed = api.client.post("/sessions/session-a/uploads/sync", json=body)
    assert committed.status_code == 200, committed.text
    staging = api.root / "uploads" / "session-a" / "staging"
    assert not list(staging.rglob("alpha.py"))
    assert (staging / pending["upload_id"] / "pending.py").read_bytes() == b"pending = 1\n"
    current = sync(api, add=[staged(api, "beta.py", "beta = 2\n")])

    replayed = api.client.post("/sessions/session-a/uploads/sync", json=body)

    assert replayed.status_code == 200, replayed.text
    assert replayed.json() == current
    assert manifest(api) == current
    assert current["revision"] == before["revision"] + 2


def test_browser_stage_rejects_oversized_content_length_before_accepting_file(api):
    before = manifest(api)
    body = (
        b'--upload-boundary\r\nContent-Disposition: form-data; name="file"; filename="alpha.py"\r\n'
        b'Content-Type: text/plain\r\n\r\nalpha = 1\n\r\n--upload-boundary--\r\n'
    )

    async def submit() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.client.app), base_url="http://testserver") as client:
            return await client.post("/sessions/session-a/uploads/staging", content=body, headers={
                "Content-Type": "multipart/form-data; boundary=upload-boundary",
                "Content-Length": str(1_048_576 + 65_536 + 1),
            })

    rejected = asyncio.run(submit())

    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == "UPLOAD_FILE_TOO_LARGE"
    assert manifest(api) == before
    staging = api.root / "uploads" / "session-a" / "staging"
    assert not [path for path in staging.rglob("*") if path.is_file()]


def test_browser_stage_stops_chunked_body_at_ingress_limit_before_parser_reads_all_bytes(api):
    before = manifest(api)
    chunks_consumed = 0

    async def body() -> AsyncIterator[bytes]:
        nonlocal chunks_consumed
        yield (
            b'--upload-boundary\r\nContent-Disposition: form-data; name="file"; filename="large.py"\r\n'
            b'Content-Type: application/octet-stream\r\n\r\n'
        )
        for _ in range(10):
            chunks_consumed += 1
            yield b"x" * 524_288
        yield b"\r\n--upload-boundary--\r\n"

    async def submit() -> httpx.Response:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.client.app), base_url="http://testserver") as client:
            return await client.post("/sessions/session-a/uploads/staging", content=body(), headers={
                "Content-Type": "multipart/form-data; boundary=upload-boundary",
            })

    rejected = asyncio.run(submit())

    assert "content-length" not in rejected.request.headers
    assert rejected.status_code == 413
    assert rejected.json()["detail"]["code"] == "UPLOAD_FILE_TOO_LARGE"
    # The third 512 KiB block crosses 1 MiB plus 64 KiB of multipart overhead.
    assert 0 < chunks_consumed <= 3
    assert manifest(api) == before
    staging = api.root / "uploads" / "session-a" / "staging"
    assert not [path for path in staging.rglob("*") if path.is_file()]
