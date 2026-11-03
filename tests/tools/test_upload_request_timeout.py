"""Query embedding observes the request deadline without changing the session index."""

import hashlib
import json
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langchain_openai import OpenAIEmbeddings

from src.core.llm_errors import LLMCallError
from src.core.planner_schema import RetrievalRequirement
from src.core.uploads import UploadRecord
from src.infra import llm_boundary
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool


@pytest.fixture
def indexed_upload(tmp_path, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(llm_boundary, "time", SimpleNamespace(monotonic=lambda: clock.now))
    behavior = SimpleNamespace(expires_during_call=False, timeout=False)
    requests = []

    def respond(request):
        requests.append(request)
        if behavior.expires_during_call:
            clock.now = 2
        if behavior.timeout:
            raise httpx.ReadTimeout("Embedding read timed out", request=request)
        count = len(json.loads(request.content)["input"])
        return httpx.Response(200, json={
            "object": "list", "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": index, "embedding": [1.0, 0.0]}
                     for index in range(count)],
            "usage": {"prompt_tokens": count, "total_tokens": count},
        })

    body = b"def load():\n    return 'document contents'\n"
    path = tmp_path / "sample.py"
    path.write_bytes(body)
    file_id = uuid4().hex
    record = UploadRecord(
        file_id=file_id, name=path.name, path=str(path), size_bytes=len(body),
        content_hash="sha256:" + hashlib.sha256(body).hexdigest(), source_uri=f"upload://deadline/{file_id}",
    )
    with httpx.Client(transport=httpx.MockTransport(respond)) as http_client:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", api_key="test", request_timeout=20,
            max_retries=0, http_client=http_client,
        )
        handle = build_upload_retriever(
            [record], session_id="deadline", generation=uuid4().hex, embeddings=embeddings, build_bm25=False,
        )
        requests.clear()
        try:
            yield SimpleNamespace(
                retriever=handle.retriever, file_id=file_id, clock=clock,
                behavior=behavior, requests=requests,
            )
        finally:
            handle.cleanup()


@pytest.mark.parametrize("scoped", [False, True])
def test_query_embedding_timeout_is_per_request_and_keeps_later_session_search(indexed_upload, scoped):
    fixture = indexed_upload
    search = build_upload_search_tool(default_mode="baseline")
    requirement = RetrievalRequirement(file_ids=[fixture.file_id] if scoped else [])

    with llm_boundary.request_deadline(2):
        fixture.clock.now = 0.5
        first = search("load contents", retriever=fixture.retriever, requirement=requirement)
    second = search("load again", retriever=fixture.retriever, requirement=requirement)

    assert first.hits and second.hits
    assert first.hits[0].evidence.id == second.hits[0].evidence.id
    assert [request.extensions["timeout"]["read"] for request in fixture.requests] == [1.5, 20]


@pytest.mark.parametrize("provider_timeout", [False, True])
def test_late_query_embedding_preserves_request_timeout_instead_of_local_rag_failure(indexed_upload, provider_timeout):
    fixture = indexed_upload
    fixture.behavior.expires_during_call = True
    fixture.behavior.timeout = provider_timeout
    search = build_upload_search_tool(default_mode="baseline")

    with llm_boundary.request_deadline(2):
        with pytest.raises(LLMCallError) as caught:
            search("load contents", retriever=fixture.retriever, requirement=RetrievalRequirement())

    assert caught.value.problem.code == "request_timeout"
    assert len(fixture.requests) == 1


def test_expired_request_does_not_start_query_embedding(indexed_upload):
    fixture = indexed_upload
    search = build_upload_search_tool(default_mode="baseline")

    with llm_boundary.request_deadline(2):
        fixture.clock.now = 2
        with pytest.raises(LLMCallError) as caught:
            search("load contents", retriever=fixture.retriever, requirement=RetrievalRequirement())

    assert caught.value.problem.code == "request_timeout"
    assert fixture.requests == []
