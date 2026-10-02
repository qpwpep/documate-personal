from __future__ import annotations

import hashlib
import threading
import time
import unittest
import json
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pytest

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

from src.app.agent_manager import AgentFlowManager
from src.core.contracts import ResponseState
from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.planner_schema import InitialPlannerOutput, RetrievalPlanOutput
from tests.web.answer_fixtures import answer_response
from src.infra.settings import AppSettings
from src.app.web.agent_request_support import build_session_metadata_snapshot
from src.app.web.session_store import InMemorySessionStore, SessionEntry
from src.app.web.schemas import AgentRequest
from src.app.web.upload_service import UploadService
from src.core.request_contracts import WireRequestContract
from src.core.uploads import UploadAddition, UploadContext, UploadSyncRequest


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class _CapturingGraph:
    def __init__(self):
        self.states: list[dict] = []

    def stream(self, state: dict, *, stream_mode: str):
        assert stream_mode == "values"
        self.states.append(dict(state))
        runtime = state["runtime"]
        yield {
            "route_decisions": [],
            "messages": [
                HumanMessage(content=runtime.user_input),
                AIMessage(content="ok"),
            ],
            "response": ResponseState(result=answer_response("ok")),
        }


class _ExplodingGraph:
    def stream(self, state: dict, *, stream_mode: str):
        if state["runtime"].retriever is not None:
            state["runtime"].retriever.invoke("probe")
        raise RuntimeError("boom")


class _SlowCapturingGraph:
    def __init__(self):
        self.max_concurrent = 0
        self._current = 0
        self._lock = threading.Lock()

    def stream(self, state: dict, *, stream_mode: str):
        runtime = state["runtime"]
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
        try:
            time.sleep(0.05)
            yield {
                "route_decisions": [],
                "messages": [
                    HumanMessage(content=runtime.user_input),
                    AIMessage(content="ok"),
                ],
                "response": ResponseState(result=answer_response("ok")),
            }
        finally:
            with self._lock:
                self._current -= 1


def _make_manager(graph: _CapturingGraph) -> AgentFlowManager:
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
    manager.graph = graph
    manager.messages = []
    manager.session_metadata = {"slack_recipient": None}
    return manager


class UploadSessionIsolationTest(unittest.TestCase):
    def test_agent_manager_passes_session_metadata_to_graph_and_clears_on_close(self) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)
        manager.set_session_metadata(
            {
                "slack_recipient": {"kind": "channel", "value": "C123BENCH"}
            }
        )

        manager.run_agent_flow("send this to slack")

        self.assertEqual(
            graph.states[-1]["runtime"].session_metadata.slack_recipient.value,
            "C123BENCH",
        )

        manager.close()

        self.assertIsNone(manager.session_metadata.slack_recipient)
        self.assertEqual(manager.messages, [])

    def test_session_metadata_snapshot_replaces_previous_slack_destination(self) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)

        manager.set_session_metadata(
            build_session_metadata_snapshot(
                AgentRequest(
                    query="share this to slack",
                    session_id="demo-session",
                    uploads=manager._ensure_session().upload_manifest().context(),
                    slack_recipient={"kind": "channel", "value": "C123BENCH"},
                )
            )
        )
        manager.run_agent_flow("share this to slack")

        self.assertEqual(
            graph.states[-1]["runtime"].session_metadata.slack_recipient.value,
            "C123BENCH",
        )

        manager.set_session_metadata(
            build_session_metadata_snapshot(
                AgentRequest(
                    query="share this to slack",
                    session_id="demo-session",
                    uploads=manager._ensure_session().upload_manifest().context(),
                )
            )
        )
        manager.run_agent_flow("share this to slack")

        self.assertIsNone(graph.states[-1]["runtime"].session_metadata.slack_recipient)
        self.assertFalse(any(message.__class__.__name__ == "SystemMessage" for message in manager.messages))


class SessionStoreCleanupTest(unittest.TestCase):
    def test_cleanup_expired_sessions_skips_active_session(self) -> None:
        store = InMemorySessionStore(
            settings=AppSettings(openai_api_key="test-key", tavily_api_key="test"),
            agent_factory=lambda: _make_manager(_CapturingGraph()),
        )
        busy_agent = _make_manager(_CapturingGraph())
        store.active_agents["busy"] = SessionEntry(
            agent=busy_agent,
            last_accessed_monotonic=0.0,
            active_request_count=1,
        )

        removed = store.cleanup_expired(now=100.0, ttl_seconds=10)

        self.assertEqual(removed, 0)
        self.assertIn("busy", store.active_agents)

    def test_cleanup_expired_sessions_calls_agent_close(self) -> None:
        store = InMemorySessionStore(
            settings=AppSettings(openai_api_key="test-key", tavily_api_key="test"),
            agent_factory=lambda: _make_manager(_CapturingGraph()),
        )
        stale_agent = _make_manager(_CapturingGraph())
        fresh_agent = _make_manager(_CapturingGraph())
        with patch.object(stale_agent, "close") as stale_close, patch.object(
            fresh_agent, "close"
        ) as fresh_close:
            store.active_agents["stale"] = SessionEntry(
                agent=stale_agent,
                last_accessed_monotonic=0.0,
            )
            store.active_agents["fresh"] = SessionEntry(
                agent=fresh_agent,
                last_accessed_monotonic=95.0,
            )

            removed = store.cleanup_expired(now=100.0, ttl_seconds=10)

            self.assertEqual(removed, 1)
            stale_close.assert_called_once_with()
            fresh_close.assert_not_called()
            self.assertNotIn("stale", store.active_agents)
            self.assertIn("fresh", store.active_agents)

    def test_lru_eviction_calls_agent_close(self) -> None:
        store = InMemorySessionStore(
            settings=AppSettings(openai_api_key="test-key", tavily_api_key="test"),
            agent_factory=lambda: _make_manager(_CapturingGraph()),
        )
        oldest_agent = _make_manager(_CapturingGraph())
        newest_agent = _make_manager(_CapturingGraph())
        with patch.object(oldest_agent, "close") as oldest_close, patch.object(
            newest_agent, "close"
        ) as newest_close:
            store.active_agents["oldest"] = SessionEntry(
                agent=oldest_agent,
                last_accessed_monotonic=1.0,
            )
            store.active_agents["newest"] = SessionEntry(
                agent=newest_agent,
                last_accessed_monotonic=2.0,
            )

            evicted = store.evict_lru_if_needed(max_active_sessions=1)

            self.assertEqual(evicted, 1)
            oldest_close.assert_called_once_with()
            newest_close.assert_not_called()
            self.assertNotIn("oldest", store.active_agents)
            self.assertIn("newest", store.active_agents)

    def test_lru_eviction_skips_locked_sessions(self) -> None:
        store = InMemorySessionStore(
            settings=AppSettings(openai_api_key="test-key", tavily_api_key="test"),
            agent_factory=lambda: _make_manager(_CapturingGraph()),
        )
        locked_agent = _make_manager(_CapturingGraph())
        unlocked_agent = _make_manager(_CapturingGraph())
        with patch.object(locked_agent, "close") as locked_close, patch.object(
            unlocked_agent, "close"
        ) as unlocked_close:
            store.active_agents["locked"] = SessionEntry(
                agent=locked_agent,
                last_accessed_monotonic=1.0,
            )
            store.active_agents["unlocked"] = SessionEntry(
                agent=unlocked_agent,
                last_accessed_monotonic=2.0,
            )

            store.active_agents["locked"].request_lock.acquire()
            try:
                evicted = store.evict_lru_if_needed(max_active_sessions=1)
            finally:
                store.active_agents["locked"].request_lock.release()

            self.assertEqual(evicted, 1)
            locked_close.assert_not_called()
            unlocked_close.assert_called_once_with()
            self.assertIn("locked", store.active_agents)
            self.assertNotIn("unlocked", store.active_agents)

    def test_lru_eviction_skips_active_session(self) -> None:
        store = InMemorySessionStore(
            settings=AppSettings(openai_api_key="test-key", tavily_api_key="test"),
            agent_factory=lambda: _make_manager(_CapturingGraph()),
        )
        oldest_agent = _make_manager(_CapturingGraph())
        newest_agent = _make_manager(_CapturingGraph())
        with patch.object(oldest_agent, "close") as oldest_close, patch.object(
            newest_agent, "close"
        ) as newest_close:
            store.active_agents["oldest"] = SessionEntry(
                agent=oldest_agent,
                last_accessed_monotonic=1.0,
                active_request_count=1,
            )
            store.active_agents["newest"] = SessionEntry(
                agent=newest_agent,
                last_accessed_monotonic=2.0,
            )

            evicted = store.evict_lru_if_needed(max_active_sessions=1)

            self.assertEqual(evicted, 1)
            oldest_close.assert_not_called()
            newest_close.assert_called_once_with()
            self.assertIn("oldest", store.active_agents)
            self.assertNotIn("newest", store.active_agents)

    def test_session_request_lock_serializes_same_session_requests(self) -> None:
        store = InMemorySessionStore(
            settings=AppSettings(openai_api_key="test-key", tavily_api_key="test"),
            agent_factory=lambda: _make_manager(_SlowCapturingGraph()),
        )
        session_entry = store.get_or_create_entry("demo-session")
        graph = session_entry.agent.graph
        barrier = threading.Barrier(3)
        results: list[tuple[int, str]] = []

        def worker(index: int) -> None:
            barrier.wait()
            with store.locked_session("demo-session") as (entry, session_lock_wait_ms):
                agent_answer = entry.agent.run_agent_flow(f"question-{index}")
            results.append((session_lock_wait_ms, export_answer_text(AnswerResponse.model_validate(agent_answer["response"]))))

        threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(2)]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()

        self.assertEqual(graph.max_concurrent, 1)
        self.assertEqual(sorted(answer for _, answer in results), ["ok", "ok"])
        self.assertTrue(any(wait_ms > 0 for wait_ms, _ in results))
        self.assertEqual(session_entry.active_request_count, 0)


class _UploadContractChatModel:
    """Deterministic model boundary for real manager, graph and retrieval calls."""

    def __init__(self, schema_name=""):
        self.schema_name = schema_name

    def with_structured_output(self, schema, **_kwargs):
        return _UploadContractChatModel(schema["name"])

    def invoke(self, messages, **_kwargs):
        if self.schema_name in {"PlannerOutput", "RetrievalPlanOutput"}:
            parsed = {
                "use_retrieval": True,
                "tasks": [{"route": "upload", "query": "read source", "k": 4}],
                "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
            }
            if self.schema_name == "RetrievalPlanOutput":
                parsed.pop("request_contract")
                parsed = RetrievalPlanOutput.model_validate(parsed).model_dump(mode="json")
            else:
                parsed = InitialPlannerOutput.model_validate(parsed).model_dump(mode="json")
        elif self.schema_name == "AnswerDocument":
            packet = json.loads(str(messages[-1].content).split("\n", 2)[2])
            parsed = {"blocks": [{"type": "code", "language": "python", "content": {
                "text": item["excerpt"], "basis": "excerpt", "refs": [item["id"]],
            }} for item in packet]}
        else:
            return AIMessage(content="Conversation summary.")
        return {"parsed": parsed, "parsing_error": None, "raw": AIMessage(content="")}


@pytest.fixture
def managed_upload_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.infra.upload_storage.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.app.web.upload_service.get_project_root_path", lambda: tmp_path)
    monkeypatch.setattr("src.infra.tools.local_rag.client.build_openai_embeddings", lambda _key: _FakeEmbeddings())
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **_kwargs: _FakeEmbeddings())
    monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **_kwargs: _UploadContractChatModel())
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    settings = AppSettings(_env_file=None, openai_api_key="test-key", tavily_api_key="test")
    store = InMemorySessionStore(settings, lambda: AgentFlowManager(settings))
    service = UploadService(settings=settings, session_store=store)
    source = tmp_path / "uploads" / "session-a" / "staging" / "source.py"
    source.parent.mkdir(parents=True)
    source.write_text("managed_value = 1\n", encoding="utf-8")
    before = service.get_manifest("session-a")
    before = service.sync("session-a", UploadSyncRequest(
        epoch=before.epoch, expected_revision=before.revision, operation_id=uuid4().hex,
        add=[UploadAddition(path=str(source), name=source.name,
                            content_hash="sha256:" + hashlib.sha256(source.read_bytes()).hexdigest())],
    ))
    agent = store.get_or_create("session-a")
    owned = Path(agent._ensure_session().upload_records[0].path)
    try:
        yield agent, service, before, source, owned
    finally:
        store.close_all()


def test_questions_preserve_committed_uploads_without_resubmitting_attachment_context(managed_upload_agent):
    """The runner reads the committed attachment set and never treats a question as a mutation."""
    agent, service, before, _source, owned = managed_upload_agent

    response = AnswerResponse.model_validate(agent.run_agent_flow("read uploads")["response"])

    assert {item.evidence.snapshot.title for item in response.citations} == {"source.py"}
    assert service.get_manifest("session-a") == before
    assert owned.is_file()


def test_failed_answer_preserves_the_committed_upload_for_the_next_question(managed_upload_agent):
    """An answer failure cannot retire the independently committed attachment transaction."""
    agent, service, before, _source, owned = managed_upload_agent
    graph = agent.graph
    agent.graph = _ExplodingGraph()

    failed = agent.run_agent_flow("read uploads")

    assert failed["debug"]["observability_status"] == "failed"
    assert service.get_manifest("session-a") == before
    assert owned.is_file()
    agent.graph = graph
    recovered = AnswerResponse.model_validate(agent.run_agent_flow("read uploads")["response"])
    assert {item.evidence.snapshot.title for item in recovered.citations} == {"source.py"}


def test_session_rejects_stale_upload_context_without_mutating_attachments(managed_upload_agent):
    agent, service, before, _source, owned = managed_upload_agent
    session = agent._ensure_session()
    stale_context = UploadContext(epoch=before.epoch, revision=before.revision - 1)

    with pytest.raises(ValueError, match="UPLOAD_REVISION_CONFLICT"):
        session.require_upload_context(stale_context)

    assert service.get_manifest("session-a") == before
    assert owned.is_file()
    session.require_upload_context(before.context())
    current = agent.run_agent_flow("read uploads")
    assert {item.evidence.snapshot.title for item in AnswerResponse.model_validate(current["response"]).citations} == {"source.py"}


@pytest.mark.parametrize("missing", ["records", "index"])
def test_incomplete_attachment_commit_preserves_the_active_search(managed_upload_agent, missing):
    agent, service, before, _source, owned = managed_upload_agent
    session = agent._ensure_session()

    with pytest.raises(ValueError, match="both records and their index"):
        session.replace_upload_resources(
            "session-a", () if missing == "records" else session.upload_records,
            None if missing == "index" else session.upload_retriever_handle,
        )

    assert service.get_manifest("session-a") == before
    assert owned.is_file()
    response = AnswerResponse.model_validate(agent.run_agent_flow("read uploads")["response"])
    assert {item.evidence.snapshot.title for item in response.citations} == {"source.py"}


def test_index_for_another_catalog_cannot_replace_committed_attachments(managed_upload_agent):
    from src.infra.tools.local_rag import build_upload_retriever

    agent, service, before, _source, owned = managed_upload_agent
    session = agent._ensure_session()
    records = [record.model_copy(update={"name": "another.py"}) for record in session.upload_records]
    candidate = build_upload_retriever(records, session_id="session-a", generation=uuid4().hex, api_key="test-key")
    try:
        with pytest.raises(ValueError, match="complete attachment catalog"):
            session.replace_upload_resources("session-a", session.upload_records, candidate)
    finally:
        candidate.cleanup()

    assert service.get_manifest("session-a") == before
    assert owned.is_file()
    response = AnswerResponse.model_validate(agent.run_agent_flow("read uploads")["response"])
    assert {item.evidence.snapshot.title for item in response.citations} == {"source.py"}


if __name__ == "__main__":
    unittest.main()
