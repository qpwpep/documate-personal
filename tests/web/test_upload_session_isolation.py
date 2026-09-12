from __future__ import annotations

import tempfile
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
from tests.web.answer_fixtures import answer_response
from src.infra.settings import AppSettings
from src.infra.tools.local_rag import build_temp_retriever
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

    def invoke(self, state: dict) -> dict:
        self.states.append(dict(state))
        runtime = state["runtime"]
        return {
            "messages": [
                HumanMessage(content=runtime.user_input),
                AIMessage(content="ok"),
            ],
            "response": ResponseState(result=answer_response("ok")),
        }


class _ResolvingGraph(_CapturingGraph):
    def invoke(self, state: dict) -> dict:
        runtime = state["runtime"]
        if runtime.retriever is not None:
            runtime.retriever.invoke("probe")
        return super().invoke(state)


class _ExplodingGraph:
    def invoke(self, state: dict) -> dict:
        if state["runtime"].retriever is not None:
            state["runtime"].retriever.invoke("probe")
        raise RuntimeError("boom")


class _SlowCapturingGraph:
    def __init__(self):
        self.max_concurrent = 0
        self._current = 0
        self._lock = threading.Lock()

    def invoke(self, state: dict) -> dict:
        runtime = state["runtime"]
        with self._lock:
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)
        try:
            time.sleep(0.05)
            return {
                "messages": [
                    HumanMessage(content=runtime.user_input),
                    AIMessage(content="ok"),
                ],
                "response": ResponseState(result=answer_response("ok")),
            }
        finally:
            with self._lock:
                self._current -= 1


class _FakeHandle:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.retriever = self
        self.cleanup_calls = 0

    @property
    def vectorstore(self):
        return None

    def invoke(self, _query: str):
        return []

    def cleanup(self) -> None:
        self.cleanup_calls += 1


def _make_manager(graph: _CapturingGraph) -> AgentFlowManager:
    manager = AgentFlowManager.__new__(AgentFlowManager)
    manager.settings = AppSettings(openai_api_key="test-key", tavily_api_key="test")
    manager.graph = graph
    manager.messages = []
    manager.session_metadata = {"slack_destination": None}
    manager.upload_retriever_handle = None
    manager.upload_file_path = None
    return manager


class UploadSessionIsolationTest(unittest.TestCase):
    def _upload(self, filename: str = "file.py", content: str = "value = 1\n") -> str:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "uploads" / "session" / filename
        path.parent.mkdir(parents=True)
        path.write_text(content, encoding="utf-8")
        return str(path)

    @patch("src.infra.tools.local_rag.client.build_openai_embeddings", return_value=_FakeEmbeddings())
    def test_build_temp_retriever_isolates_per_session_collection(self, _mock_embeddings) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            uploads_root = Path(tmp_dir) / "uploads"
            path_one = uploads_root / "session-one" / "sample_one.py"
            path_two = uploads_root / "session-two" / "sample_two.py"

            path_one.parent.mkdir(parents=True, exist_ok=True)
            path_two.parent.mkdir(parents=True, exist_ok=True)
            path_one.write_text("alpha session one", encoding="utf-8")
            path_two.write_text("beta session two", encoding="utf-8")

            handle_one = build_temp_retriever(str(path_one), api_key="test-key")
            handle_two = build_temp_retriever(str(path_two), api_key="test-key")
            self.addCleanup(handle_one.cleanup)
            self.addCleanup(handle_two.cleanup)

            metadatas = handle_two.retriever.vectorstore.get().get("metadatas", [])
            sources = [item.get("source") for item in metadatas]

            self.assertEqual(handle_one.collection_name, "upload-session-session-one")
            self.assertEqual(handle_two.collection_name, "upload-session-session-two")
            self.assertEqual(sources, [str(path_two)])

    @patch("src.infra.tools.local_rag.client.build_openai_embeddings", return_value=_FakeEmbeddings())
    def test_failed_legacy_cleanup_cannot_later_delete_a_reused_collection(self, _mock_embeddings) -> None:
        """Legacy collection names are reused, so failed disposal must not become a delayed deletion."""
        original = build_temp_retriever(self._upload("old.py"), api_key="test-key")
        database = original._vectorstore._client
        with patch.object(database, "delete_collection", side_effect=RuntimeError("unavailable")):
            with self.assertRaises(RuntimeError):
                original.cleanup()
        current = build_temp_retriever(self._upload("new.py", "value = 2\n"), api_key="test-key")
        try:
            original.cleanup()
            self.assertEqual(database.get_collection(current.collection_name).name, current.collection_name)
        finally:
            if current.collection_name in {collection.name for collection in database.list_collections()}:
                current.cleanup()

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_previous_handle_when_upload_changes(
        self,
        mock_build_temp_retriever,
    ) -> None:
        graph = _ResolvingGraph()
        manager = _make_manager(graph)
        handle_one = _FakeHandle("upload-session-session")
        handle_two = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.side_effect = [handle_one, handle_two]

        manager.run_agent_flow("first", upload_file_path=self._upload("file_one.py"))
        manager.run_agent_flow("second", upload_file_path=self._upload("file_two.py"))

        self.assertEqual(handle_one.cleanup_calls, 1)
        self.assertIs(manager.upload_retriever_handle, handle_two)
        self.assertIsNotNone(graph.states[-1]["runtime"].retriever)

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_agent_manager_overlaps_upload_retriever_build_with_graph(
        self,
        mock_build_temp_retriever,
    ) -> None:
        graph_started = threading.Event()
        handle = _FakeHandle("upload-session-session")
        test_case = self

        def build_retriever(*_args, **_kwargs):
            self.assertTrue(graph_started.wait(timeout=1.0))
            return handle

        class _Graph(_CapturingGraph):
            def invoke(self, state: dict) -> dict:
                self.states.append(dict(state))
                test_case.assertIsNone(manager.upload_retriever_handle)
                test_case.assertIsNotNone(state["runtime"].retriever)
                graph_started.set()
                state["runtime"].retriever.invoke("probe")
                return {
                    "messages": [
                        HumanMessage(content=state["runtime"].user_input),
                        AIMessage(content="ok"),
                    ],
                    "response": ResponseState(result=answer_response("ok")),
                }

        graph = _Graph()
        manager = _make_manager(graph)
        mock_build_temp_retriever.side_effect = build_retriever

        manager.run_agent_flow("with upload", upload_file_path=self._upload())

        self.assertIs(manager.upload_retriever_handle, handle)
        self.assertEqual(mock_build_temp_retriever.call_count, 1)

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_handle_when_upload_removed(self, mock_build_temp_retriever) -> None:
        graph = _ResolvingGraph()
        manager = _make_manager(graph)
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        manager.run_agent_flow("with upload", upload_file_path=self._upload())
        manager.run_agent_flow("without upload")

        self.assertEqual(handle.cleanup_calls, 1)
        self.assertIsNone(manager.upload_retriever_handle)
        self.assertIsNone(graph.states[-1]["runtime"].retriever)

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_memory_summary_reaches_plain_new_and_reused_upload_paths(
        self,
        mock_build_temp_retriever,
    ) -> None:
        graph = _ResolvingGraph()
        manager = _make_manager(graph)
        manager.memory_summary = "stable upload summary"
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        upload_path = self._upload()
        manager.run_agent_flow("new upload", upload_file_path=upload_path)
        manager.run_agent_flow("reuse upload", upload_file_path=upload_path)
        manager.run_agent_flow("plain request")

        self.assertEqual(
            [state["runtime"].memory_summary for state in graph.states],
            ["stable upload summary"] * 3,
        )

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_handle_on_exit(self, mock_build_temp_retriever) -> None:
        graph = _ResolvingGraph()
        manager = _make_manager(graph)
        manager.memory_summary = "summary to clear"
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        borrowed_path = self._upload()
        manager.run_agent_flow("with upload", upload_file_path=borrowed_path)
        manager.run_agent_flow("exit")

        self.assertEqual(handle.cleanup_calls, 1)
        self.assertIsNone(manager.upload_retriever_handle)
        self.assertEqual(manager.messages, [])
        self.assertIsNone(manager.memory_summary)
        self.assertTrue(Path(borrowed_path).is_file())

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_handle_on_exception(self, mock_build_temp_retriever) -> None:
        manager = _make_manager(_ExplodingGraph())
        manager.messages = [
            HumanMessage(content="stable request"),
            AIMessage(content="stable answer"),
        ]
        manager.memory_summary = "stable summary"
        before = manager._ensure_session().snapshot_conversation_memory()
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        result = manager.run_agent_flow("with upload", upload_file_path=self._upload())

        self.assertEqual(export_answer_text(AnswerResponse.model_validate(result["response"])), "boom")
        self.assertEqual(handle.cleanup_calls, 1)
        self.assertIsNone(manager.upload_retriever_handle)
        self.assertEqual(manager._ensure_session().snapshot_conversation_memory(), before)

    @patch("src.app.agent_manager.build_temp_retriever")
    def test_replacing_bytes_at_the_same_upload_path_rebuilds_retrieval(self, build_retriever) -> None:
        """A new document revision replaces retrieval even when its filename is unchanged."""
        graph = _ResolvingGraph()
        manager = _make_manager(graph)
        first = _FakeHandle("first-version")
        second = _FakeHandle("second-version")
        build_retriever.side_effect = [first, second]
        upload_path = self._upload(content="value = 1\n")

        manager.run_agent_flow("initial", upload_file_path=upload_path)
        manager.run_agent_flow("same bytes", upload_file_path=upload_path)
        self.assertIs(manager.upload_retriever_handle, first)
        Path(upload_path).write_text("value = 2\n", encoding="utf-8")
        manager.run_agent_flow("new bytes", upload_file_path=upload_path)

        self.assertIs(manager.upload_retriever_handle, second)
        self.assertEqual(first.cleanup_calls, 1)
        self.assertEqual([state["runtime"].user_input for state in graph.states], ["initial", "same bytes", "new bytes"])

    def test_agent_manager_passes_session_metadata_to_graph_and_clears_on_close(self) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)
        manager.set_session_metadata(
            {
                "slack_destination": {
                    "channel_id": "C123BENCH",
                    "user_id": None,
                    "email": None,
                }
            }
        )

        manager.run_agent_flow("send this to slack")

        self.assertEqual(
            graph.states[-1]["runtime"].session_metadata.slack_destination.channel_id,
            "C123BENCH",
        )

        manager.close()

        self.assertIsNone(manager.session_metadata.slack_destination)
        self.assertEqual(manager.messages, [])

    def test_session_metadata_snapshot_replaces_previous_slack_destination(self) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)

        manager.set_session_metadata(
            build_session_metadata_snapshot(
                AgentRequest(
                    query="share this to slack",
                    session_id="demo-session",
                    slack_channel_id="C123BENCH",
                )
            )
        )
        manager.run_agent_flow("share this to slack")

        self.assertEqual(
            graph.states[-1]["runtime"].session_metadata.slack_destination.channel_id,
            "C123BENCH",
        )

        manager.set_session_metadata(
            build_session_metadata_snapshot(
                AgentRequest(
                    query="share this to slack",
                    session_id="demo-session",
                )
            )
        )
        manager.run_agent_flow("share this to slack")

        self.assertIsNone(graph.states[-1]["runtime"].session_metadata.slack_destination)
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
            _agent, agent_answer, session_lock_wait_ms, _manifest = store.run_session_request(
                session_id="demo-session",
                session_metadata=build_session_metadata_snapshot(
                    AgentRequest(query=f"question-{index}", session_id="demo-session")
                ),
                user_input=f"question-{index}",
                upload_file_path=None,
            )
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

    def invoke(self, messages):
        if self.schema_name == "PlannerOutput":
            parsed = {
                "use_retrieval": True,
                "tasks": [{"route": "upload", "query": "read source", "k": 4}],
                "request_contract": WireRequestContract().model_dump(mode="json"),
            }
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
    monkeypatch.setattr("src.app.web.cleanup.get_project_root_path", lambda: tmp_path)
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
        add=[UploadAddition(path=str(source), name=source.name)],
    )).manifest
    agent = store.get_or_create("session-a")
    owned = Path(agent._ensure_session().upload_records[0].path)
    try:
        yield agent, service, before, source, owned
    finally:
        store.close_all()


@pytest.mark.parametrize("legacy_path_present", [False, True])
def test_direct_legacy_transition_retires_managed_uploads_and_rejects_old_context(
    managed_upload_agent, legacy_path_present,
):
    """Changing from managed attachments to a legacy call retires their source set and version together."""
    agent, service, before, source, owned = managed_upload_agent
    borrowed = source.with_name("borrowed.py")
    borrowed.write_text("borrowed_value = 2\n", encoding="utf-8")
    original = agent.run_agent_flow("read uploads", uploads=before.context())
    assert {item.evidence.snapshot.title for item in AnswerResponse.model_validate(original["response"]).citations} == {"source.py"}
    result = agent.run_agent_flow("read uploads", upload_file_path=str(borrowed) if legacy_path_present else None)
    after = service.get_manifest("session-a")
    assert after.files == []
    assert after.epoch == before.epoch and after.revision > before.revision
    assert not owned.exists() and source.is_file() and borrowed.is_file()
    response = AnswerResponse.model_validate(result["response"])
    assert {item.evidence.snapshot.title for item in response.citations} == ({"borrowed.py"} if legacy_path_present else set())

    stale = agent.run_agent_flow("read uploads", uploads=before.context())
    assert "UPLOAD_REVISION_CONFLICT" in export_answer_text(AnswerResponse.model_validate(stale["response"]))
    assert service.get_manifest("session-a") == after
    empty_context = agent.run_agent_flow("read uploads", uploads=after.context())
    assert AnswerResponse.model_validate(empty_context["response"]).citations == []
    if legacy_path_present:
        repeated = agent.run_agent_flow("read uploads", upload_file_path=str(borrowed))
        assert {item.evidence.snapshot.title for item in AnswerResponse.model_validate(repeated["response"]).citations} == {"borrowed.py"}
        borrowed.write_text("borrowed_value = 3\n", encoding="utf-8")
        replaced = agent.run_agent_flow("read uploads", upload_file_path=str(borrowed))
        assert "borrowed_value = 3" in export_answer_text(AnswerResponse.model_validate(replaced["response"]))
        cleared = agent.run_agent_flow("read uploads")
        assert AnswerResponse.model_validate(cleared["response"]).citations == []
        assert borrowed.is_file()


def test_direct_stale_context_cannot_reset_current_managed_uploads(managed_upload_agent):
    """A stale versioned reset preserves the current epoch, files and searchable source."""
    agent, service, before, _source, owned = managed_upload_agent
    stale_context = UploadContext(epoch=before.epoch, revision=before.revision - 1)

    reset = agent.run_agent_flow("exit", uploads=stale_context)

    assert "UPLOAD_REVISION_CONFLICT" in export_answer_text(AnswerResponse.model_validate(reset["response"]))
    assert service.get_manifest("session-a") == before
    assert owned.is_file()
    current = agent.run_agent_flow("read uploads", uploads=before.context())
    assert {item.evidence.snapshot.title for item in AnswerResponse.model_validate(current["response"]).citations} == {"source.py"}


if __name__ == "__main__":
    unittest.main()
