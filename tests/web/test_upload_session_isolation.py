from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

from src.agent_manager import AgentFlowManager
from src.contracts import ResponseState
from src.settings import AppSettings
from src.tools.local_rag import build_temp_retriever
from src.web.routes import build_session_metadata_snapshot
from src.web.session_store import InMemorySessionStore, SessionEntry
from src.web.schemas import AgentRequest


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
            "response": ResponseState(final_answer="ok", payload={"answer": "ok", "claims": [], "evidence": [], "confidence": None}),
        }


class _ExplodingGraph:
    def invoke(self, _state: dict) -> dict:
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
                "response": ResponseState(
                    final_answer="ok",
                    payload={"answer": "ok", "claims": [], "evidence": [], "confidence": None},
                ),
            }
        finally:
            with self._lock:
                self._current -= 1


class _FakeHandle:
    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.retriever = object()
        self.cleanup_calls = 0

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
    @patch("src.tools.local_rag.build_openai_embeddings", return_value=_FakeEmbeddings())
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

    @patch("src.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_previous_handle_when_upload_changes(
        self,
        mock_build_temp_retriever,
    ) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)
        handle_one = _FakeHandle("upload-session-session")
        handle_two = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.side_effect = [handle_one, handle_two]

        manager.run_agent_flow("first", upload_file_path="uploads/session/file_one.py")
        manager.run_agent_flow("second", upload_file_path="uploads/session/file_two.py")

        self.assertEqual(handle_one.cleanup_calls, 1)
        self.assertIs(manager.upload_retriever_handle, handle_two)
        self.assertIs(graph.states[-1]["runtime"].retriever, handle_two.retriever)

    @patch("src.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_handle_when_upload_removed(self, mock_build_temp_retriever) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        manager.run_agent_flow("with upload", upload_file_path="uploads/session/file.py")
        manager.run_agent_flow("without upload")

        self.assertEqual(handle.cleanup_calls, 1)
        self.assertIsNone(manager.upload_retriever_handle)
        self.assertIsNone(graph.states[-1]["runtime"].retriever)

    @patch("src.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_handle_on_exit(self, mock_build_temp_retriever) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        manager.run_agent_flow("with upload", upload_file_path="uploads/session/file.py")
        manager.run_agent_flow("exit")

        self.assertEqual(handle.cleanup_calls, 1)
        self.assertIsNone(manager.upload_retriever_handle)
        self.assertEqual(manager.messages, [])

    @patch("src.agent_manager.build_temp_retriever")
    def test_agent_manager_cleans_handle_on_exception(self, mock_build_temp_retriever) -> None:
        manager = _make_manager(_ExplodingGraph())
        handle = _FakeHandle("upload-session-session")
        mock_build_temp_retriever.return_value = handle

        result = manager.run_agent_flow("with upload", upload_file_path="uploads/session/file.py")

        self.assertEqual(result["message"], "boom")
        self.assertEqual(handle.cleanup_calls, 1)
        self.assertIsNone(manager.upload_retriever_handle)

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
            created_monotonic=0.0,
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
                created_monotonic=0.0,
            )
            store.active_agents["fresh"] = SessionEntry(
                agent=fresh_agent,
                last_accessed_monotonic=95.0,
                created_monotonic=0.0,
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
                created_monotonic=0.0,
            )
            store.active_agents["newest"] = SessionEntry(
                agent=newest_agent,
                last_accessed_monotonic=2.0,
                created_monotonic=0.0,
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
                created_monotonic=0.0,
            )
            store.active_agents["unlocked"] = SessionEntry(
                agent=unlocked_agent,
                last_accessed_monotonic=2.0,
                created_monotonic=0.0,
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
                created_monotonic=0.0,
                active_request_count=1,
            )
            store.active_agents["newest"] = SessionEntry(
                agent=newest_agent,
                last_accessed_monotonic=2.0,
                created_monotonic=0.0,
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
            _agent, agent_answer, session_lock_wait_ms = store.run_session_request(
                session_id="demo-session",
                session_metadata=build_session_metadata_snapshot(
                    AgentRequest(query=f"question-{index}", session_id="demo-session")
                ),
                user_input=f"question-{index}",
                upload_file_path=None,
            )
            results.append((session_lock_wait_ms, str(agent_answer.get("message") or "")))

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


if __name__ == "__main__":
    unittest.main()
