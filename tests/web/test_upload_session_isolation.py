from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, HumanMessage

from src.agent_manager import AgentFlowManager
from src.settings import AppSettings
from src.upload_helpers import build_temp_retriever
from src.web import main as web_main
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
        return {
            "messages": [
                HumanMessage(content=state["user_input"]),
                AIMessage(content="ok"),
            ]
        }


class _ExplodingGraph:
    def invoke(self, _state: dict) -> dict:
        raise RuntimeError("boom")


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
    @patch("src.upload_helpers.OpenAIEmbeddings", return_value=_FakeEmbeddings())
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
        self.assertIs(graph.states[-1]["retriever"], handle_two.retriever)

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
        self.assertNotIn("retriever", graph.states[-1])

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
            graph.states[-1]["session_metadata"]["slack_destination"]["channel_id"],
            "C123BENCH",
        )

        manager.close()

        self.assertEqual(manager.session_metadata, {"slack_destination": None})
        self.assertEqual(manager.messages, [])

    def test_session_metadata_snapshot_replaces_previous_slack_destination(self) -> None:
        graph = _CapturingGraph()
        manager = _make_manager(graph)

        manager.set_session_metadata(
            web_main._build_session_metadata_snapshot(
                AgentRequest(
                    query="share this to slack",
                    session_id="demo-session",
                    slack_channel_id="C123BENCH",
                )
            )
        )
        manager.run_agent_flow("share this to slack")

        self.assertEqual(
            graph.states[-1]["session_metadata"]["slack_destination"]["channel_id"],
            "C123BENCH",
        )

        manager.set_session_metadata(
            web_main._build_session_metadata_snapshot(
                AgentRequest(
                    query="share this to slack",
                    session_id="demo-session",
                )
            )
        )
        manager.run_agent_flow("share this to slack")

        self.assertIsNone(graph.states[-1]["session_metadata"]["slack_destination"])
        self.assertFalse(any(message.__class__.__name__ == "SystemMessage" for message in manager.messages))


class SessionCacheCleanupTest(unittest.TestCase):
    def setUp(self) -> None:
        self._original_active_agents = dict(web_main.active_agents)
        web_main.active_agents.clear()

    def tearDown(self) -> None:
        web_main.active_agents.clear()
        web_main.active_agents.update(self._original_active_agents)

    def test_cleanup_expired_sessions_calls_agent_close(self) -> None:
        stale_agent = _make_manager(_CapturingGraph())
        fresh_agent = _make_manager(_CapturingGraph())
        with patch.object(stale_agent, "close") as stale_close, patch.object(
            fresh_agent, "close"
        ) as fresh_close:
            web_main.active_agents["stale"] = web_main.SessionEntry(
                agent=stale_agent,
                last_accessed_monotonic=0.0,
                created_monotonic=0.0,
            )
            web_main.active_agents["fresh"] = web_main.SessionEntry(
                agent=fresh_agent,
                last_accessed_monotonic=95.0,
                created_monotonic=0.0,
            )

            removed = web_main._cleanup_expired_sessions(now=100.0, ttl_seconds=10)

            self.assertEqual(removed, 1)
            stale_close.assert_called_once_with()
            fresh_close.assert_not_called()
            self.assertNotIn("stale", web_main.active_agents)
            self.assertIn("fresh", web_main.active_agents)

    def test_lru_eviction_calls_agent_close(self) -> None:
        oldest_agent = _make_manager(_CapturingGraph())
        newest_agent = _make_manager(_CapturingGraph())
        with patch.object(oldest_agent, "close") as oldest_close, patch.object(
            newest_agent, "close"
        ) as newest_close:
            web_main.active_agents["oldest"] = web_main.SessionEntry(
                agent=oldest_agent,
                last_accessed_monotonic=1.0,
                created_monotonic=0.0,
            )
            web_main.active_agents["newest"] = web_main.SessionEntry(
                agent=newest_agent,
                last_accessed_monotonic=2.0,
                created_monotonic=0.0,
            )

            evicted = web_main._evict_lru_if_needed(max_active_sessions=1)

            self.assertEqual(evicted, 1)
            oldest_close.assert_called_once_with()
            newest_close.assert_not_called()
            self.assertNotIn("oldest", web_main.active_agents)
            self.assertIn("newest", web_main.active_agents)


if __name__ == "__main__":
    unittest.main()
