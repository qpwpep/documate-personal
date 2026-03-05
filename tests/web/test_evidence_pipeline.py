import json
import unittest

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src.agent_manager import AgentFlowManager
from src.settings import AppSettings
from src.tools import build_tool_registry


class _FakeGraph:
    def __init__(self, evidence_payload: list[dict]):
        self._evidence_payload = evidence_payload

    def invoke(self, state: dict) -> dict:
        return {
            "messages": [
                HumanMessage(content=state["user_input"]),
                ToolMessage(
                    content=json.dumps(self._evidence_payload, ensure_ascii=False),
                    name="tavily_search",
                    tool_call_id="call-1",
                ),
                AIMessage(content="final answer"),
            ]
        }


class _FakeGraphWithSave:
    def invoke(self, state: dict) -> dict:
        _ = state
        return {
            "messages": [
                HumanMessage(content="question"),
                AIMessage(content="final answer before save"),
                ToolMessage(
                    content=json.dumps(
                        {
                            "message": "Saved output to response_20260101_010101.txt",
                            "file_path": "output/save_text/response_20260101_010101.txt",
                        },
                        ensure_ascii=False,
                    ),
                    name="save_text",
                    tool_call_id="save-1",
                ),
            ]
        }


class _FakeVectorStore:
    def similarity_search_with_relevance_scores(self, query: str, k: int = 4):
        _ = (query, k)
        return [
            (
                Document(
                    page_content="uploaded snippet",
                    metadata={"source": "uploads/session/sample_pipeline.ipynb"},
                ),
                0.87,
            )
        ]


class _FakeRetriever:
    def __init__(self):
        self.vectorstore = _FakeVectorStore()


class EvidencePipelineTest(unittest.TestCase):
    def test_extract_observed_evidence_uses_tool_native_payloads(self) -> None:
        tavily_item = {
            "kind": "official",
            "tool": "tavily_search",
            "source_id": "url:https://numpy.org/doc/stable/",
            "url_or_path": "https://numpy.org/doc/stable/",
            "title": "NumPy docs",
            "snippet": "broadcasting",
            "score": 0.98,
        }
        rag_item = {
            "kind": "local",
            "tool": "rag_search",
            "source_id": "path:uploads/s1/sample.ipynb",
            "url_or_path": "uploads/s1/sample.ipynb",
            "title": None,
            "snippet": "local snippet",
            "score": 0.71,
        }

        messages = [
            ToolMessage(content=json.dumps([tavily_item, tavily_item]), name="tavily_search", tool_call_id="1"),
            ToolMessage(content=json.dumps([rag_item]), name="rag_search", tool_call_id="2"),
            ToolMessage(content="not-json", name="upload_search", tool_call_id="3"),
            ToolMessage(content=json.dumps([rag_item]), name="save_text", tool_call_id="4"),
        ]
        errors: list[str] = []

        observed = AgentFlowManager._extract_observed_evidence(messages, errors=errors)

        self.assertEqual(len(observed), 2)
        self.assertTrue(any(item["tool"] == "tavily_search" for item in observed))
        self.assertTrue(any(item["tool"] == "rag_search" for item in observed))
        self.assertTrue(any("tool:upload_search" in error for error in errors))

    def test_response_and_debug_share_same_evidence_source(self) -> None:
        evidence_payload = [
            {
                "kind": "official",
                "tool": "tavily_search",
                "source_id": "url:https://numpy.org/doc/stable/",
                "url_or_path": "https://numpy.org/doc/stable/",
                "title": "NumPy docs",
                "snippet": "broadcasting",
                "score": 0.99,
            }
        ]

        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = None
        manager.graph = _FakeGraph(evidence_payload)
        manager.messages = []
        manager.retriever = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("question")

        self.assertIs(result["response_payload"]["evidence"], result["debug"]["observed_evidence"])
        self.assertEqual(result["response_payload"]["evidence"][0]["url_or_path"], "https://numpy.org/doc/stable/")

    def test_upload_search_returns_typed_evidence_and_handles_missing_retriever(self) -> None:
        registry = build_tool_registry(AppSettings(openai_api_key="test", tavily_api_key="test"))

        no_retriever = registry.upload_search_tool.func(query="uploaded info", k=3, retriever=None)
        self.assertEqual(no_retriever, [])

        with_retriever = registry.upload_search_tool.func(
            query="uploaded info",
            k=3,
            retriever=_FakeRetriever(),
        )
        self.assertEqual(len(with_retriever), 1)
        self.assertEqual(with_retriever[0]["kind"], "local")
        self.assertEqual(with_retriever[0]["tool"], "upload_search")
        self.assertEqual(with_retriever[0]["url_or_path"], "uploads/session/sample_pipeline.ipynb")
        self.assertEqual(with_retriever[0]["source_id"], "path:uploads/session/sample_pipeline.ipynb")
        self.assertAlmostEqual(with_retriever[0]["score"], 0.87)

    def test_save_tool_message_does_not_override_final_answer(self) -> None:
        manager = AgentFlowManager.__new__(AgentFlowManager)
        manager.settings = None
        manager.graph = _FakeGraphWithSave()
        manager.messages = []
        manager.retriever = None
        manager.upload_file_path = None

        result = manager.run_agent_flow("save this")

        self.assertEqual(result["message"], "final answer before save")
        self.assertEqual(result["response_payload"]["answer"], "final answer before save")


if __name__ == "__main__":
    unittest.main()
