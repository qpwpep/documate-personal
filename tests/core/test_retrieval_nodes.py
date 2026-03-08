import json
import time
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from src.make_graph import build_graph
from src.nodes.retrieval import make_retrieve_dispatch_node
from src.nodes.session import add_user_message
from src.nodes.state import State
from src.nodes.validation import make_validate_evidence_node
from src.planner_schema import PlannerOutput, RetrievalTask

from .helpers import _ToolWrapper, _tool_payload


class RetrievalNodeTest(unittest.TestCase):
    def test_retrieve_dispatch_merges_evidence_and_tool_messages(self) -> None:
        docs_evidence = [
            {
                "kind": "official",
                "tool": "tavily_search",
                "source_id": "url:https://numpy.org/doc/stable/",
                "url_or_path": "https://numpy.org/doc/stable/",
                "title": "NumPy Docs",
                "snippet": "official docs",
                "score": 0.99,
            }
        ]
        local_evidence = [
            {
                "kind": "local",
                "tool": "rag_search",
                "source_id": "path:data/notebooks/example.ipynb",
                "url_or_path": "data/notebooks/example.ipynb",
                "title": None,
                "snippet": "local snippet",
                "score": 0.88,
            }
        ]

        docs_calls = {"count": 0}
        upload_calls = {"count": 0}
        local_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            return _tool_payload(
                docs_evidence if query else [],
                tool="tavily_search",
                route="docs",
                status="success" if query else "no_result",
                message="",
                query=query,
            )

        def _upload_search(query: str, k: int, retriever=None):
            _ = (query, k, retriever)
            upload_calls["count"] += 1
            return _tool_payload(
                [],
                tool="upload_search",
                route="upload",
                status="no_result",
                message="no uploaded evidence found",
                query=query,
            )

        def _local_search(query: str, k: int):
            _ = k
            local_calls["count"] += 1
            return _tool_payload(
                local_evidence if query else [],
                tool="rag_search",
                route="local",
                status="success" if query else "no_result",
                message="",
                query=query,
            )

        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(_docs_search),
            _ToolWrapper(_upload_search),
            _ToolWrapper(_local_search),
            verbose=False,
        )

        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="numpy docs", k=3),
                RetrievalTask(route="local", query="numpy notebook", k=3),
            ],
        )

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=lambda state: {"planner_output": planner_output},
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "final_answer": "final answer",
                "synthesis_attempt": 1,
                "needs_retry": False,
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke({"user_input": "question", "messages": []})
        retrieved = result.get("retrieved_evidence", [])
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(docs_calls["count"], 1)
        self.assertEqual(local_calls["count"], 1)
        self.assertEqual(upload_calls["count"], 0)

        tool_messages = [
            message for message in result["messages"] if isinstance(message, ToolMessage) and getattr(message, "name", "")
        ]
        tool_names = {message.name for message in tool_messages}
        self.assertIn("tavily_search", tool_names)
        self.assertIn("rag_search", tool_names)
        self.assertNotIn("upload_search", tool_names)
        self.assertEqual(result["retrieval_diagnostics"][0]["status"], "success")
        self.assertEqual(result["retrieval_diagnostics"][1]["status"], "success")

    def test_retrieve_dispatch_records_error_diagnostics(self) -> None:
        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(lambda query: (_ for _ in ()).throw(RuntimeError("boom"))),
            _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
            _ToolWrapper(lambda query, k: _tool_payload([], tool="rag_search", route="local", status="no_result", message="", query=query)),
            verbose=False,
        )

        updates = retrieve_dispatch(
            {
                "planner_output": PlannerOutput(
                    use_retrieval=True,
                    tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
                ),
                "retry_context": {"attempt": 0},
            }
        )

        payload = json.loads(updates["messages"][0].content)
        self.assertEqual(payload["diagnostics"]["status"], "error")
        self.assertEqual(updates["retrieval_diagnostics"][0]["status"], "error")

    def test_retrieve_dispatch_preserves_planner_task_order_under_parallel_execution(self) -> None:
        def _docs_search(query: str):
            time.sleep(0.05)
            return _tool_payload(
                [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://docs.example.com/",
                        "url_or_path": "https://docs.example.com/",
                        "title": "Docs",
                        "snippet": "docs snippet",
                        "score": 0.8,
                    }
                ],
                tool="tavily_search",
                route="docs",
                status="success",
                message="",
                query=query,
            )

        def _local_search(query: str, k: int):
            _ = k
            return _tool_payload(
                [
                    {
                        "kind": "local",
                        "tool": "rag_search",
                        "source_id": "path:data/example.ipynb#chunk=0;start=0;end=10",
                        "url_or_path": "data/example.ipynb",
                        "title": None,
                        "snippet": "local snippet",
                        "score": 0.7,
                    }
                ],
                tool="rag_search",
                route="local",
                status="success",
                message="",
                query=query,
            )

        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(_docs_search),
            _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
            _ToolWrapper(_local_search),
            verbose=False,
        )

        updates = retrieve_dispatch(
            {
                "planner_output": PlannerOutput(
                    use_retrieval=True,
                    tasks=[
                        RetrievalTask(route="docs", query="docs query", k=3),
                        RetrievalTask(route="local", query="local query", k=3),
                    ],
                ),
                "retry_context": {"attempt": 0},
            }
        )

        self.assertEqual(
            [item["route"] for item in updates["retrieval_diagnostics"]],
            ["docs", "local"],
        )
        route_events = [
            item
            for item in updates["latency_trace"]
            if item.get("kind") == "retrieval_route"
        ]
        self.assertEqual([item["route"] for item in route_events], ["docs", "local"])
