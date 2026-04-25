import json
import time
import unittest

from langchain_core.messages import AIMessage, ToolMessage

from src.core.contracts import GraphState, PlannerState, ResponseState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.runtime.make_graph import build_graph
from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
from src.runtime.nodes.session import add_user_message
from src.runtime.nodes.validation import make_validate_evidence_node
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.infra.tools.local_rag import build_local_rag_tools

from .helpers import _ToolWrapper, _tool_payload, build_legacy_state


class RetrievalNodeTest(unittest.TestCase):
    def test_upload_search_reranks_parameter_query_toward_usage_cell(self) -> None:
        settings = type("Settings", (), {"openai_api_key": "test-key"})()
        _rag_tool, upload_tool = build_local_rag_tools(settings)

        class _Doc:
            def __init__(self, text, cell_id):
                self.page_content = text
                self.metadata = {
                    "source": "uploads/demo/sample_pipeline.ipynb",
                    "cell_id": cell_id,
                    "chunk_id": 0,
                    "start_offset": 0,
                    "end_offset": len(text),
                }

        class _VectorStore:
            def similarity_search_with_score(self, query, k=4):
                _ = (query, k)
                return [
                    (_Doc("from sklearn.model_selection import train_test_split", 1), 0.30),
                    (_Doc("train_test_split(X, y, test_size=0.2, random_state=42)", 2), 0.28),
                ]

        retriever = type("Retriever", (), {"vectorstore": _VectorStore()})()
        payload = upload_tool.func(
            query="업로드 노트북에서 train_test_split 파라미터를 찾아줘",
            k=4,
            retriever=retriever,
        )

        evidence = payload["evidence"]
        self.assertEqual(evidence[0]["cell_id"], 2)
        self.assertIn("test_size=0.2", evidence[0]["snippet"])

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
            state_type=GraphState,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=lambda state: {"planner": PlannerState(output=planner_output)},
            retrieve_dispatch_node=retrieve_dispatch,
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "response": ResponseState(final_answer="final answer", synthesis_attempt=1),
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
            summary_max_turns=6,
        )

        result = graph.invoke(build_graph_state_input(user_input="question", messages=[]))
        retrieved = result["retrieval"].evidence_log
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
        self.assertEqual(result["debug"].retrieval_diagnostics[0].status, "success")
        self.assertEqual(result["debug"].retrieval_diagnostics[1].status, "success")

    def test_retrieve_dispatch_records_error_diagnostics(self) -> None:
        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(lambda query: (_ for _ in ()).throw(RuntimeError("boom"))),
            _ToolWrapper(
                lambda query, k, retriever=None: _tool_payload(
                    [],
                    tool="upload_search",
                    route="upload",
                    status="no_result",
                    message="",
                    query=query,
                )
            ),
            _ToolWrapper(
                lambda query, k: _tool_payload(
                    [],
                    tool="rag_search",
                    route="local",
                    status="no_result",
                    message="",
                    query=query,
                )
            ),
            verbose=False,
        )

        updates = retrieve_dispatch(
            build_legacy_state(
                {
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
                    ),
                    "retry_context": {"attempt": 0},
                }
            )
        )

        payload = json.loads(updates["messages"][0].content)
        self.assertEqual(payload["diagnostics"]["status"], "error")
        self.assertEqual(updates["debug"].retrieval_diagnostics[0].status, "error")

    def test_retrieve_dispatch_injects_benchmark_faults_without_calling_tool(self) -> None:
        docs_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            return _tool_payload([], tool="tavily_search", route="docs", status="success", message="", query=query)

        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(_docs_search),
            _ToolWrapper(lambda query, k, retriever=None: _tool_payload([], tool="upload_search", route="upload", status="no_result", message="", query=query)),
            _ToolWrapper(lambda query, k: _tool_payload([], tool="rag_search", route="local", status="no_result", message="", query=query)),
            verbose=False,
        )

        updates = retrieve_dispatch(
            build_legacy_state(
                {
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
                    ),
                    "retry_context": {"attempt": 0},
                    "eval_faults": {"tavily": "timeout"},
                }
            )
        )

        payload = json.loads(updates["messages"][0].content)
        self.assertEqual(docs_calls["count"], 0)
        self.assertEqual(payload["diagnostics"]["error_code"], "RETRIEVAL_DOCS_TIMEOUT")
        self.assertIn("RETRIEVAL_DOCS_TIMEOUT", updates["debug"].error_codes)

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
            _ToolWrapper(
                lambda query, k, retriever=None: _tool_payload(
                    [],
                    tool="upload_search",
                    route="upload",
                    status="no_result",
                    message="",
                    query=query,
                )
            ),
            _ToolWrapper(_local_search),
            verbose=False,
        )

        updates = retrieve_dispatch(
            build_legacy_state(
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
        )

        self.assertEqual(
            [item.route for item in updates["debug"].retrieval_diagnostics],
            ["docs", "local"],
        )
        route_events = [
            item
            for item in updates["debug"].latency_trace
            if item.get("kind") == "retrieval_route"
        ]
        self.assertEqual([item["route"] for item in route_events], ["docs", "local"])

    def test_retrieve_dispatch_reuses_preserved_upload_results_on_docs_retry(self) -> None:
        docs_calls = {"count": 0}
        upload_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            return _tool_payload(
                [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                        "url_or_path": "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                        "title": "train_test_split",
                        "snippet": "train_test_split splits arrays or matrices.",
                        "score": 0.9,
                    }
                ],
                tool="tavily_search",
                route="docs",
                status="success",
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
                message="should not run on retry",
                query=query,
            )

        retrieve_dispatch = make_retrieve_dispatch_node(
            _ToolWrapper(_docs_search),
            _ToolWrapper(_upload_search),
            _ToolWrapper(
                lambda query, k: _tool_payload(
                    [],
                    tool="rag_search",
                    route="local",
                    status="no_result",
                    message="",
                    query=query,
                )
            ),
            verbose=False,
        )

        upload_query = "uploaded notebook example"
        updates = retrieve_dispatch(
            build_legacy_state(
                {
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[
                            RetrievalTask(route="docs", query="train_test_split", k=3),
                            RetrievalTask(route="upload", query=upload_query, k=3),
                        ],
                    ),
                    "retry_context": {
                        "attempt": 1,
                        "failed_routes": ["docs"],
                        "preserved_evidence": [
                            {
                                "kind": "local",
                                "tool": "upload_search",
                                "source_id": "path:uploads/demo/sample.ipynb#cell=1;chunk=0;start=0;end=64",
                                "document_id": "path:uploads/demo/sample.ipynb",
                                "url_or_path": "uploads/demo/sample.ipynb",
                                "snippet": "X_train, X_test, y_train, y_test = train_test_split(...)",
                                "score": 0.0,
                                "cell_id": 1,
                                "chunk_id": 0,
                                "start_offset": 0,
                                "end_offset": 64,
                            }
                        ],
                        "preserved_retrieval_diagnostics": [
                            {
                                "tool": "upload_search",
                                "route": "upload",
                                "status": "success",
                                "message": "",
                                "query": upload_query,
                                "attempt": 1,
                            }
                        ],
                    },
                }
            )
        )

        self.assertEqual(docs_calls["count"], 1)
        self.assertEqual(upload_calls["count"], 0)
        self.assertEqual(
            [item["tool"] for item in updates["retrieval"].evidence_log],
            ["tavily_search", "upload_search"],
        )
        self.assertEqual(
            [item.route for item in updates["debug"].retrieval_diagnostics],
            ["docs", "upload"],
        )
