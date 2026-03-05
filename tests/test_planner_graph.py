import json
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from pydantic import ValidationError

from src.make_graph import build_graph
from src.node import (
    State,
    add_user_message,
    make_action_postprocess_node,
    make_planner_node,
    make_retrieve_docs_node,
    make_retrieve_local_node,
    make_retrieve_upload_node,
    make_synthesize_node,
    make_validate_evidence_node,
)
from src.planner_schema import PlannerOutput, RetrievalTask
from src.prompts import SYS_POLICY


class _ToolWrapper:
    def __init__(self, func):
        self.func = func


class _FailingPlannerLLM:
    def invoke(self, _messages):
        raise RuntimeError("planner exploded")


class _InvalidPlannerLLM:
    def invoke(self, _messages):
        return {
            "use_retrieval": False,
            "tasks": [
                {"route": "docs", "query": "numpy", "k": 4},
            ],
        }


class _CaptureSynthesizeLLM:
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content="synth result")


class _CapturePlannerLLM:
    def __init__(self, planner_output):
        self.planner_output = planner_output
        self.last_messages = None
        self.call_count = 0

    def invoke(self, messages):
        self.last_messages = messages
        self.call_count += 1
        return self.planner_output


class PlannerGraphTest(unittest.TestCase):
    def test_planner_schema_rules(self) -> None:
        self.assertEqual(PlannerOutput(use_retrieval=False, tasks=[]).tasks, [])

        with self.assertRaises(ValidationError):
            PlannerOutput(
                use_retrieval=False,
                tasks=[RetrievalTask(route="docs", query="numpy", k=4)],
            )

        with self.assertRaises(ValidationError):
            PlannerOutput(use_retrieval=True, tasks=[])

        with self.assertRaises(ValidationError):
            PlannerOutput(
                use_retrieval=True,
                tasks=[
                    RetrievalTask(route="docs", query="numpy", k=4),
                    RetrievalTask(route="docs", query="python", k=4),
                ],
            )

    def test_planner_node_falls_back_when_invoke_fails(self) -> None:
        planner_node = make_planner_node(_FailingPlannerLLM(), verbose=False)
        updates = planner_node({"messages": [HumanMessage(content="hi")], "user_input": "hi"})
        planner_output = updates["planner_output"]
        self.assertFalse(planner_output.use_retrieval)
        self.assertEqual(planner_output.tasks, [])
        self.assertTrue(any("planner" in error for error in updates.get("retrieval_errors", [])))

    def test_planner_node_falls_back_when_schema_invalid(self) -> None:
        planner_node = make_planner_node(_InvalidPlannerLLM(), verbose=False)
        updates = planner_node({"messages": [HumanMessage(content="hi")], "user_input": "hi"})
        planner_output = updates["planner_output"]
        self.assertFalse(planner_output.use_retrieval)
        self.assertEqual(planner_output.tasks, [])
        self.assertTrue(any("validation failed" in error for error in updates.get("retrieval_errors", [])))

    def test_parallel_retrieval_fan_in_merges_evidence_and_tool_messages(self) -> None:
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

        retrieve_docs = make_retrieve_docs_node(
            _ToolWrapper(lambda query: docs_evidence if query else []),
            verbose=False,
        )
        retrieve_upload = make_retrieve_upload_node(
            _ToolWrapper(lambda query, k, retriever=None: []),
            verbose=False,
        )
        retrieve_local = make_retrieve_local_node(
            _ToolWrapper(lambda query, k: local_evidence if query else []),
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
            retrieve_docs_node=retrieve_docs,
            retrieve_upload_node=retrieve_upload,
            retrieve_local_node=retrieve_local,
            synthesize_node=lambda state: {
                "messages": [AIMessage(content="final answer")],
                "final_answer": "final answer",
                "synthesis_attempt": 1,
                "needs_retry": False,
            },
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
        )

        result = graph.invoke({"user_input": "question", "messages": []})
        retrieved = result.get("retrieved_evidence", [])
        self.assertEqual(len(retrieved), 2)

        tool_messages = [
            m for m in result["messages"] if isinstance(m, ToolMessage) and getattr(m, "name", "")
        ]
        tool_names = {m.name for m in tool_messages}
        self.assertIn("tavily_search", tool_names)
        self.assertIn("rag_search", tool_names)
        self.assertNotIn("upload_search", tool_names)

    def test_retry_path_replans_and_reruns_retrieval(self) -> None:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        capture_planner = _CapturePlannerLLM(planner_output)
        planner_node = make_planner_node(capture_planner, verbose=False)

        docs_calls = {"count": 0}
        synth_calls = {"count": 0}

        def _docs_search(query: str):
            docs_calls["count"] += 1
            if docs_calls["count"] == 1:
                return []
            return [
                {
                    "kind": "official",
                    "tool": "tavily_search",
                    "source_id": "url:https://numpy.org/doc/stable/",
                    "url_or_path": "https://numpy.org/doc/stable/",
                    "title": "NumPy Docs",
                    "snippet": "official docs",
                    "score": 0.92,
                }
            ]

        retrieve_docs = make_retrieve_docs_node(_ToolWrapper(_docs_search), verbose=False)
        retrieve_upload = make_retrieve_upload_node(
            _ToolWrapper(lambda query, k, retriever=None: []),
            verbose=False,
        )
        retrieve_local = make_retrieve_local_node(
            _ToolWrapper(lambda query, k: []),
            verbose=False,
        )

        def _synthesize(state):
            synth_calls["count"] += 1
            answer = f"answer-{synth_calls['count']}"
            return {
                "messages": [AIMessage(content=answer)],
                "final_answer": answer,
                "synthesis_attempt": int(state.get("synthesis_attempt", 0)) + 1,
                "needs_retry": False,
            }

        graph = build_graph(
            state_type=State,
            add_user_node=add_user_message,
            summarize_node=lambda state: state,
            planner_node=planner_node,
            retrieve_docs_node=retrieve_docs,
            retrieve_upload_node=retrieve_upload,
            retrieve_local_node=retrieve_local,
            synthesize_node=_synthesize,
            validate_evidence_node=make_validate_evidence_node(verbose=False),
            action_postprocess_node=lambda state: {},
        )

        result = graph.invoke({"user_input": "question", "messages": []})
        self.assertEqual(capture_planner.call_count, 2)
        self.assertEqual(docs_calls["count"], 2)
        self.assertEqual(synth_calls["count"], 2)
        self.assertEqual(result.get("final_answer"), "answer-2")

    def test_validate_evidence_retries_once_then_returns_uncertainty(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )

        first = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        second = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retry_context": first["retry_context"],
            }
        )
        self.assertTrue(first["needs_retry"])
        self.assertEqual(first["retry_context"]["attempt"], 1)
        self.assertEqual(first["retry_context"]["retry_reason"], "no_evidence")
        self.assertFalse(second["needs_retry"])
        self.assertIn("final_answer", second)
        self.assertIn("retry_reason: no_evidence", second["final_answer"])

    def test_validate_evidence_sets_tool_error_reason(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "retrieval_errors": ["tavily_search: failed (timeout)"],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "tool_error")

    def test_validate_evidence_sets_low_score_reason(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://numpy.org/doc/stable/",
                        "url_or_path": "https://numpy.org/doc/stable/",
                        "title": "NumPy Docs",
                        "snippet": "official docs",
                        "score": 0.2,
                    }
                ],
                "retry_context": {
                    "attempt": 0,
                    "max_retries": 1,
                    "evidence_start_index": 0,
                    "retrieval_error_start_index": 0,
                },
            }
        )
        self.assertTrue(result["needs_retry"])
        self.assertEqual(result["retry_context"]["retry_reason"], "low_score")
        self.assertAlmostEqual(result["retry_context"]["score_avg"], 0.2)

    def test_validate_evidence_passes_when_retrieval_not_required(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(use_retrieval=False, tasks=[])
        result = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "synthesis_attempt": 1,
            }
        )
        self.assertFalse(result["needs_retry"])

    def test_synthesize_node_keeps_sys_policy_persona(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=6)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "retrieved_evidence": [],
                "synthesis_attempt": 0,
            }
        )
        self.assertEqual(updates["final_answer"], "synth result")
        self.assertIsNotNone(capture_llm.last_messages)
        self.assertIsInstance(capture_llm.last_messages[0], SystemMessage)
        self.assertEqual(capture_llm.last_messages[0].content, SYS_POLICY)

    def test_synthesize_uses_only_current_attempt_evidence_window(self) -> None:
        capture_llm = _CaptureSynthesizeLLM()
        synthesize_node = make_synthesize_node(capture_llm, verbose=False, max_turns=8)

        updates = synthesize_node(
            {
                "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                "retrieved_evidence": [
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://old.example.com/",
                        "url_or_path": "https://old.example.com/",
                        "title": "Old Evidence",
                        "snippet": "old snippet",
                        "score": 0.8,
                    },
                    {
                        "kind": "official",
                        "tool": "tavily_search",
                        "source_id": "url:https://new.example.com/",
                        "url_or_path": "https://new.example.com/",
                        "title": "New Evidence",
                        "snippet": "new snippet",
                        "score": 0.9,
                    },
                ],
                "retry_context": {"evidence_start_index": 1},
                "synthesis_attempt": 0,
            }
        )

        self.assertEqual(updates["final_answer"], "synth result")
        retrieved_evidence_messages = [
            m.content
            for m in (capture_llm.last_messages or [])
            if isinstance(m, SystemMessage) and "[Retrieved Evidence]" in str(m.content)
        ]
        self.assertEqual(len(retrieved_evidence_messages), 1)
        self.assertIn("https://new.example.com/", retrieved_evidence_messages[0])
        self.assertNotIn("https://old.example.com/", retrieved_evidence_messages[0])

    def test_planner_includes_retry_context_system_message_on_retry(self) -> None:
        capture_planner = _CapturePlannerLLM(PlannerOutput(use_retrieval=False, tasks=[]))
        planner_node = make_planner_node(capture_planner, verbose=False)

        _ = planner_node(
            {
                "messages": [HumanMessage(content="hi")],
                "user_input": "hi",
                "planner_output": PlannerOutput(
                    use_retrieval=True,
                    tasks=[RetrievalTask(route="docs", query="numpy", k=3)],
                ),
                "retry_context": {
                    "attempt": 1,
                    "max_retries": 1,
                    "retry_reason": "no_evidence",
                    "retrieval_feedback": "query too narrow",
                },
            }
        )

        retry_context_messages = [
            m.content
            for m in (capture_planner.last_messages or [])
            if isinstance(m, SystemMessage) and "[Retry Context]" in str(m.content)
        ]
        self.assertEqual(len(retry_context_messages), 1)
        self.assertIn("reason=no_evidence", retry_context_messages[0])
        self.assertIn("previous_tasks=docs:numpy(k=3)", retry_context_messages[0])

    def test_action_postprocess_save_adds_tool_message_without_touching_answer(self) -> None:
        def _save_fn(content: str, filename_prefix: str = "response"):
            _ = (content, filename_prefix)
            return {"message": "Saved output to response_20260101_010101.txt", "file_path": "output/save.txt"}

        action_node = make_action_postprocess_node(
            save_text_tool=_ToolWrapper(_save_fn),
            slack_notify_tool=_ToolWrapper(lambda **kwargs: {"status": "ok"}),
            verbose=False,
        )

        updates = action_node(
            {
                "user_input": "save this answer to txt",
                "final_answer": "final answer text",
                "messages": [],
            }
        )

        self.assertNotIn("final_answer", updates)
        tool_messages = updates.get("messages", [])
        self.assertEqual(len(tool_messages), 1)
        self.assertEqual(tool_messages[0].name, "save_text")
        payload = json.loads(tool_messages[0].content)
        self.assertIn("file_path", payload)


if __name__ == "__main__":
    unittest.main()
