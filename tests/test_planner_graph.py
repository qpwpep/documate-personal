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

    def test_validate_evidence_retries_once_when_required_evidence_missing(self) -> None:
        validate_node = make_validate_evidence_node(verbose=False)
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="docs", query="numpy docs", k=3)],
        )

        first = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "synthesis_attempt": 1,
            }
        )
        second = validate_node(
            {
                "planner_output": planner_output,
                "retrieved_evidence": [],
                "synthesis_attempt": 2,
            }
        )
        self.assertTrue(first["needs_retry"])
        self.assertFalse(second["needs_retry"])

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
