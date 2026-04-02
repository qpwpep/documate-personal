import importlib
import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.answer_schema import (
    AgentResponsePayloadModel,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
)
from src.contracts import RetrievalDiagnostic, RetryState
from src.evidence import EvidenceItem
from src.nodes.planner.policy import build_deterministic_planner_decision
from src.nodes.retrieval import collect_retrieval_result
from src.nodes.retrieval.node import _collect_retrieval_batch, _execute_retrieval_batch
from src.nodes.synthesis.payload_builder import build_plain_summary_attach_payload
from src.nodes.synthesis.prompt_builder import build_synthesis_messages
from src.nodes.validation.evidence_validator import assess_validation, build_validation_snapshot
from src.nodes.validation.recovery import apply_validation_outcome
from src.planner_schema import PlannerOutput, RetrievalTask
from src.prompts import SYS_POLICY

from .helpers import build_legacy_state


def _docs_evidence(
    *,
    source_id: str = "url:https://numpy.org/doc/stable/",
    snippet: str = "Broadcasting expands compatible array shapes.",
    score: float = 0.9,
) -> dict:
    return {
        "kind": "official",
        "tool": "tavily_search",
        "source_id": source_id,
        "document_id": source_id,
        "url_or_path": source_id.removeprefix("url:"),
        "title": "NumPy Docs",
        "snippet": snippet,
        "score": score,
    }


def _upload_evidence(
    *,
    source_id: str = "path:uploads/demo/sample.ipynb#cell=1;chunk=0;start=0;end=64",
    snippet: str = "X_train, X_test = train_test_split(...)",
    score: float = 0.0,
) -> dict:
    return {
        "kind": "local",
        "tool": "upload_search",
        "source_id": source_id,
        "document_id": "path:uploads/demo/sample.ipynb",
        "url_or_path": "uploads/demo/sample.ipynb",
        "snippet": snippet,
        "score": score,
        "cell_id": 1,
        "chunk_id": 0,
        "start_offset": 0,
        "end_offset": 64,
    }


class NodeRefactorTest(unittest.TestCase):
    def test_planner_policy_facade_reexports_deterministic_builder(self) -> None:
        policy_module = importlib.import_module("src.nodes.planner.policy")
        deterministic_module = importlib.import_module("src.nodes.planner.deterministic")

        self.assertIs(
            policy_module.build_deterministic_planner_decision,
            deterministic_module.build_deterministic_planner_decision,
        )

    def test_synthesis_package_reexports_models_and_factory(self) -> None:
        synthesis_module = importlib.import_module("src.nodes.synthesis")
        models_module = importlib.import_module("src.nodes.synthesis.models")
        node_module = importlib.import_module("src.nodes.synthesis.node")

        self.assertIs(synthesis_module.PreparedSynthesisInputs, models_module.PreparedSynthesisInputs)
        self.assertIs(synthesis_module.SynthesisPipelineResult, models_module.SynthesisPipelineResult)
        self.assertIs(synthesis_module.make_synthesize_node, node_module.make_synthesize_node)

    def test_planner_policy_builds_deterministic_hybrid_decision(self) -> None:
        decision = build_deterministic_planner_decision(
            user_input="Explain pandas concat from official docs and compare it with the uploaded notebook example.",
            has_retriever=True,
        )

        self.assertIsNotNone(decision)
        self.assertEqual(decision.status, "deterministic")
        self.assertEqual([task.route for task in decision.output.tasks], ["docs", "upload"])
        self.assertEqual(decision.output.tasks[0].query, "pandas concat")
        self.assertEqual(decision.output.tasks[1].query, "uploaded notebook example")

    def test_retrieval_batch_reuses_preserved_results_and_keeps_task_order(self) -> None:
        docs_calls = {"count": 0}
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="uploaded notebook example", k=3),
            ],
        )
        retry_context = RetryState(
            attempt=1,
            failed_routes=["docs"],
            preserved_evidence=[_upload_evidence()],
            preserved_retrieval_diagnostics=[
                RetrievalDiagnostic(
                    tool="upload_search",
                    route="upload",
                    status="success",
                    message="",
                    query="uploaded notebook example",
                    attempt=1,
                )
            ],
        )
        route_handlers = {
            "docs": (
                "tavily_search",
                lambda task: {
                    "evidence": [
                        _docs_evidence(
                            source_id="url:https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html",
                            snippet="Split arrays or matrices into random train and test subsets.",
                        )
                    ],
                    "diagnostics": {
                        "tool": "tavily_search",
                        "route": "docs",
                        "status": "success",
                        "message": "",
                        "query": task.query,
                        "attempt": 2,
                    },
                }
            ),
            "upload": (
                "upload_search",
                lambda task: (_ for _ in ()).throw(AssertionError(f"unexpected upload call: {task.query}")),
            ),
        }

        batch_plan = _collect_retrieval_batch(
            planner_output=planner_output,
            retry_context=retry_context,
            route_handlers=route_handlers,
        )
        self.assertEqual(len(batch_plan.indexed_tasks), 1)
        self.assertEqual(len(batch_plan.reused_results), 1)

        batch_result = _execute_retrieval_batch(batch_plan)
        self.assertEqual(
            [item.route for item in batch_result.retrieval_diagnostics],
            ["docs", "upload"],
        )
        self.assertEqual(batch_result.tool_messages[0].name, "tavily_search")
        self.assertEqual(batch_result.tool_messages[1].name, "upload_search")
        self.assertEqual(len(batch_result.evidence_updates), 2)

    def test_synthesis_prompt_builder_preserves_fixed_messages_while_trimming_history(self) -> None:
        state = build_legacy_state(
            {
                "messages": [
                    HumanMessage(content="u1"),
                    AIMessage(content="a1"),
                    HumanMessage(content="u2"),
                    AIMessage(content="a2"),
                    HumanMessage(content="u3"),
                    AIMessage(content="a3"),
                    HumanMessage(content="u4"),
                ],
                "memory_summary": "older summary",
            }
        )
        model_messages, history_before, history_after = build_synthesis_messages(
            state=state,
            action_rules=["Produce the final answer content to save now."],
            deduped_evidence=[_docs_evidence()],
            attempt=2,
            max_turns=2,
        )

        self.assertGreater(history_before, history_after)
        self.assertIsInstance(model_messages[0], SystemMessage)
        self.assertEqual(model_messages[0].content, SYS_POLICY)
        self.assertTrue(any("[Conversation Summary]" in str(message.content) for message in model_messages))
        self.assertTrue(any("[Retrieved Evidence]" in str(message.content) for message in model_messages))
        self.assertTrue(any("Do not list raw links or search results" in str(message.content) for message in model_messages))
        self.assertTrue(
            any(
                "Retry after evidence validation failed" in str(message.content)
                for message in model_messages
            )
        )
        self.assertFalse(any(isinstance(message, HumanMessage) and message.content == "u1" for message in model_messages))
        self.assertTrue(any(isinstance(message, HumanMessage) and message.content == "u4" for message in model_messages))

    def test_synthesis_prompt_builder_adds_hybrid_guidance_when_official_and_local_evidence_exist(self) -> None:
        state = build_legacy_state({"messages": [HumanMessage(content="질문")]})
        model_messages, _, _ = build_synthesis_messages(
            state=state,
            action_rules=[],
            deduped_evidence=[_docs_evidence(), _upload_evidence()],
            attempt=1,
            max_turns=2,
        )

        self.assertTrue(
            any(
                "docs plus uploaded/local evidence" in str(message.content)
                for message in model_messages
            )
        )

    def test_synthesis_prompt_builder_truncates_long_evidence_snippets(self) -> None:
        state = build_legacy_state({"messages": [HumanMessage(content="Explain it.")]})
        long_snippet = "A" * 400

        model_messages, _, _ = build_synthesis_messages(
            state=state,
            action_rules=[],
            deduped_evidence=[_docs_evidence(snippet=long_snippet)],
            attempt=1,
            max_turns=2,
        )

        retrieved_evidence_messages = [
            str(message.content)
            for message in model_messages
            if isinstance(message, SystemMessage) and "[Retrieved Evidence]" in str(message.content)
        ]
        self.assertEqual(len(retrieved_evidence_messages), 1)
        self.assertIn("A" * 40, retrieved_evidence_messages[0])
        self.assertNotIn("A" * 320, retrieved_evidence_messages[0])
        self.assertIn("...", retrieved_evidence_messages[0])

    def test_synthesis_payload_builder_adopts_plain_summary_segments(self) -> None:
        evidence_items = [
            EvidenceItem.model_validate(_docs_evidence()),
            EvidenceItem.model_validate(
                _docs_evidence(
                    source_id="url:https://numpy.org/doc/stable/broadcasting-2",
                    snippet="Broadcasting keeps loops in C.",
                )
            ),
        ]
        payload = build_plain_summary_attach_payload(
            content="NumPy broadcasting expands compatible shapes.\nIt avoids Python-level loops.",
            evidence_items=evidence_items,
        )

        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload.claims[0].evidence_ids, ["url:https://numpy.org/doc/stable/"])
        self.assertEqual(
            payload.claims[1].evidence_ids,
            ["url:https://numpy.org/doc/stable/broadcasting-2"],
        )
        self.assertIn("[1]", payload.answer)
        self.assertIn("[2]", payload.answer)

    def test_validation_assessment_flags_only_docs_route_on_hybrid_low_score(self) -> None:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[
                RetrievalTask(route="docs", query="train_test_split official docs", k=3),
                RetrievalTask(route="upload", query="uploaded notebook example", k=3),
            ],
        )
        snapshot = build_validation_snapshot(
            user_input="Compare official docs with the uploaded notebook example.",
            planner_output=planner_output,
            parsed_evidence=[
                EvidenceItem.model_validate(
                    _docs_evidence(
                        source_id="url:https://huggingface.co/docs/bad",
                        snippet="unrelated content",
                        score=0.1,
                    )
                ),
                EvidenceItem.model_validate(_upload_evidence()),
            ],
            current_attempt_retrieval_errors=[],
            current_attempt_retrieval_diagnostics=[
                RetrievalDiagnostic(
                    tool="tavily_search",
                    route="docs",
                    status="success",
                    message="",
                    query="train_test_split official docs",
                    attempt=1,
                ),
                RetrievalDiagnostic(
                    tool="upload_search",
                    route="upload",
                    status="success",
                    message="",
                    query="uploaded notebook example",
                    attempt=1,
                ),
            ],
            response_payload=build_empty_response_payload(answer="draft"),
        )

        assessment = assess_validation(snapshot)
        self.assertEqual(assessment.retry_reason, "low_score")
        self.assertEqual(assessment.failed_routes, {"docs"})
        self.assertFalse(assessment.blocked_missing_upload)

    def test_validation_recovery_filters_unsupported_claims_to_grounded_subset(self) -> None:
        valid_source = "path:data/notebooks/example.ipynb#cell=0;chunk=0;start=0;end=12"
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route="local", query="example", k=3)],
        )
        response_payload = AgentResponsePayloadModel.model_validate(
            {
                "answer": "kept [1] dropped [2]",
                "claims": [
                    {"text": "kept", "evidence_ids": [valid_source], "confidence": 0.9},
                    {
                        "text": "dropped",
                        "evidence_ids": ["path:data/notebooks/example.ipynb#cell=0;chunk=99;start=0;end=12"],
                        "confidence": 0.1,
                    },
                ],
                "evidence": [],
                "confidence": 0.5,
            }
        )
        snapshot = build_validation_snapshot(
            user_input="Explain the local example.",
            planner_output=planner_output,
            parsed_evidence=[
                EvidenceItem.model_validate(
                    {
                        "kind": "local",
                        "tool": "rag_search",
                        "source_id": valid_source,
                        "document_id": "path:data/notebooks/example.ipynb",
                        "url_or_path": "data/notebooks/example.ipynb",
                        "snippet": "example snippet",
                        "score": 0.9,
                        "cell_id": 0,
                        "chunk_id": 0,
                        "start_offset": 0,
                        "end_offset": 12,
                    }
                )
            ],
            current_attempt_retrieval_errors=[],
            current_attempt_retrieval_diagnostics=[],
            response_payload=response_payload,
        )

        assessment = assess_validation(snapshot)
        self.assertEqual(assessment.retry_reason, "unsupported_claims")

        updates = apply_validation_outcome(
            snapshot=snapshot,
            assessment=assessment,
            attempt=1,
            needs_retry=False,
        )
        self.assertEqual(updates["response"].final_answer, "kept [1]")
        self.assertEqual(len(updates["response"].payload.claims), 1)
        self.assertEqual(updates["response"].payload.claims[0].evidence_ids, [valid_source])

    def test_deterministic_grounded_payload_strips_markdown_and_navigation_noise(self) -> None:
        payload = build_deterministic_grounded_payload(
            evidence_items=[
                EvidenceItem.model_validate(
                    _docs_evidence(
                        snippet=(
                            "# Table of Contents\n"
                            "Home > Docs > API > Broadcasting\n"
                            "[NumPy broadcasting](https://numpy.org/doc/stable/)\n"
                            "Previous: Installation\n"
                            "Next: Indexing\n"
                            "Broadcasting expands compatible array shapes."
                        )
                    )
                )
            ]
        )

        self.assertEqual(
            payload.claims[0].text,
            "Broadcasting expands compatible array shapes.",
        )
        self.assertNotIn("Table of Contents", payload.answer)
        self.assertNotIn("Home >", payload.answer)
        self.assertNotIn("Previous", payload.answer)
        self.assertNotIn("Next", payload.answer)

    def test_deterministic_grounded_payload_drops_pipe_navigation_lines(self) -> None:
        payload = build_deterministic_grounded_payload(
            evidence_items=[
                EvidenceItem.model_validate(
                    _docs_evidence(
                        snippet=(
                            "[API Reference](https://numpy.org/doc/stable/) | "
                            "[User Guide](https://numpy.org/doc/stable/user/) | Previous | Next\n"
                            "`numpy.broadcast_to` repeats an array across a new shape."
                        )
                    )
                )
            ]
        )

        self.assertEqual(payload.claims[0].text, "numpy.broadcast_to repeats an array across a new shape.")
        self.assertNotIn("API Reference", payload.answer)
        self.assertNotIn("Previous", payload.answer)

    def test_collect_retrieval_result_filters_cross_library_docs_domains(self) -> None:
        payload_dicts, diagnostic = collect_retrieval_result(
            raw_payload={
                "evidence": [
                    _docs_evidence(
                        source_id="url:https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
                        snippet="Join a sequence of arrays.",
                    ),
                    _docs_evidence(
                        source_id="url:https://pandas.pydata.org/docs/reference/api/pandas.concat.html",
                        snippet="Concatenate pandas objects.",
                    ),
                ],
                "diagnostics": {
                    "tool": "tavily_search",
                    "route": "docs",
                    "status": "success",
                    "message": "",
                    "query": "numpy official docs",
                    "attempt": 1,
                },
            },
            tool_name="tavily_search",
            route="docs",
            query="numpy official docs",
            attempt=1,
            local_errors=[],
        )

        self.assertEqual(len(payload_dicts), 1)
        self.assertEqual(
            payload_dicts[0]["url_or_path"],
            "https://numpy.org/doc/stable/reference/generated/numpy.concatenate.html",
        )
        self.assertIn("cross_library_domain_filtered", diagnostic.warnings)


if __name__ == "__main__":
    unittest.main()
