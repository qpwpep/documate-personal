import unittest

from langchain_core.messages import HumanMessage, SystemMessage

from src.core.answer_schema import AnswerSection, ClaimItem
from src.core.contracts.boundary.response import get_response_state
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis import make_synthesize_node

from .helpers import _CaptureStructuredSynthesizeLLM, build_legacy_state


class _BindableCaptureStructuredSynthesizeLLM(_CaptureStructuredSynthesizeLLM):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bound_max_tokens: list[int] = []

    def bind(self, **kwargs):
        max_tokens = kwargs.get("max_tokens")
        if max_tokens is not None:
            self.bound_max_tokens.append(int(max_tokens))
        return self


def _docs_evidence(snippet: str = "official docs evidence") -> dict:
    return {
        "kind": "official",
        "tool": "tavily_search",
        "source_id": "url:https://docs.example.com/api",
        "document_id": "url:https://docs.example.com/api",
        "url_or_path": "https://docs.example.com/api",
        "title": "Official Docs",
        "snippet": snippet,
        "score": 0.95,
    }


def _upload_evidence(snippet: str = "uploaded code evidence") -> dict:
    return {
        "kind": "local",
        "tool": "upload_search",
        "source_id": "path:uploads/demo.py#chunk=0;start=0;end=80",
        "document_id": "path:uploads/demo.py",
        "url_or_path": "uploads/demo.py",
        "title": "demo.py",
        "snippet": snippet,
        "score": 0.9,
    }


class SynthesisPromptBudgetTest(unittest.TestCase):
    def test_synthesize_node_truncates_prompt_evidence_for_small_token_budget(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(
            capture_llm,
            verbose=False,
            synthesis_max_tokens=100,
            prompt_snippet_char_limit=400,
        )
        long_snippet = "broadcast " * 300

        _ = synthesize_node(
            build_legacy_state(
                {
                    "messages": [HumanMessage(content="Explain numpy broadcasting.")],
                    "retrieved_evidence": [
                        {
                            "kind": "official",
                            "tool": "tavily_search",
                            "source_id": "url:https://numpy.org/doc/stable/",
                            "document_id": "url:https://numpy.org/doc/stable/",
                            "url_or_path": "https://numpy.org/doc/stable/",
                            "title": "NumPy Docs",
                            "snippet": long_snippet,
                            "score": 0.95,
                        }
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        evidence_messages = [
            str(message.content)
            for message in (capture_llm.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retrieved Evidence]" in str(message.content)
        ]
        self.assertEqual(len(evidence_messages), 1)
        self.assertIn("source_id: url:https://numpy.org/doc/stable/", evidence_messages[0])
        self.assertNotIn(long_snippet.strip(), evidence_messages[0])
        self.assertLess(len(evidence_messages[0]), 900)

    def test_synthesize_node_applies_prompt_budget_to_local_evidence_by_default(self) -> None:
        capture_llm = _CaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(
            capture_llm,
            verbose=False,
            synthesis_max_tokens=1920,
            prompt_snippet_char_limit=160,
        )
        long_snippet = "setup line\n" + ("local option detail " * 120) + "\naxis=0\n"

        _ = synthesize_node(
            build_legacy_state(
                {
                    "user_input": "Compare api evidence official docs with uploaded local option detail axis.",
                    "messages": [HumanMessage(content="Compare api evidence official docs with uploaded local option detail axis.")],
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[
                            RetrievalTask(route="docs", query="api evidence official docs", k=3),
                            RetrievalTask(route="upload", query="local option detail axis", k=3),
                        ],
                    ),
                    "retrieved_evidence": [
                        _docs_evidence("official docs evidence"),
                        _upload_evidence(long_snippet),
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        evidence_messages = [
            str(message.content)
            for message in (capture_llm.last_messages or [])
            if isinstance(message, SystemMessage) and "[Retrieved Evidence]" in str(message.content)
        ]
        self.assertEqual(len(evidence_messages), 1)
        self.assertIn("source_id: path:uploads/demo.py#chunk=0;start=0;end=80", evidence_messages[0])
        self.assertNotIn(long_snippet.strip(), evidence_messages[0])
        self.assertLess(len(evidence_messages[0]), 700)

    def test_synthesize_node_binds_category_specific_token_budget(self) -> None:
        capture_llm = _BindableCaptureStructuredSynthesizeLLM(include_raw=True)
        synthesize_node = make_synthesize_node(
            capture_llm,
            verbose=False,
            synthesis_max_tokens=1920,
            prompt_snippet_char_limit=400,
        )

        _ = synthesize_node(
            build_legacy_state(
                {
                    "user_input": "Explain official docs.",
                    "messages": [HumanMessage(content="Explain official docs.")],
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[RetrievalTask(route="docs", query="official docs", k=3)],
                    ),
                    "retrieved_evidence": [_docs_evidence()],
                    "synthesis_attempt": 0,
                }
            )
        )

        self.assertEqual(capture_llm.bound_max_tokens[0], 1024)

    def test_synthesize_node_clamps_hybrid_claims_and_section_bodies(self) -> None:
        claims = [
            ClaimItem(text=f"Claim {index}.", evidence_ids=["url:https://docs.example.com/api"])
            for index in range(1, 6)
        ]
        payload = {
            "answer": "",
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "sections": [
                AnswerSection(
                    kind="official_docs",
                    heading="Official",
                    body="One. Two. Three. Four.",
                ).model_dump(mode="json"),
                AnswerSection(
                    kind="comparison",
                    heading="Comparison",
                    body="A. B. C.",
                ).model_dump(mode="json"),
            ],
            "confidence": None,
        }
        synthesize_node = make_synthesize_node(
            _CaptureStructuredSynthesizeLLM(payload=payload, include_raw=True),
            verbose=False,
            synthesis_max_tokens=1920,
            prompt_snippet_char_limit=400,
        )

        result = synthesize_node(
            build_legacy_state(
                {
                    "user_input": "Compare official docs with uploaded code.",
                    "messages": [HumanMessage(content="Compare official docs with uploaded code.")],
                    "planner_output": PlannerOutput(
                        use_retrieval=True,
                        tasks=[
                            RetrievalTask(route="docs", query="api official docs", k=3),
                            RetrievalTask(route="upload", query="api uploaded code", k=3),
                        ],
                    ),
                    "retrieved_evidence": [
                        _docs_evidence("api official docs"),
                        _upload_evidence("api uploaded code"),
                    ],
                    "synthesis_attempt": 0,
                }
            )
        )

        response = get_response_state(result)
        self.assertEqual(len(response.payload.claims), 4)
        section_bodies = {section.kind: section.body for section in response.payload.sections}
        self.assertEqual(section_bodies["official_docs"], "One. Two. Three.")
        self.assertEqual(section_bodies["comparison"], "A. B.")


if __name__ == "__main__":
    unittest.main()
