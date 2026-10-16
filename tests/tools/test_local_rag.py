import hashlib
import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from langchain_core.embeddings import Embeddings

from src.core.planner_schema import RetrievalRequirement
from src.core.evidence import EvidenceRef
from src.infra.chunking import chunk_notebook_path, chunk_python_text
from src.core.uploads import UploadRecord
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool
from src.infra.tools.local_rag.serialization import build_local_hit_bundle
from tests.synthesis_fixtures import synthesis_excerpt_limits


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count("train_test_split"))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("train_test_split"))]


def _index_upload(path: Path):
    data = path.read_bytes()
    file_id = uuid4().hex
    record = UploadRecord(
        file_id=file_id, name=path.name, path=str(path), size_bytes=len(data),
        content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
        source_uri=f"upload:///rag-tests/{file_id}",
    )
    return build_upload_retriever([record], session_id="rag-tests", generation=uuid4().hex, api_key="test-key")


def _write_notebook(path: Path, *sources: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cells = []
    for index, source in enumerate(sources):
        cells.append(
            {
                "cell_type": "markdown" if index == 0 else "code",
                "metadata": {},
                "source": [source],
                **({"execution_count": None, "outputs": []} if index > 0 else {}),
            }
        )
    path.write_text(
        json.dumps(
            {
                "cells": cells,
                "metadata": {},
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


class LocalRagTest(unittest.TestCase):




    @patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings())
    def test_long_line_search_excerpt_survives_synthesis_and_exact_citation(self, _embeddings) -> None:
        """A retrieved fragment of a long source line remains model-visible and exactly citable."""
        import time

        from src.core.answer_schema import AnswerDocument, CodeBlock, ContentUnit, UnitCheck
        from src.core.contracts import PlannerState
        from src.core.contracts.boundary.graph import build_graph_state_input
        from src.core.planner_schema import PlannerOutput, RetrievalTask
        from src.core.request_contracts import RequestContract
        from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
        from src.runtime.nodes.synthesis.budgets import resolve_evidence_budgets
        from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
        from src.runtime.nodes.synthesis.pipeline import run_synthesis_pipeline

        class ExcerptSynthesizer:
            def __init__(self):
                self.packet = []

            def invoke(self, messages):
                raw = str(messages[-1].content)
                self.packet = json.loads(raw[raw.index("[", len("[Evidence Packet]")):])
                source = self.packet[0]
                return AnswerDocument(blocks=[CodeBlock(language="python", content=ContentUnit(
                    text=source["excerpt"], basis="excerpt", refs=[source["id"]],
                ))])

        text = "PROMPT = " + repr("explain something " * 300)
        task = RetrievalTask(route="upload", query="explain PROMPT", k=4)
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "uploads" / "long-line-synthesis" / "source.py"
            path.parent.mkdir(parents=True)
            path.write_text(text, encoding="utf-8")
            handle = _index_upload(path)
            try:
                state = build_graph_state_input(user_input=task.query, retriever=handle.retriever,
                    planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
                    request_contract=RequestContract())
                state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))
            finally:
                handle.cleanup()
        hits = state["retrieval"].hit_log
        diagnostic = state["debug"].retrieval_diagnostics[0]
        self.assertEqual(diagnostic.status, "success")
        self.assertTrue(hits)
        self.assertTrue(all(len(hit.evidence.excerpt) > 500 for hit in hits))

        plan = state["planner"].output
        context = build_synthesis_context(state=state)
        profile, _ = resolve_evidence_budgets(plan=plan, limits=synthesis_excerpt_limits(normal_chars=1800))
        prepared = prepare_synthesis_inputs(
            state=state, context=context, budget=profile, max_turns=6,
        )
        self.assertTrue(prepared.evidence_packet)
        synthesizer = ExcerptSynthesizer()
        outcome = run_synthesis_pipeline(
            structured_synthesizer=synthesizer, structured_synthesizer_compact=None,
            prepared=prepared, compact_prepared=None, stage_started=time.perf_counter(),
        )
        self.assertEqual(outcome.synthesis_errors, [])
        self.assertTrue(synthesizer.packet[0]["is_partial"])
        evidence = prepared.evidence_packet[0]
        self.assertTrue(any(
            evidence.snapshot == hit.evidence.snapshot and evidence.element == hit.evidence.element
            and hit.evidence.selection.start <= evidence.selection.start < evidence.selection.end <= hit.evidence.selection.end
            for hit in hits
        ))
        self.assertEqual(synthesizer.packet[0]["excerpt"], text[evidence.selection.start:evidence.selection.end])
        self.assertEqual([citation.evidence for citation in outcome.result.citations], [evidence])
        self.assertEqual(outcome.result.content.blocks[0].content, ContentUnit(
            text=evidence.excerpt, basis="excerpt", refs=[evidence.id],
        ))
        self.assertEqual(outcome.result.checks, [UnitCheck(
            unit_id="b0.content", reference_status="resolved", support_status="exact_match",
        )])









if __name__ == "__main__":
    unittest.main()
