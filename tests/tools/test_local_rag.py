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
from src.core.evidence import EvidenceRef, parse_search_hits
from src.infra.chunking import chunk_notebook_path, chunk_python_text
from src.core.uploads import UploadRecord
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool
from src.infra.tools.local_rag.ranking import rank_retrieval_rows
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
        from src.runtime.nodes.retrieval.executor import collect_retrieval_result
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
                payload = build_upload_search_tool()(requirement=task.requirement, query=task.query, k=task.k, retriever=handle.retriever)
            finally:
                handle.cleanup()
        errors = []
        hit_payloads, diagnostic = collect_retrieval_result(
            raw_payload=payload, tool_name="upload_search", route="upload", query=task.query,
            attempt=1, local_errors=errors, task=task,
        )
        hits = parse_search_hits(hit_payloads)
        self.assertEqual(errors, [])
        self.assertEqual(diagnostic.status, "success")
        self.assertTrue(hits)
        self.assertTrue(all(len(hit.evidence.excerpt) > 500 for hit in hits))

        plan = PlannerOutput(use_retrieval=True, tasks=[task])
        state = build_graph_state_input(
            user_input=task.query, planner=PlannerState(output=plan), request_contract=RequestContract(),
            retrieval={"hit_log": hit_payloads},
        )
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




    def test_chunk_boundary_does_not_discard_complete_source_ast(self) -> None:
        """Code facts retain original line numbers even when indexing splits a call."""
        text = "# setup\n" * 20 + "model = train_model(\n    data,\n    learning_rate=0.01,\n    epochs=20,\n)\n"
        indexed = chunk_python_text(path="uploads/session/model.py", text=text, chunk_size=45, chunk_overlap=5)
        chunk = next(doc for doc in indexed.chunks if "learning_rate" in doc.page_content)
        evidence = indexed.hydrate(chunk)
        call = evidence.element.metadata["code_metadata"]["calls"][0]
        self.assertEqual(call, {"call_name": "train_model", "kwargs": {"learning_rate": "0.01", "epochs": "20"}, "line": 21})







    def test_rank_retrieval_rows_prefers_parameter_cell_for_parameter_queries(self) -> None:
        import_source = "from sklearn.model_selection import train_test_split\nfrom sklearn.preprocessing import StandardScaler"
        usage_source = "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)"
        first = chunk_python_text(path="imports.py", text=import_source, chunk_size=800, chunk_overlap=120)
        second = chunk_python_text(path="usage.py", text=usage_source, chunk_size=800, chunk_overlap=120)
        ranked = rank_retrieval_rows(
            [(first.hydrate(first.chunks[0]), 0.3), (second.hydrate(second.chunks[0]), 0.28)],
            query="sample_pipeline.ipynb 기준으로 train_test_split 파라미터를 찾아줘",
        )
        self.assertEqual(ranked[0][0].excerpt, usage_source)



if __name__ == "__main__":
    unittest.main()
