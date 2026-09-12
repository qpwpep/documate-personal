import json
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.core.evidence import EvidenceRef, parse_search_hits
from src.infra.chunking import chunk_notebook_path, chunk_python_text
from src.infra.tools.local_rag import build_temp_retriever, build_upload_search_tool
from src.infra.tools.local_rag.ranking import rank_retrieval_rows
from src.infra.tools.local_rag.serialization import build_local_snippet, build_query_focused_snippet


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count("train_test_split"))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("train_test_split"))]


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
        from src.runtime.nodes.synthesis.budgets import resolve_synthesis_budget_profile
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
            handle = build_temp_retriever(str(path), api_key="test-key")
            try:
                payload = build_upload_search_tool()(query=task.query, k=task.k, retriever=handle.retriever)
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
        self.assertEqual({len(hit.evidence.excerpt) for hit in hits}, {500})

        plan = PlannerOutput(use_retrieval=True, tasks=[task])
        state = build_graph_state_input(
            user_input=task.query, planner=PlannerState(output=plan), request_contract=RequestContract(),
            retrieval={"hit_log": hit_payloads},
        )
        context = build_synthesis_context(state=state, has_default_slack_destination=False)
        profile = resolve_synthesis_budget_profile(
            user_input=task.query, planner_output=plan, snippet_char_limit=1800,
        )
        prepared = prepare_synthesis_inputs(
            state=state, context=context, budget_profile=profile, max_turns=6,
            prompt_snippet_char_limit=profile.snippet_chars, prompt_evidence_char_budget=profile.evidence_chars,
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

    @patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings())
    def test_index_metadata_stays_compact_and_returned_citations_survive_cleanup(self, _embeddings) -> None:
        """The real index stores lightweight locations while returned citations own their source."""
        text = ("# source marker\n" + ("value = 123456789\n" * 1400))[:20480]
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "uploads" / "compact-index" / "source.py"
            path.parent.mkdir(parents=True)
            path.write_bytes(text.encode("utf-8"))
            handle = build_temp_retriever(str(path), api_key="test-key")
            try:
                stored = handle.retriever.vectorstore.get()["metadatas"]
                self.assertTrue(stored)
                self.assertTrue(all("evidence_ref" not in metadata for metadata in stored))
                metadata_bytes = sum(len(json.dumps(metadata).encode("utf-8")) for metadata in stored)
                self.assertLess(metadata_bytes, len(text.encode("utf-8")) * 2)
                payload = build_upload_search_tool()(query="value", k=2, retriever=handle.retriever)
                returned = parse_search_hits(payload)
                direct = handle.retriever.invoke("value")
                self.assertTrue(direct)
                self.assertTrue(all(EvidenceRef.model_validate_json(doc.metadata["evidence_ref"]).element.text == text for doc in direct))
            finally:
                handle.cleanup()
            path.unlink()
        self.assertTrue(returned)
        self.assertTrue(all(hit.evidence.element.text == text for hit in returned))
        self.assertTrue(all(hit.evidence.excerpt == text[hit.evidence.selection.start:hit.evidence.selection.end] for hit in returned))

    def test_selected_hit_retains_exact_source_text_after_file_removal(self) -> None:
        """A query excerpt keeps its exact range and complete element after the source disappears."""
        from src.infra.tools.local_rag.serialization import build_local_hit_bundle

        text = "# 한글 원문\r\n" + ("setup = 1\r\n" * 60) + "if ready:\r\n    target_call(\r\n        random_state=42\r\n    )\r\n"
        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "source.py"
            path.write_bytes(text.encode("utf-8"))
            indexed = chunk_python_text(path=str(path), text=text, chunk_size=1000, chunk_overlap=100)
            target = next(doc for doc in indexed.chunks if "target_call" in doc.page_content)
            hits, _, _, errors = build_local_hit_bundle([(indexed.hydrate(target), 0.2)], query="target_call random_state")
            path.unlink()
        self.assertEqual(errors, [])
        hit = hits[0]
        self.assertEqual(hit.evidence.element.text, text)
        self.assertEqual(hit.evidence.excerpt, text[hit.evidence.selection.start:hit.evidence.selection.end])
        self.assertIn("    target_call(", hit.evidence.excerpt)
        self.assertIn("\r\n", hit.evidence.excerpt)
        self.assertEqual(hit.score.raw, 0.2)

    def test_same_path_changed_content_has_new_snapshot_identity(self) -> None:
        """Source identity distinguishes different bytes saved at the same path."""
        from src.infra.tools.local_rag.serialization import build_local_hit_bundle

        def snapshot(text):
            indexed = chunk_python_text(path="uploads/session/code.py", text=text, chunk_size=800, chunk_overlap=120)
            hits, _, _, _ = build_local_hit_bundle([(indexed.hydrate(indexed.chunks[0]), None)], query="target")
            return hits[0].evidence.snapshot

        original = snapshot("target = 1\n")
        changed = snapshot("target = 2\n")
        self.assertEqual(original.document_id, changed.document_id)
        self.assertNotEqual(original.snapshot_id, changed.snapshot_id)
        self.assertNotEqual(original.content_hash, changed.content_hash)

    def test_chunk_boundary_does_not_discard_complete_source_ast(self) -> None:
        """Code facts retain original line numbers even when indexing splits a call."""
        text = "# setup\n" * 20 + "model = train_model(\n    data,\n    learning_rate=0.01,\n    epochs=20,\n)\n"
        indexed = chunk_python_text(path="uploads/session/model.py", text=text, chunk_size=45, chunk_overlap=5)
        chunk = next(doc for doc in indexed.chunks if "learning_rate" in doc.page_content)
        evidence = EvidenceRef.model_validate_json(indexed.hydrate(chunk).metadata["evidence_ref"])
        call = evidence.element.metadata["code_metadata"]["calls"][0]
        self.assertEqual(call, {"call_name": "train_model", "kwargs": {"learning_rate": "0.01", "epochs": "20"}, "line": 21})

    def test_query_changes_selection_without_changing_snapshot_or_element(self) -> None:
        """Different queries select different ranges in the same preserved element."""
        from src.infra.tools.local_rag.serialization import build_local_hit_bundle

        text = "alpha(value=1)\n" + "setup = 1\n" * 12 + "beta(value=2)\n"
        indexed = chunk_python_text(path="uploads/session/query.py", text=text, chunk_size=800, chunk_overlap=120)
        doc = indexed.hydrate(indexed.chunks[0])
        first = build_local_hit_bundle([(doc, 0.2)], query="alpha")[0][0].evidence
        second = build_local_hit_bundle([(doc, 0.3)], query="beta")[0][0].evidence
        self.assertEqual(first.snapshot, second.snapshot)
        self.assertEqual(first.element, second.element)
        self.assertNotEqual(first.selection, second.selection)
        self.assertNotEqual(first.id, second.id)

    def test_build_query_focused_snippet_centers_on_matching_identifier(self) -> None:
        text = (
            "import pandas as pd\n\n"
            "sales_q1 = pd.DataFrame(...)\n"
            "sales_q2 = pd.DataFrame(...)\n"
            "profiles = pd.DataFrame(...)\n"
            "# concat example\n"
            "all_sales = pd.concat([sales_q1, sales_q2], ignore_index=True)\n"
            "# groupby example\n"
            'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()\n'
            "# merge example\n"
            'sales_with_profile = all_sales.merge(profiles, on="user_id", how="left")\n'
        )

        snippet = build_query_focused_snippet(
            text,
            query="업로드한 파일에서 groupby를 어떻게 쓰는지 찾아서 설명해줘.",
            max_length=120,
        )

        self.assertIn("groupby", snippet)
        self.assertNotIn("sales_q1 = pd.DataFrame", snippet)

    def test_build_query_focused_snippet_prefers_more_relevant_usage_over_earliest_match(self) -> None:
        text = (
            "# train_test_split helper notes\n"
            "from sklearn.model_selection import train_test_split\n"
            "# later usage\n"
            "X_train, X_test, y_train, y_test = train_test_split(\n"
            "    X,\n"
            "    y,\n"
            "    test_size=0.2,\n"
            "    random_state=42,\n"
            ")\n"
        )

        snippet = build_query_focused_snippet(
            text,
            query="train_test_split random_state parameter",
            max_length=120,
        )

        self.assertIn("random_state=42", snippet)
        self.assertNotIn("helper notes", snippet)

    def test_build_local_snippet_uses_query_window_for_single_chunk_documents(self) -> None:
        text = "header\n" + ("value = 1\n" * 80) + "target_call(random_state=42)\n"

        snippet = build_local_snippet(
            text,
            query="random_state parameter",
            metadata={"document_chunk_count": 1, "document_char_count": len(text)},
        )

        self.assertIn("target_call(random_state=42)", snippet)
        self.assertNotIn("header", snippet)
        self.assertLess(len(snippet), 500)
        self.assertLessEqual(len(snippet.splitlines()), 5)

    def test_build_local_snippet_uses_query_window_for_short_documents(self) -> None:
        text = "header\n" + ("step = 1\n" * 40) + "target_call(random_state=42)\n"

        snippet = build_local_snippet(
            text,
            query="random_state parameter",
            metadata={"document_chunk_count": 3, "document_char_count": len(text)},
        )

        self.assertIn("target_call(random_state=42)", snippet)
        self.assertNotIn("header", snippet)
        self.assertLessEqual(len(snippet.splitlines()), 5)
        self.assertLessEqual(len(text), 1200)

    def test_build_local_snippet_preserves_full_chunk_for_explicit_extraction(self) -> None:
        text = "header\n" + ("step = 1\n" * 40) + "target_call(random_state=42)\n"

        snippet = build_local_snippet(
            text,
            query="show the exact code snippet with random_state",
            metadata={"document_chunk_count": 1, "document_char_count": len(text)},
        )

        self.assertEqual(snippet, text.strip())

    def test_build_local_snippet_places_concat_window_first_for_short_files(self) -> None:
        text = (
            "import pandas as pd\n"
            + ("setup_value = 1\n" * 32)
            + "all_sales = pd.concat([sales_q1, sales_q2], ignore_index=True)\n"
            + "print(all_sales.shape)\n"
        )

        snippet = build_local_snippet(
            text,
            query="concat ignore_index option",
            metadata={"document_chunk_count": 1, "document_char_count": len(text)},
            max_length=768,
        )

        self.assertTrue(snippet.startswith("all_sales = pd.concat"))
        self.assertIn("ignore_index=True", snippet)
        self.assertNotIn("import pandas as pd", snippet)

    def test_chunk_python_text_annotates_document_counts(self) -> None:
        text = "line = 1\n" * 240

        indexed = chunk_python_text(
            path="uploads/session/sample.py",
            text=text,
            chunk_size=200,
            chunk_overlap=20,
        )

        docs = indexed.chunks
        self.assertGreater(len(docs), 1)
        self.assertTrue(all(doc.metadata["document_chunk_count"] == len(docs) for doc in docs))
        self.assertTrue(all(doc.metadata["document_char_count"] == len(text) for doc in docs))

    def test_chunk_python_text_adds_ast_code_metadata(self) -> None:
        indexed = chunk_python_text(
            path="uploads/session/model.py",
            text=(
                "from sklearn.linear_model import LogisticRegression\n"
                "model = LogisticRegression(max_iter=200, random_state=42)\n"
            ),
            chunk_size=800,
            chunk_overlap=120,
        )

        metadata = indexed.parsed.elements[0].metadata["code_metadata"]
        self.assertEqual(metadata["calls"][0]["call_name"], "LogisticRegression")
        self.assertEqual(metadata["calls"][0]["kwargs"]["max_iter"], "200")
        self.assertIn("max_iter=200", metadata["option_literals"])

    def test_chunk_notebook_path_adds_cell_ast_code_metadata(self) -> None:
        with TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "sample_pipeline.ipynb"
            _write_notebook(
                notebook_path,
                "# Sample pipeline",
                "print('setup')\n",
                "from sklearn.linear_model import LogisticRegression\n"
                "model = LogisticRegression(max_iter=200)\n",
            )

            indexed = chunk_notebook_path(
                path=str(notebook_path),
                chunk_size=800,
                chunk_overlap=120,
            )

        matching = [
            element.metadata["code_metadata"]
            for element in indexed.parsed.elements
            if "LogisticRegression" in element.text
        ]
        self.assertEqual(matching[0]["cell_id"], 2)
        self.assertEqual(matching[0]["calls"][0]["call_name"], "LogisticRegression")
        self.assertEqual(matching[0]["calls"][0]["kwargs"]["max_iter"], "200")

    def test_chunk_notebook_path_normalizes_list_sources_without_list_repr(self) -> None:
        with TemporaryDirectory() as temp_dir:
            notebook_path = Path(temp_dir) / "sample_pipeline.ipynb"
            notebook_path.write_text(
                json.dumps(
                    {
                        "cells": [
                            {
                                "cell_type": "code",
                                "execution_count": None,
                                "metadata": {},
                                "outputs": [],
                                "source": [
                                    "from sklearn.preprocessing import StandardScaler\r\n",
                                    "scaler = StandardScaler()\r\n",
                                ],
                            }
                        ],
                        "metadata": {},
                        "nbformat": 4,
                        "nbformat_minor": 5,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            indexed = chunk_notebook_path(
                path=str(notebook_path),
                chunk_size=800,
                chunk_overlap=120,
            )

        docs = indexed.chunks
        self.assertEqual(len(docs), 1)
        expected_source = (
            "from sklearn.preprocessing import StandardScaler\n"
            "scaler = StandardScaler()\n"
        )
        self.assertEqual(docs[0].page_content, expected_source)
        self.assertEqual(docs[0].metadata["document_char_count"], len(expected_source))
        snippet = build_local_snippet(
            docs[0].page_content,
            query="StandardScaler initialization",
            metadata=docs[0].metadata,
        )
        self.assertEqual(snippet, expected_source.strip())
        self.assertNotIn("['from sklearn", docs[0].page_content)
        self.assertNotIn("['from sklearn", snippet)

    def test_rank_retrieval_rows_prefers_parameter_cell_for_parameter_queries(self) -> None:
        import_doc = Document(
            page_content=(
                "from sklearn.model_selection import train_test_split\n"
                "from sklearn.preprocessing import StandardScaler"
            ),
            metadata={"cell_id": 1},
        )
        usage_doc = Document(
            page_content=(
                "X_train, X_test, y_train, y_test = train_test_split("
                "X, y, test_size=0.2, random_state=42)"
            ),
            metadata={"cell_id": 2},
        )

        ranked = rank_retrieval_rows(
            [(import_doc, 0.3), (usage_doc, 0.28)],
            query="sample_pipeline.ipynb 기준으로 train_test_split 파라미터를 찾아줘",
        )

        self.assertEqual(ranked[0][0].metadata["cell_id"], 2)

    @patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings())
    def test_upload_rag_search_preserves_notebook_snapshot_and_raw_l2_scores_without_userwarning(
        self,
        _mock_openai_embeddings,
    ) -> None:
        with TemporaryDirectory() as temp_dir:
            uploads_root = Path(temp_dir) / "uploads" / "session-a"
            notebook_path = uploads_root / "sample_pipeline.ipynb"
            _write_notebook(
                notebook_path,
                "# Sample pipeline",
                "from sklearn.model_selection import train_test_split\n"
                "X_train, X_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
            )

            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                handle = build_temp_retriever(str(notebook_path), api_key="test-key")
                try:
                    upload_tool = build_upload_search_tool()
                    payload = upload_tool(
                        query="train_test_split parameter",
                        k=2,
                        retriever=handle.retriever,
                    )
                finally:
                    handle.cleanup()

            self.assertEqual(caught, [])
            self.assertFalse((notebook_path.parent / ".canonical").exists())
            self.assertEqual(payload["diagnostics"]["metric"], "l2")
            self.assertEqual(payload["diagnostics"]["score_direction"], "lower_is_better")
            self.assertEqual(payload["diagnostics"]["route"], "upload")
            hits = parse_search_hits(payload)
            self.assertEqual({(hit.evidence.snapshot.source_type, hit.evidence.snapshot.source_uri) for hit in hits},
                             {("upload", str(notebook_path))})
            self.assertTrue(all(0.0 <= hit.score.normalized <= 1.0 for hit in hits))
            self.assertTrue(all(hit.evidence.element.anchors[0].cell_id for hit in hits))
            self.assertTrue(all(hit.evidence.snapshot.capture_scope == "full_document" for hit in hits))


if __name__ == "__main__":
    unittest.main()
