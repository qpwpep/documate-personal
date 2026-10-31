import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.evidence import selected_source_anchors
from src.core.planner_schema import RetrievalRequirement
from src.core.uploads import UploadRecord
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool
from tests.core.helpers import mock_chat_model
from tests.synthesis_fixtures import prepare_synthesis_state, synthesis_excerpt_limits


class _FakeEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), float(text.count("train_test_split"))] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), float(text.count("train_test_split"))]


def _index_upload(path: Path, *, embeddings: Embeddings | None = None):
    data = path.read_bytes()
    file_id = uuid4().hex
    record = UploadRecord(
        file_id=file_id, name=path.name, path=str(path), size_bytes=len(data),
        content_hash="sha256:" + hashlib.sha256(data).hexdigest(),
        source_uri=f"upload:///rag-tests/{file_id}",
    )
    return build_upload_retriever([record], session_id="rag-tests", generation=uuid4().hex,
                                  api_key="test-key", embeddings=embeddings)


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


@pytest.mark.parametrize("source", [123, {"text": "not source text"}, ["text", 123], None])
def test_notebook_upload_rejects_invalid_cell_source(tmp_path, source):
    path = tmp_path / "source.ipynb"
    content = json.dumps({
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": source},
            {"cell_type": "markdown", "metadata": {}, "source": "Valid source text."},
        ],
    }).encode("utf-8")
    path.write_bytes(content)

    with patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings()):
        with pytest.raises(ValueError, match="source.ipynb"):
            handle = _index_upload(path)
            handle.cleanup()

    assert path.read_bytes() == content


def test_notebook_upload_rejects_duplicate_cell_ids(tmp_path):
    path = tmp_path / "duplicate.ipynb"
    content = json.dumps({
        "nbformat": 4, "nbformat_minor": 5, "metadata": {},
        "cells": [
            {"id": "duplicate", "cell_type": "markdown", "metadata": {}, "source": "First source."},
            {"id": "duplicate", "cell_type": "markdown", "metadata": {}, "source": "Second source."},
        ],
    }).encode("utf-8")
    path.write_bytes(content)

    with patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings()):
        with pytest.raises(ValueError, match="duplicate.*cell.*ID"):
            handle = _index_upload(path)
            handle.cleanup()

    assert path.read_bytes() == content


@pytest.mark.parametrize("minor_version", [4, 5])
def test_notebook_upload_preserves_source_bytes_and_stable_cell_locations(tmp_path, minor_version):
    path = tmp_path / "source.ipynb"
    content = json.dumps({
        "nbformat": 4, "nbformat_minor": minor_version, "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": "# 제목\r\n원문\r끝"},
            {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": ["value = 1\r\n", "print(value)\r"]},
        ],
    }, ensure_ascii=False).encode("utf-8")
    path.write_bytes(content)
    cell_ids = []

    with patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings()):
        for _ in range(2):
            handle = _index_upload(path)
            try:
                document, = handle.retriever.source_documents
                assert [element.text for element in document.elements] == [
                    "# 제목\n원문\n끝", "value = 1\nprint(value)\n",
                ]
                assert document.snapshot.content_hash == "sha256:" + hashlib.sha256(content).hexdigest()
                anchors = [element.anchors[0] for element in document.elements]
                assert [(anchor.cell_index, anchor.line_start, anchor.line_end, anchor.precision)
                        for anchor in anchors] == [(0, 1, 3, "exact"), (1, 1, 2, "exact")]
                cell_ids.append([anchor.cell_id for anchor in anchors])
            finally:
                handle.cleanup()

    assert cell_ids[0] == cell_ids[1]
    assert all(cell_ids[0]) and len(set(cell_ids[0])) == 2
    assert path.read_bytes() == content


@pytest.mark.parametrize("notebook", [False, True], ids=["python", "notebook"])
def test_utf8_bom_upload_keeps_raw_hash_and_searchable_definition(tmp_path: Path, notebook: bool) -> None:
    source = '# 원문\r\n\r\ndef greet(name):\r\n    return f"Hello, {name}\ufeff"\r\n'
    path = tmp_path / ("source.ipynb" if notebook else "source.py")
    if notebook:
        payload = json.dumps({
            "nbformat": 4, "nbformat_minor": 5, "metadata": {},
            "cells": [{"id": "greeting", "cell_type": "code", "metadata": {},
                       "execution_count": None, "outputs": [], "source": source}],
        }, ensure_ascii=False)
    else:
        payload = source
    content = payload.encode("utf-8-sig")
    path.write_bytes(content)
    expected_source = source.replace("\r\n", "\n") if notebook else source
    expected_definition = expected_source[expected_source.index("def greet"):].rstrip("\r\n")

    handle = _index_upload(path, embeddings=_FakeEmbeddings())
    try:
        result = build_upload_search_tool()(
            query="greet 함수 정의 원문", retriever=handle.retriever,
            requirement=RetrievalRequirement(symbols=["greet"], match="definition"),
        )
        assert result.diagnostics.answerability == "covered"
        hit, = result.hits
        assert hit.evidence.excerpt == expected_definition
        assert hit.evidence.element.text == expected_source
        assert hit.evidence.snapshot.content_hash == "sha256:" + hashlib.sha256(content).hexdigest()
        anchor, = selected_source_anchors(hit.evidence)
        assert (anchor.line_start, anchor.line_end, anchor.precision) == (3, 4, "exact")
        if notebook:
            assert (anchor.cell_id, anchor.cell_index) == ("greeting", 0)
    finally:
        handle.cleanup()

    assert path.read_bytes() == content


class LocalRagTest(unittest.TestCase):
    @patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings())
    def test_long_line_search_excerpt_survives_synthesis_and_exact_citation(self, _embeddings) -> None:
        """A retrieved fragment of a long source line remains model-visible and exactly citable."""
        from src.core.answer_schema import AnswerDocument, CodeBlock, ContentUnit, UnitCheck
        from src.core.contracts import PlannerState
        from src.core.contracts.boundary.graph import build_graph_state_input
        from src.core.planner_schema import PlannerOutput, RetrievalTask
        from src.core.request_contracts import RequestContract
        from src.runtime.nodes.retrieval import make_retrieve_dispatch_node
        from src.runtime.nodes.synthesis import make_synthesize_node

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

        prepare_synthesis_state(state, limits=synthesis_excerpt_limits(normal_chars=1800))
        prepared = state["retrieval"].prepared_evidence.normal
        self.assertTrue(prepared.evidence)
        synthesizer = ExcerptSynthesizer()
        updates = make_synthesize_node(mock_chat_model(synthesizer.invoke))(state)
        result = updates["response"].result
        self.assertEqual(updates["debug"].synthesis_errors, [])
        self.assertTrue(synthesizer.packet[0]["is_partial"])
        evidence = prepared.evidence[0]
        self.assertTrue(any(
            evidence.snapshot == hit.evidence.snapshot and evidence.element == hit.evidence.element
            and hit.evidence.selection.start <= evidence.selection.start < evidence.selection.end <= hit.evidence.selection.end
            for hit in hits
        ))
        self.assertEqual(synthesizer.packet[0]["excerpt"], text[evidence.selection.start:evidence.selection.end])
        self.assertEqual([citation.evidence for citation in result.citations], [evidence])
        self.assertEqual(result.content.blocks[0].content, ContentUnit(
            text=evidence.excerpt, basis="excerpt", refs=[evidence.id],
        ))
        self.assertEqual(result.checks, [UnitCheck(
            unit_id="b0.content", reference_status="resolved", support_status="exact_match",
        )])

    @patch("src.infra.chroma_store.OpenAIEmbeddings", return_value=_FakeEmbeddings())
    def test_upload_rag_search_preserves_notebook_snapshot_and_raw_l2_scores(
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
            original_bytes = notebook_path.read_bytes()
            original_entries = set(notebook_path.parent.iterdir())

            handle = _index_upload(notebook_path)
            try:
                upload_tool = build_upload_search_tool()
                payload = upload_tool(requirement=RetrievalRequirement(),
                    query="train_test_split parameter",
                    k=2,
                    retriever=handle.retriever,
                    mode="baseline",
                )
            finally:
                handle.cleanup()
            self.assertEqual(notebook_path.read_bytes(), original_bytes)
            self.assertEqual(set(notebook_path.parent.iterdir()), original_entries)
            self.assertEqual(payload.diagnostics.metric, "l2")
            self.assertEqual(payload.diagnostics.score_direction, "lower_is_better")
            self.assertEqual(payload.diagnostics.route, "upload")
            hits = payload.hits
            self.assertEqual({(hit.evidence.snapshot.source_type, hit.evidence.snapshot.source_uri) for hit in hits},
                             {("upload", handle.retriever.upload_files[0].source_uri)})
            self.assertTrue(all(0.0 <= hit.score.normalized <= 1.0 for hit in hits))
            self.assertTrue(all(hit.evidence.element.anchors[0].cell_id for hit in hits))
            self.assertEqual({hit.evidence.snapshot.content_hash for hit in hits},
                             {"sha256:" + hashlib.sha256(original_bytes).hexdigest()})
            self.assertTrue(all(hit.evidence.snapshot.capture_scope == "full_document" for hit in hits))


if __name__ == "__main__":
    unittest.main()
