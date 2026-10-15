import hashlib
from contextlib import contextmanager
from uuid import uuid4

import pytest
from langchain_core.embeddings import Embeddings

from src.core.documents import DocumentElement, ParsedDocument, build_snapshot
from src.core.planner_schema import RetrievalRequirement, RetrievalTask
from src.core.uploads import UploadRecord
from src.infra.document_ingestion import ConversionPolicy, DocumentIngestionContext
from src.infra.tools.local_rag import build_upload_retriever, build_upload_search_tool
from src.infra.tools.local_rag.requirements import resolve_source_requirement


def test_converted_document_cannot_prove_a_python_definition_is_absent():
    """Complete document conversion does not imply exhaustive Python source analysis."""
    snapshot = build_snapshot(source_uri="upload:///session/pdf", title="guide.pdf", media_type="application/pdf",
                              source_type="upload", content=b"pdf", parser="docling", parser_version="test")
    parsed = ParsedDocument(snapshot=snapshot, elements=[DocumentElement(element_id="p", kind="paragraph", text="run example")])
    result = resolve_source_requirement(requirement=RetrievalRequirement(symbols=["run"], match="definition"),
                                        source_documents=(parsed,))
    assert result.answerability == "unknown"
    assert result.missing_requirements == []


def test_ocr_code_is_not_published_as_an_exact_python_implementation():
    """Code-looking OCR text retains document provenance without claiming native AST coverage."""
    snapshot = build_snapshot(source_uri="upload:///session/pdf", title="guide.pdf", media_type="application/pdf",
                              source_type="upload", content=b"pdf", parser="docling", parser_version="test")
    parsed = ParsedDocument(snapshot=snapshot, elements=[DocumentElement(element_id="p", kind="code", text="def run():\n    return 1")])
    result = resolve_source_requirement(requirement=RetrievalRequirement(symbols=["run"], match="definition"),
                                        source_documents=(parsed,))
    assert result.answerability == "unknown"
    assert result.hits == []


class _DocumentEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [[float(len(text)), 0.0] for text in texts]

    def embed_query(self, text):
        return [float(len(text)), 0.0]


@pytest.fixture
def converted_uploads(tmp_path, monkeypatch):
    monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: _DocumentEmbeddings())

    @contextmanager
    def create(contents):
        records = []
        parsed = []
        for name, text in contents.items():
            path = tmp_path / name
            raw = b"%PDF-" + name.encode()
            path.write_bytes(raw)
            file_id = uuid4().hex
            record = UploadRecord(file_id=file_id, name=name, path=str(path), size_bytes=len(raw),
                content_hash="sha256:" + hashlib.sha256(raw).hexdigest(), source_uri=f"upload:///documents/{file_id}")
            records.append(record)
            snapshot = build_snapshot(source_uri=record.source_uri, title=name, media_type="application/pdf",
                source_type="upload", content=raw, parser="docling", parser_version="test")
            parsed.append(ParsedDocument(snapshot=snapshot, elements=[DocumentElement(
                element_id="body", kind="paragraph", text=text)]))

        class Converter:
            def convert_many(self, files, **kwargs):
                return tuple(parsed)

        ingestion = DocumentIngestionContext(converter=Converter(), policy=ConversionPolicy(), workspace=tmp_path)
        handle = build_upload_retriever(records, session_id="documents", generation=uuid4().hex,
            api_key="test-key", ingestion=ingestion)
        try:
            yield records, handle.retriever
        finally:
            handle.cleanup()

    return create




