from src.core.documents import DocumentElement, ParsedDocument, build_snapshot
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.local_rag.requirements import resolve_source_requirement


def test_converted_document_cannot_prove_a_python_definition_is_absent():
    """Complete document conversion does not imply exhaustive Python source analysis."""
    snapshot = build_snapshot(source_uri="upload:///session/pdf", title="guide.pdf", media_type="application/pdf",
                              source_type="upload", content=b"pdf", parser="docling", parser_version="test")
    parsed = ParsedDocument(snapshot=snapshot, elements=[DocumentElement(element_id="p", kind="paragraph", text="run example")])
    result = resolve_source_requirement(requirement=RetrievalRequirement(symbols=["run"], match="definition"),
                                        source_document=parsed, candidate_rows=[])
    assert result.answerability == "unknown"
    assert result.missing_requirements == []


def test_ocr_code_is_not_published_as_an_exact_python_implementation():
    """Code-looking OCR text retains document provenance without claiming native AST coverage."""
    snapshot = build_snapshot(source_uri="upload:///session/pdf", title="guide.pdf", media_type="application/pdf",
                              source_type="upload", content=b"pdf", parser="docling", parser_version="test")
    parsed = ParsedDocument(snapshot=snapshot, elements=[DocumentElement(element_id="p", kind="code", text="def run():\n    return 1")])
    result = resolve_source_requirement(requirement=RetrievalRequirement(symbols=["run"], match="definition"),
                                        source_document=parsed, candidate_rows=[])
    assert result.answerability == "unknown"
    assert result.hits == []
