"""Document upload transactions through HTTP, real vector search and graph citations."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.core.answer_schema import export_answer_text
from src.core.documents import DocumentElement, ParsedDocument, SourceAnchor, TableCell, TableData, build_snapshot
from tests.web.test_multi_upload_api import api, answer, context, manifest, staged, sync


class LocalDocumentConverter:
    """Replace only the external document conversion boundary with retained structured sources."""

    def convert_many(self, files, *, policy, workspace, cache_dir=None, deadline=None):
        documents = []
        for file in files:
            raw = Path(file.path).read_bytes()
            assert Path(file.name).suffix in {".pdf", ".docx"}, "Native code must retain its existing parser"
            pdf = file.name.endswith(".pdf")
            warnings = ["OCR extraction may contain recognition errors."] if pdf else []
            snapshot = build_snapshot(
                source_uri=file.source_uri, title=file.name, source_type="upload", content=raw,
                media_type=("application/pdf" if pdf else
                            "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
                parser="docling", parser_version="test-1", parser_config=policy.extraction_options(),
                quality_issues=warnings,
            )
            if pdf:
                page_one = SourceAnchor(kind="page", page_no=1, bbox=(0.1, 0.2, 0.9, 0.8), precision="element")
                page_two = SourceAnchor(kind="page", page_no=2, precision="page")
                elements = [DocumentElement(
                    element_id="value-table", kind="table", heading_path=["Values"],
                    anchors=[page_one, page_two], table=TableData(cells=[
                        TableCell(cell_id="title", row=0, col=0, col_span=2, text="Values", is_header=True,
                                  anchors=[page_one]),
                        TableCell(cell_id="label", row=1, col=0, text="PDF value", anchors=[page_two]),
                        TableCell(cell_id="value", row=1, col=1, text="7", anchors=[page_two]),
                    ]),
                )]
            else:
                elements = [DocumentElement(element_id="body", kind="paragraph", text="DOCX value is 9.",
                                            heading_path=["Values"])]
            documents.append(ParsedDocument(snapshot=snapshot, elements=elements))
        return tuple(documents)

    def close(self):
        pass


@pytest.fixture
def document_api(api):
    api.settings.docling_enabled = True
    api.converter = LocalDocumentConverter()
    api.client.app.state.upload_service.converter = api.converter
    return api


def pdf_bytes():
    return b"%PDF-1.7\n%value=7\n%%EOF"


def docx_bytes():
    result = BytesIO()
    with ZipFile(result, "w") as archive:
        archive.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')
        archive.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:body><w:p><w:r><w:t>DOCX value is 9.</w:t></w:r></w:p></w:body></w:document>')
    return result.getvalue()


def evidence_by_name(result):
    return {citation.evidence.snapshot.title: citation.evidence for citation in result.citations}


def test_mixed_http_uploads_preserve_native_code_structured_table_pages_and_docx_citations(document_api):
    """Mixed attachments reach the real graph with immutable source ranges, table selections and page locations."""
    api = document_api
    current = sync(api, add=[staged(api, "sample.py", "native_value = 3\n"),
                             staged(api, "report.pdf", pdf_bytes()),
                             staged(api, "guide.docx", docx_bytes())])

    result = answer(api, uploads=context(current))

    sources = evidence_by_name(result)
    assert set(sources) == {"sample.py", "report.pdf", "guide.docx"}
    assert {item.element.metadata["file_id"] for item in sources.values()} == {file["file_id"] for file in current["files"]}
    assert sources["sample.py"].snapshot.parser == "python-ast"
    assert sources["sample.py"].element.text == "native_value = 3\n"
    table = sources["report.pdf"]
    assert table.element.text == ""
    assert set(table.selection.cell_ids) == {"title", "label", "value"}
    assert table.excerpt == "Values\nPDF value | 7"
    assert [(anchor.page_no, anchor.bbox) for anchor in table.element.anchors] == [(1, (0.1, 0.2, 0.9, 0.8)), (2, None)]
    assert table.element.table.cells[0].col_span == 2
    assert sources["guide.docx"].excerpt == "DOCX value is 9."
    assert "OCR extraction may contain recognition errors." in export_answer_text(result, include_sources=True)
    assert all(check.support_status == "exact_match" for check in result.checks)
    assert manifest(api) == current
