from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

pytest.importorskip("docling")

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import ConversionStatus, DoclingComponentType, ErrorItem, FailureCategory, InputFormat
from docling.datamodel.document import ConversionResult, InputDocument
from docling_core.types.doc import (
    BoundingBox, ContentLayer, CoordOrigin, DocItemLabel, DoclingDocument,
    ProvenanceItem, Size, TableCell as DoclingCell, TableData as DoclingTable,
)

from src.core.documents import ParsedDocument
from src.core.uploads import UploadRecord
from src.infra.docling_adapter import convert_file, map_docling_document
from src.infra.document_ingestion import ConversionPolicy, IngestionError


FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "documents"


@pytest.fixture
def uploaded():
    path = FIXTURES / "bilingual.pdf"
    raw = path.read_bytes()
    return UploadRecord(file_id="sample", name=path.name, path=str(path), size_bytes=len(raw),
                        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
                        source_uri="upload://session/sample/bilingual.pdf")


def result_for(file, doc, status=ConversionStatus.SUCCESS):
    input_doc = InputDocument(path_or_stream=Path(file.path), format=InputFormat.PDF,
                              backend=DoclingParseDocumentBackend)
    return ConversionResult(input=input_doc, status=status, document=doc)


def document():
    doc = DoclingDocument(name="sample")
    doc.add_page(1, Size(width=612, height=792))
    doc.add_page(2, Size(width=612, height=792))
    return doc


def prov(page=1, box=(42, 600, 450, 650)):
    return ProvenanceItem(page_no=page,
                          bbox=BoundingBox(l=box[0], b=box[1], r=box[2], t=box[3], coord_origin=CoordOrigin.BOTTOMLEFT),
                          charspan=(0, 10))


def test_conversion_preserves_source_identity_hierarchy_and_page_coordinates(uploaded):
    """A converted passage retains its exact source bytes, heading ancestry and known page rectangle."""
    doc = document()
    title = doc.add_text(label=DocItemLabel.TITLE, text="학습 자료")
    heading = doc.add_heading("설정", level=1, parent=title)
    passage = doc.add_text(label=DocItemLabel.PARAGRAPH, text="한국어와 English", parent=heading, prov=prov(2))
    doc.add_text(label=DocItemLabel.PAGE_HEADER, text="Repeated header", content_layer=ContentLayer.FURNITURE)
    parsed = map_docling_document(result_for(uploaded, doc), uploaded, {"adapter_version": "1", "device": "cpu"})
    assert parsed.snapshot.content_hash == uploaded.content_hash
    assert parsed.snapshot.source_uri == uploaded.source_uri
    assert parsed.snapshot.title == uploaded.name
    assert parsed.snapshot.parser == "docling"
    assert parsed.snapshot.parser_config == {"adapter_version": "1", "device": "cpu"}
    assert [element.element_id for element in parsed.elements] == [title.self_ref, heading.self_ref, passage.self_ref]
    selected = parsed.elements[2]
    assert selected.parent_id == heading.self_ref
    assert selected.heading_path == ["학습 자료", "설정"]
    assert selected.text == "한국어와 English"
    assert selected.anchors[0].model_dump(exclude_none=True) == {
        "kind": "page", "page_no": 2, "bbox": (42.0, 600.0, 450.0, 650.0),
        "coordinate_space": "points", "coordinate_origin": "bottom_left",
        "page_width": 612.0, "page_height": 792.0, "precision": "element",
    }
    assert ParsedDocument.model_validate_json(parsed.model_dump_json()) == parsed






















