"""Physical source coordinates survive selection and answer delivery."""
from pathlib import Path

import pytest

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.evidence import build_evidence, selected_source_anchors
from src.core.evidence import source_line_records
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.infra.chunking import chunk_python_text


def test_selected_sample_statement_keeps_physical_line_thirty_in_export():
    path = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/uploads/sample_data_analysis.py"
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    statement = 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
    upload = chunk_python_text(path=str(path), text=text, source_content=raw,
                               chunk_size=2000, chunk_overlap=0)
    original = upload.hydrate(upload.chunks[0])
    start = text.index(statement)
    selected = build_evidence(snapshot=original.snapshot, element=original.element,
                              start=start, end=start + len(statement))

    assert selected.excerpt == statement
    anchors = selected_source_anchors(selected)
    assert [(item.line_start, item.line_end) for item in anchors] == [(30, 30)]
    response = finalize_answer(text_document(statement, basis="excerpt", refs=[selected.id]),
                               [selected], retrieval_required=True)
    exported = export_answer_text(response, include_sources=True)
    assert "lines 30-30" in exported
    assert "lines 1-36" not in exported


def test_exact_element_with_original_offset_keeps_physical_origin():
    text = "# 앞부분\r\n\r\nvalue = 1\r\n"
    snapshot = build_snapshot(source_uri="upload:///offset.py", title="offset.py",
                              media_type="text/x-python", source_type="upload", content=text,
                              parser="test", parser_version="1")
    element = DocumentElement(element_id="section", kind="code", language="python", text=text,
                              anchors=[SourceAnchor(kind="code", start=600, end=600 + len(text),
                                                    line_start=17, line_end=19, precision="exact")])
    start = text.index("value")
    evidence = build_evidence(snapshot=snapshot, element=element, start=start, end=start + len("value = 1"))
    assert source_line_records(evidence) == [{
        "line_number": 19, "text": "value = 1", "start": start, "end": start + 9,
        "partial_start": False, "partial_end": True,
    }]
    anchor, = selected_source_anchors(evidence)
    assert (anchor.line_start, anchor.line_end, anchor.start, anchor.end) == (19, 19, 600 + start, 600 + start + 9)


def _mapped_evidence(*, anchors: list[SourceAnchor], start: int, end: int):
    text = "alpha\r\n\r\nbeta\rgamma\n"
    snapshot = build_snapshot(source_uri="upload:///mapped.py", title="mapped.py",
                              media_type="text/x-python", source_type="upload", content=text,
                              parser="test", parser_version="1")
    return build_evidence(
        snapshot=snapshot,
        element=DocumentElement(element_id="section", kind="code", text=text, anchors=anchors),
        start=start, end=end,
    )


def test_duplicate_exact_anchors_keep_one_physical_line_mapping():
    anchor = SourceAnchor(kind="code", start=600, end=620, line_start=17, line_end=20,
                          precision="exact")
    evidence = _mapped_evidence(anchors=[anchor, anchor.model_copy(deep=True)], start=9, end=13)

    assert source_line_records(evidence) == [{
        "line_number": 19, "text": "beta", "start": 9, "end": 13,
        "partial_start": False, "partial_end": True,
    }]
    selected, = selected_source_anchors(evidence)
    assert (selected.line_start, selected.line_end) == (19, 19)


@pytest.mark.parametrize(("start", "end"), [(0, 13), (9, 20), (0, 20)])
def test_partial_anchor_does_not_claim_a_mapping_for_the_whole_selection(start, end):
    anchor = SourceAnchor(kind="code", start=9, end=13, line_start=19, line_end=19,
                          precision="exact")
    evidence = _mapped_evidence(anchors=[anchor], start=start, end=end)

    assert source_line_records(evidence) == []


@pytest.mark.parametrize("offset", [0, 2, 7, 20])
def test_empty_selection_has_no_physical_source_lines(offset):
    anchor = SourceAnchor(kind="code", start=600, end=620, line_start=17, line_end=20,
                          precision="exact")
    evidence = _mapped_evidence(anchors=[anchor], start=offset, end=offset)

    assert evidence.excerpt == ""
    assert source_line_records(evidence) == []
