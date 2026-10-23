"""Source popovers show captured physical positions without renumbering excerpts."""
from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence


def _source_app(payload):
    from src.app.web.streamlit_sources import render_evidence
    from src.core.evidence import EvidenceRef

    render_evidence(EvidenceRef.model_validate(payload))


def _render(text: str, selected: str, *, anchor: SourceAnchor | list[SourceAnchor], offset: int | None = None):
    snapshot = build_snapshot(
        source_uri="upload:///sample.py", title="sample.py", media_type="text/x-python",
        source_type="upload", content=text, parser="python", parser_version="1",
    )
    start = text.index(selected) if offset is None else offset
    evidence = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(
            element_id="source", kind="code", text=text, language="python",
            anchors=anchor if isinstance(anchor, list) else [anchor],
        ),
        start=start, end=start + len(selected),
    )
    app = AppTest.from_function(_source_app, args=(evidence.model_dump(mode="json"),)).run()
    assert not app.exception
    return app


def test_cropped_sample_code_keeps_physical_line_30_without_relative_gutter():
    fixture = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/uploads/sample_data_analysis.py"
    text = fixture.read_bytes().decode("utf-8")
    selected = 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
    app = _render(text, selected, anchor=SourceAnchor(
        kind="code", start=0, end=len(text), line_start=1, line_end=36, precision="exact",
    ))

    assert "사용 범위: 원문 30행" in [item.value for item in app.caption]
    assert app.code[0].value == selected
    assert not app.code[0].proto.show_line_numbers


def test_coarse_element_provenance_does_not_invent_selected_physical_line():
    text = "header\r\n    target = 3\r\n"
    app = _render(text, "target = 3", anchor=SourceAnchor(
        kind="code", start=0, end=len(text), line_start=40, line_end=41,
        precision="element",
    ))

    captions = [item.value for item in app.caption]
    assert "사용 범위: 수집한 요소의 2행 (원문 줄 번호는 확인되지 않았습니다)" in captions
    assert "사용 범위: 원문 41행" not in captions
    assert not any(item.proto.show_line_numbers for item in app.code)


def test_cropped_crlf_fragment_uses_original_offset_and_keeps_code_unchanged():
    text = "skip = 1\r\n    target = 3\r\nnext = 4\r\n"
    app = _render(text, "target = 3", anchor=SourceAnchor(
        kind="code", start=600, end=600 + len(text), line_start=17, line_end=19,
        precision="exact",
    ))

    assert "사용 범위: 원문 18행" in [item.value for item in app.caption]
    assert app.code[0].value == "target = 3"
    assert not any(item.proto.show_line_numbers for item in app.code)


def test_missing_exact_offsets_does_not_guess_selected_source_lines():
    text = "header\n    target = 3\n"
    app = _render(text, "target = 3", anchor=SourceAnchor(
        kind="code", line_start=17, line_end=18, precision="exact",
    ))

    captions = [item.value for item in app.caption]
    assert "사용 범위: 수집한 요소의 2행 (원문 줄 번호는 확인되지 않았습니다)" in captions
    assert "원문 요소 위치: 17–18행" in captions
    assert "사용 범위: 원문 18행" not in captions


@pytest.mark.parametrize(("kind", "cell_index", "expected"), [
    ("code", None, "사용 범위: 원문 19행"),
    ("notebook", 2, "사용 범위: 셀 3의 19행"),
])
def test_coarse_anchor_before_exact_mapping_does_not_replace_selected_lines(kind, cell_index, expected):
    text = "first\nsecond\n    target = 3\nlast\n"
    app = _render(text, "target = 3", anchor=[
        SourceAnchor(kind="code", line_start=1, line_end=100, precision="element"),
        SourceAnchor(kind=kind, start=0, end=len(text), line_start=17, line_end=20,
                     cell_index=cell_index, precision="exact"),
    ])

    captions = [item.value for item in app.caption]
    assert expected in captions
    assert "사용 범위: 원문 1–100행" not in captions
    assert "원문 요소 위치: 1–100행" in captions


def test_partly_mapped_selection_does_not_claim_one_physical_range_for_all_text():
    text = "outside\nmapped\noutside\n"
    app = _render(text, text, anchor=SourceAnchor(
        kind="code", start=8, end=15, line_start=50, line_end=50, precision="exact",
    ))

    captions = [item.value for item in app.caption]
    assert "사용 범위: 수집한 요소의 1–3행 (원문 줄 번호는 확인되지 않았습니다)" in captions
    assert not any(value.startswith("사용 범위: 원문") for value in captions)


def test_empty_selection_does_not_claim_a_physical_source_line():
    text = "header\nvalue = 3\n"
    app = _render(text, "", offset=3, anchor=SourceAnchor(
        kind="code", start=0, end=len(text), line_start=17, line_end=18, precision="exact",
    ))

    captions = [item.value for item in app.caption]
    assert "사용 범위: 수집한 요소의 1행 (원문 줄 번호는 확인되지 않았습니다)" in captions
    assert not any(value.startswith("사용 범위: 원문") for value in captions)


def test_unmapped_anchor_cannot_supply_notebook_cell_for_known_physical_lines():
    text = "first\nsecond\n    target = 3\nlast\n"
    app = _render(text, "target = 3", anchor=[
        SourceAnchor(kind="notebook", line_start=19, line_end=19, cell_index=9, precision="exact"),
        SourceAnchor(kind="notebook", start=0, end=len(text), line_start=17, line_end=20,
                     cell_index=2, precision="exact"),
    ])

    captions = [item.value for item in app.caption]
    assert "사용 범위: 셀 3의 19행" in captions
    assert "사용 범위: 셀 10의 19행" not in captions
    assert "원문 요소 위치: 셀 10 · 19행" in captions
