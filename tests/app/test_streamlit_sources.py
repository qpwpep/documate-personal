"""Source popovers show captured physical positions without renumbering excerpts."""
from __future__ import annotations

from pathlib import Path
from html import unescape

import pytest
from markdown_it import MarkdownIt
from streamlit.testing.v1 import AppTest

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence


def _answer_app(payload):
    from src.app.web.streamlit_chat import render_answer_response
    from src.core.answer_schema import AnswerResponse

    render_answer_response(AnswerResponse.model_validate(payload), "http://localhost:8000")


@pytest.mark.parametrize(("title", "layout"), [
    ("__init__.py", "paragraph"),
    ("# notes.py", "paragraph"),
    ("notes [draft] & `copy`.py", "paragraph"),
    ("notes | draft.py", "table"),
    ("__init__.py", "inline_code"),
    ("__init__.py", "code"),
])
def test_structured_location_preserves_literal_filename_in_rendered_body_and_txt(title, layout):
    source = "value = 1\n"
    evidence = build_evidence(
        snapshot=build_snapshot(source_uri="upload:///source", title=title, media_type="text/x-python",
                                source_type="upload", content=source, parser="python", parser_version="1"),
        element=DocumentElement(element_id="source", kind="code", text=source, language="python",
                                anchors=[SourceAnchor(kind="code", start=0, end=len(source),
                                                      line_start=1, line_end=1, precision="exact")]),
    )
    unit = {"text": "source position", "basis": "source", "refs": [evidence.id],
            "source_location": {"ref": evidence.id, "start": 0, "end": 9}}
    if layout == "paragraph":
        block = {"type": "paragraph", "content": [unit]}
    elif layout == "table":
        block = {"type": "table", "columns": [{"text": "Source"}], "rows": [[unit]]}
    else:
        block = {"type": "code", "inline": layout == "inline_code", "content": unit}
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [block]}), [evidence])
    original = answer.model_dump(mode="json")
    exported = export_answer_text(answer)

    app = AppTest.from_function(_answer_app, args=(original,)).run()

    assert not app.exception
    expected = f"{title}:1"
    if layout == "code":
        assert app.code[0].value == expected
    else:
        tokens = MarkdownIt("commonmark").enable("table").parse(app.markdown[0].value)
        displayed = ["".join(child.content for child in token.children if child.type in {"text", "code_inline"})
                     for token in tokens if token.type == "inline"]
        assert expected + ("" if layout == "inline_code" else " [1]") in displayed
        assert not any(child.type in {"strong_open", "em_open", "link_open"}
                       for token in tokens for child in token.children or [])
        if layout == "paragraph":
            assert not any(token.type == "heading_open" for token in tokens)
        if layout == "table":
            assert sum(token.type == "td_open" for token in tokens) == 1
    if layout != "table":
        assert expected in exported
    assert export_answer_text(answer) == exported
    assert answer.model_dump(mode="json") == original


def test_answer_code_opens_matching_physical_line_with_context_and_original_copy():
    from src.core.answer_schema import finalize_answer, text_document

    fixture = Path(__file__).resolve().parents[2] / "data/benchmarks/fixtures/uploads/sample_data_analysis.py"
    text = fixture.read_bytes().decode("utf-8")
    selected = 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="upload:///sample_data_analysis.py", title="sample_data_analysis.py",
            media_type="text/x-python", source_type="upload", content=text,
            parser="python", parser_version="1",
        ),
        element=DocumentElement(
            element_id="source", kind="code", text=text, language="python",
            anchors=[SourceAnchor(kind="code", start=0, end=len(text), line_start=1,
                                  line_end=36, precision="exact")],
        ),
    )
    answer = finalize_answer(text_document(
        f"sample_data_analysis.py의 30행: `{selected}`", basis="source", refs=[evidence.id],
    ), [evidence])

    app = AppTest.from_function(_answer_app, args=(answer.model_dump(mode="json"),)).run()

    assert not app.exception
    assert "답변의 코드와 일치하는 원문: 30행" in [item.value for item in app.caption]
    rendered = unescape(app.get("html")[0].proto.body)
    assert 'aria-label="원문 30행"' in rendered
    assert f"<mark>{selected}</mark>" in rendered
    assert 'aria-label="원문 28행"' in rendered
    assert 'aria-label="원문 32행"' in rendered
    assert 'aria-label="원문 1행"' not in rendered
    assert "전체 원문 보기" in [item.label for item in app.expander]
    assert app.code[0].value == text


def _source_app(payload, cited_texts=None):
    from src.app.web.streamlit_sources import render_evidence
    from src.core.evidence import EvidenceRef

    render_evidence(EvidenceRef.model_validate(payload), cited_texts=cited_texts)


def _render(text: str, selected: str, *, anchor: SourceAnchor | list[SourceAnchor], offset: int | None = None,
            cited_texts: list[str] | None = None):
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
    app = AppTest.from_function(_source_app, args=(evidence.model_dump(mode="json"), cited_texts)).run()
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
    rendered = unescape(app.get("html")[0].proto.body)
    assert selected in rendered
    assert 'aria-label="원문 30행"' in rendered
    assert 'aria-label="원문 1행"' not in rendered
    assert app.code[0].value == text
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
    assert not app.get("html")
    assert not any(item.proto.show_line_numbers for item in app.code)


def test_cropped_crlf_fragment_uses_original_offset_and_keeps_code_unchanged():
    text = "skip = 1\r\n    target = 3\r\nnext = 4\r\n"
    app = _render(text, "target = 3", anchor=SourceAnchor(
        kind="code", start=600, end=600 + len(text), line_start=17, line_end=19,
        precision="exact",
    ))

    assert "사용 범위: 원문 18행" in [item.value for item in app.caption]
    rendered = unescape(app.get("html")[0].proto.body)
    assert "target = 3" in rendered
    assert 'aria-label="원문 18행"' in rendered
    assert 'aria-label="원문 1행"' not in rendered
    assert app.code[0].value == text.replace("\r\n", "\n")
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


def test_retained_source_identifies_content_hash_without_calling_it_a_document_version():
    text = "target = 3\n"
    app = _render(text, "target = 3", anchor=SourceAnchor(
        kind="code", start=0, end=len(text), line_start=1, line_end=1, precision="exact",
    ))

    captions = [item.value for item in app.caption]
    assert any(value.startswith("자료 내용 해시: sha256:") for value in captions)
    assert not any(value.startswith("자료 버전:") for value in captions)
    assert app.code[0].value == text


def test_current_source_page_link_is_distinct_from_retained_source_content():
    captured_text = "Captured documentation for the answer."
    source_uri = "https://docs.example.org/guide?section=old#retained"
    snapshot = build_snapshot(
        source_uri=source_uri, title="Guide", media_type="text/html",
        source_type="official", content=captured_text, parser="html", parser_version="1",
    )
    evidence = build_evidence(
        snapshot=snapshot,
        element=DocumentElement(element_id="paragraph", kind="paragraph", text=captured_text),
        start=0, end=len(captured_text),
    )

    app = AppTest.from_function(_source_app, args=(evidence.model_dump(mode="json"),)).run()

    assert not app.exception
    markdown = [item.value for item in app.markdown]
    assert f"[현재 출처 페이지 열기]({source_uri})" in markdown
    assert not any("[현재 원문 열기]" in value for value in markdown)
    assert captured_text in app.get("html")[0].proto.body
    assert app.code[0].value == captured_text


def test_repeated_cited_code_shows_all_confirmed_occurrences_without_guessing_from_prose():
    text = "begin\ntarget = 3\nmiddle\ntarget = 3\nend\n"
    app = _render(text, text, anchor=SourceAnchor(
        kind="code", start=500, end=500 + len(text), line_start=40, line_end=44, precision="exact",
    ), cited_texts=["30행의 `target = 3`을 확인하세요."])

    assert "답변의 코드와 일치하는 원문: 41행, 43행" in [item.value for item in app.caption]
    rendered = unescape(app.get("html")[0].proto.body)
    assert rendered.count("<mark>target = 3</mark>") == 2
    assert 'aria-label="원문 30행"' not in rendered


def test_official_markdown_is_readable_with_source_link_first_and_verbatim_capture_available():
    captured = "## numpy.reshape\n\nReturn an array with a new **shape**.\n\n- `order` controls the traversal.\n\n[[source]](https://github.com/numpy/numpy/blob/main/numpy/_core/fromnumeric.py)"
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html",
            title="numpy.reshape", media_type="text/markdown", source_type="official",
            content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )

    app = AppTest.from_function(_source_app, args=(evidence.model_dump(mode="json"),)).run()

    assert not app.exception
    markdown = [item.value for item in app.markdown]
    rendered = app.get("html")[0].proto.body
    assert "<h2>numpy.reshape</h2>" in rendered
    assert "<li><code>order</code> controls the traversal.</li>" in rendered
    assert markdown.index(f"[현재 출처 페이지 열기]({evidence.snapshot.source_uri})") < markdown.index("**답변에 사용한 부분**")
    assert "수집한 원문 그대로 보기" in [item.label for item in app.expander]
    assert app.code[0].value == captured


def test_official_links_resolve_against_captured_page_without_rewriting_code_or_raw_source():
    source_uri = "https://numpy.org/doc/2.1/user/absolute_beginners.html"
    relative = "../reference/generated/numpy.concatenate.html#numpy.concatenate"
    absolute = "https://numpy.org/doc/2.1/reference/generated/numpy.concatenate.html#numpy.concatenate"
    captured = (
        f"[concatenate]({relative}) and [reference][api] and [section](#creating-arrays).\n\n"
        f"![diagram](../images/arrays.svg)\n\n"
        f"`[code example]({relative})`\n\n"
        f"```markdown\n[code example]({relative})\n```\n\n"
        f"[api]: {relative}\n\n"
        "<script>alert('source')</script>\n\n[unsafe](javascript:alert)\n"
    )
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri=source_uri, title="NumPy source", media_type="text/markdown",
            source_type="official", content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="source", kind="paragraph", text=captured),
    )

    app = AppTest.from_function(_source_app, args=(evidence.model_dump(mode="json"),)).run()

    assert not app.exception
    reader_output = "\n".join(item.proto.body for item in app.get("html")) or "\n".join(item.value for item in app.markdown)
    assert absolute in reader_output
    rendered = app.get("html")[0].proto.body
    assert rendered.count(f'href="{absolute}"') == 2
    assert f'href="{source_uri}#creating-arrays"' in rendered
    assert 'src="https://numpy.org/doc/2.1/images/arrays.svg"' in rendered
    assert f"<code>[code example]({relative})</code>" in rendered
    assert f'<code class="language-markdown">[code example]({relative})\n</code>' in rendered
    assert "<script>" not in rendered
    assert 'href="javascript:' not in rendered
    assert app.code[0].value == captured
