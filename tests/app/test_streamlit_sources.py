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
        assert not any(child.type in {"strong_open", "em_open"}
                       for token in tokens for child in token.children or [])
        assert all(child.attrGet("href").startswith("#dm-citation-")
                   for token in tokens for child in token.children or [] if child.type == "link_open")
        if layout == "paragraph":
            assert not any(token.type == "heading_open" for token in tokens)
        if layout == "table":
            assert sum(token.type == "td_open" for token in tokens) == 1
    if layout != "table":
        assert expected in exported
    context = next(item.proto.body for item in app.get("html") if "dm-source-answer-context" in item.proto.body)
    assert expected in unescape(context)
    assert "source position" not in context
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
    rendered = unescape(next(item.proto.body for item in app.get("html") if "dm-source-code" in item.proto.body))
    assert 'aria-label="원문 30행"' in rendered
    assert f"<mark>{selected}</mark>" in rendered
    assert 'aria-label="원문 28행"' in rendered
    assert 'aria-label="원문 32행"' in rendered
    assert 'aria-label="원문 1행"' not in rendered
    assert "전체 원문 보기" in [item.label for item in app.expander]
    assert app.code[0].value == text


@pytest.mark.parametrize("inline", [False, True], ids=["block", "inline"])
def test_cited_code_context_preserves_literal_code_and_source_matching(inline):
    code = '__version__ = "1.0"' if inline else '__version__ = "1.0"\n# important comment\nprint("<tag>", "`")\n'
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="upload:///version.py", title="version.py", media_type="text/x-python",
            source_type="upload", content=code, parser="python", parser_version="1",
        ),
        element=DocumentElement(
            element_id="source", kind="code", text=code, language="python",
            anchors=[SourceAnchor(kind="code", start=0, end=len(code), line_start=1,
                                  line_end=len(code.splitlines()), precision="exact")],
        ),
    )
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [{
        "type": "code", "language": "python", "inline": inline,
        "content": {"text": code, "basis": "source", "refs": [evidence.id]},
    }]}), [evidence])
    original = answer.model_dump(mode="json")
    exported = export_answer_text(answer)

    app = AppTest.from_function(_answer_app, args=(original,)).run()

    assert not app.exception
    # Markdown's code renderer replaces inline wrapping styles with its highlighter.
    # Keep this already escaped HTML at the direct HTML rendering boundary.
    contexts = [item.proto.body for item in app.get("html") if "dm-source-answer-context" in item.proto.body]
    assert len(contexts) == 1
    context = contexts[0]
    assert code in unescape(context)
    assert '<code style="white-space:pre-wrap;overflow-wrap:anywhere">' in context
    assert "<strong>" not in context and "<h1>" not in context and "<tag>" not in context
    assert ("<pre>" in context) is (not inline)
    expected_lines = "1행" if inline else "1행, 2행, 3행"
    assert "답변의 코드와 일치하는 원문: " + expected_lines in [item.value for item in app.caption]
    assert app.code[-1].value == code
    assert answer.model_dump(mode="json") == original
    assert export_answer_text(answer) == exported


def _source_app(payload, cited_texts=None):
    from src.app.web.streamlit_sources import render_evidence
    from src.core.evidence import EvidenceRef

    render_evidence(EvidenceRef.model_validate(payload), cited_texts=cited_texts)


def test_source_panel_shows_original_excerpt_before_collapsed_answer_context():
    captured = "Original documentation excerpt."
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://docs.example.org/guide", title="Guide <draft>",
            media_type="text/plain", source_type="official", content=captured,
            parser="text", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    answer_texts = ["첫 번째 답변 문장", "둘째 답변의 <literal> 값", "함께 연결된 세 번째 문장"]
    original = evidence.model_dump(mode="json")

    app = AppTest.from_function(_source_app, args=(original, answer_texts)).run()

    assert not app.exception
    markdown = [item.value for item in app.markdown]
    header = next(value for value in markdown if "dm-source-header" in value)
    assert 'type="button"' in header and 'aria-label="근거 닫기"' in header
    assert ">닫기</button>" in header
    assert "Guide &lt;draft&gt;" in header
    context_element = next(item for item in app.get("html") if "dm-source-answer-context" in item.proto.body)
    context = context_element.proto.body
    assert "이 근거가 연결된 답변" in context
    assert answer_texts[0] in context and "둘째 답변의 &lt;literal&gt; 값" in context
    assert answer_texts[2] in context
    context_panel = next(item for item in app.expander if item.label == "연결된 답변 보기")
    assert not context_panel.proto.expanded
    assert context_element in list(context_panel)
    elements = list(app.main)
    excerpt_heading = next(item for item in app.markdown if item.value == "**답변에 사용한 부분**")
    original_excerpt = next(item for item in app.get("html") if "Original documentation excerpt." in item.proto.body)
    assert elements.index(excerpt_heading) < elements.index(original_excerpt) < elements.index(context_panel)
    assert markdown.index("[현재 출처 페이지 열기](https://docs.example.org/guide)") < markdown.index("**답변에 사용한 부분**")
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original


def test_answer_context_renders_markdown_and_line_breaks_without_executing_html():
    captured = "BATCH_SIZE = 64\nMAX_RETRIES = 4"
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="upload:///batch_limits.py", title="batch_limits.py",
            media_type="text/plain", source_type="upload", content=captured,
            parser="text", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")
    cited_texts = [
        '`batch_limits.py`의 배치 크기는 **64개**입니다.\n<script>alert(1)</script><img src=x onerror=alert(1)>',
        captured,
        "**추가 근거**를 확인하세요.",
    ]

    app = AppTest.from_function(_source_app, args=(original, cited_texts)).run()

    assert not app.exception
    context = next(item.proto.body for item in app.get("html") if "dm-source-answer-context" in item.proto.body)
    assert "<code>batch_limits.py</code>" in context
    assert "<strong>64개</strong>" in context
    assert "`batch_limits.py`" not in context and "**64개**" not in context
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in context
    assert "&lt;img src=x onerror=alert(1)&gt;" in context
    assert "<script>" not in context and "<img " not in context
    assert "BATCH_SIZE = 64<br />\nMAX_RETRIES = 4" in context
    assert "<strong>추가 근거</strong>" in context
    assert evidence.model_dump(mode="json") == original


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


def test_long_official_evidence_previews_broad_range_and_preserves_full_excerpt():
    captured = "## pandas.merge\n\nMerge DataFrames using a database-style join.\n\n" + (
        "Additional parameter context belongs to this retained source.\n" * 20
    ) + "The final retained warning must remain available."
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html",
            title="pandas.merge", media_type="text/markdown", source_type="official",
            content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")

    app = AppTest.from_function(_source_app, args=(original, ["두 데이터프레임을 결합합니다."])).run()

    assert not app.exception
    assert any("넓은 원문 범위" in item.value for item in app.caption)
    assert captured.startswith(app.text[0].value.removesuffix("…"))
    assert "final retained warning" not in app.text[0].value
    excerpt = next(item for item in app.expander if item.label == "답변에 사용한 전체 발췌")
    assert not excerpt.proto.expanded
    assert "The final retained warning must remain available." in excerpt.get("html")[0].proto.body
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original


def test_long_official_evidence_focuses_only_exact_answer_phrase_without_changing_selection():
    exact = "Both key columns containing null values will match each other."
    captured = "Earlier documentation context.\n" * 30 + exact + "\nOther retained context.\n" * 20
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html",
            title="pandas.merge", media_type="text/markdown", source_type="official",
            content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")

    app = AppTest.from_function(_source_app, args=(original, [exact, "This unsupported wording is not in the source."])).run()

    assert not app.exception
    focused = next(item.proto.body for item in app.get("html") if "dm-source-markdown" in item.proto.body)
    assert f">{exact}</mark>" in focused
    assert "Earlier documentation context" not in focused
    assert "unsupported wording" not in focused
    assert "답변 문구와 정확히 일치하는 원문" in " ".join(item.value for item in app.caption)
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original


def test_paraphrased_answer_previews_matching_terms_before_unrelated_source_prefix():
    relevant = "pandas.merge combines two DataFrame objects or named Series with a database-style join."
    captured = "Deprecated copy behavior is unrelated to the question.\n\n" * 20 + relevant + (
        "\n\nThe remaining captured material stays available."
    )
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html",
            title="pandas.merge", media_type="text/markdown", source_type="official",
            content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")
    answer_text = "`pandas.merge`는 두 DataFrame 또는 이름이 있는 Series를 결합합니다."

    app = AppTest.from_function(_source_app, args=(original, [answer_text])).run()

    assert not app.exception
    previews = [item.proto.body for item in app.get("html") if "dm-source-related-preview" in item.proto.body]
    assert previews
    assert "database-style join" in previews[0]
    assert "Deprecated" not in previews[0]
    assert "<mark>pandas.merge</mark>" in previews[0]
    assert any("용어가 포함된 원문" in item.value and "근거성 판정은 아닙니다" in item.value for item in app.caption)
    full_excerpt = next(item for item in app.expander if item.label == "답변에 사용한 전체 발췌")
    assert app.code[0].value == captured
    assert "Deprecated copy behavior" in full_excerpt.get("html")[0].proto.body
    assert evidence.model_dump(mode="json") == original


def test_related_preview_centers_on_late_term_without_matching_identifier_substrings():
    captured = "A NotDataFrameWrapper is unrelated.\n\n" + "Earlier context. " * 80 + (
        "DataFrame joins preserve the selected keys. <script>not executable</script>"
    )
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://docs.example.org/merge", title="merge", media_type="text/markdown",
            source_type="official", content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")

    app = AppTest.from_function(_source_app, args=(original, ["`DataFrame`을 결합합니다."])).run()

    assert not app.exception
    previews = [item.proto.body for item in app.get("html") if "dm-source-related-preview" in item.proto.body]
    assert len(previews) == 1
    assert "<mark>DataFrame</mark> joins preserve the selected keys." in previews[0]
    assert "NotDataFrameWrapper" not in previews[0]
    assert "<script>" not in previews[0] and "&lt;script&gt;" in previews[0]
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original


def test_related_preview_prefers_description_over_standalone_matching_names():
    captured = "Earlier context unrelated to the answer.\n\n" * 20 + (
        "DataFrame\n\nA DataFrame of the two merged objects.\n\nDataFrame.join"
    )
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://docs.example.org/merge", title="merge", media_type="text/markdown",
            source_type="official", content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")

    app = AppTest.from_function(
        _source_app, args=(original, ["`pandas.merge`는 두 DataFrame 객체를 결합합니다."]),
    ).run()

    assert not app.exception
    previews = [item.proto.body for item in app.get("html") if "dm-source-related-preview" in item.proto.body]
    assert len(previews) == 3
    assert "A <mark>DataFrame</mark> of the two merged objects." in previews[0]
    assert "<mark>DataFrame</mark>.join" in previews[2]
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original


def test_broad_excerpt_search_finds_literal_text_without_claiming_semantic_support():
    captured = "Earlier context without the warning.\n\n" * 30 + (
        "Both key columns containing NULL values will match each other.\n"
        "Check the result before using it."
    )
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://docs.example.org/merge", title="merge", media_type="text/markdown",
            source_type="official", content=captured, parser="provider", parser_version="1",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")
    app = AppTest.from_function(_source_app, args=(original, ["결측값 처리에 주의하세요."])).run()

    next(item for item in app.text_input if item.label == "발췌에서 찾기").set_value("null").run()

    assert not app.exception
    results = [item.proto.body for item in app.get("html") if "dm-source-search-result" in item.proto.body]
    assert len(results) == 1
    assert "<mark>NULL</mark>" in results[0]
    assert "Check the result before using it." in results[0]
    assert "Earlier context" not in results[0]
    assert any("근거성 판정" in item.value for item in app.caption)
    assert any("문장별 일치 위치 미확인" in item.value for item in app.caption)
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original

    next(item for item in app.text_input if item.label == "발췌에서 찾기").set_value("absent phrase").run()

    assert not app.exception
    assert any("일치하는 문구가 없습니다" in item.value for item in app.info)
    assert not any("dm-source-search-result" in item.proto.body for item in app.get("html"))


def test_repeated_cited_code_shows_all_confirmed_occurrences_without_guessing_from_prose():
    text = "begin\ntarget = 3\nmiddle\ntarget = 3\nend\n"
    app = _render(text, text, anchor=SourceAnchor(
        kind="code", start=500, end=500 + len(text), line_start=40, line_end=44, precision="exact",
    ), cited_texts=["30행의 `target = 3`을 확인하세요."])

    assert "답변의 코드와 일치하는 원문: 41행, 43행" in [item.value for item in app.caption]
    rendered = unescape(next(item.proto.body for item in app.get("html") if "dm-source-code" in item.proto.body))
    assert rendered.count("<mark>target = 3</mark>") == 2
    assert 'aria-label="원문 30행"' not in rendered


def test_official_markdown_shows_excerpt_before_source_details_and_preserves_capture():
    captured = "## numpy.reshape\n\nReturn an array with a new **shape**.\n\n- `order` controls the traversal.\n\n[[source]](https://github.com/numpy/numpy/blob/main/numpy/_core/fromnumeric.py)"
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://numpy.org/doc/stable/reference/generated/numpy.reshape.html",
            title="numpy.reshape", media_type="text/markdown", source_type="official",
            content=captured, parser="provider", parser_version="1", capture_scope="provider_excerpt",
            quality_issues=["html_structure_flattened", "freshness_unverified"],
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
    details = next(item for item in app.expander if item.label == "출처와 수집 정보")
    assert not details.proto.expanded
    assert "원문의 일부만 수집한 자료입니다." in " ".join(item.value for item in details.caption)
    assert len(details.warning) == 2
    assert "최신 문서와 다를 수 있습니다." in details.warning[1].value
    assert "수집한 원문 그대로 보기" in [item.label for item in app.expander]
    assert app.code[0].value == captured


def test_flattened_provider_heading_separates_title_from_description_without_changing_evidence():
    captured = (
        "Title: pandas.merge — pandas 3.0.6 documentation\n"
        "# `pandas.merge`#. pandas.merge(*left*, *right*, *how='inner'*)#. "
        ": Merge DataFrame or named Series objects with a database-style join. "
        "See the [merge guide](../user_guide/merging.html).\n\n"
        "## Examples\n\n```python\nresult = pd.merge(left, right)\n```"
    )
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://pandas.pydata.org/docs/reference/api/pandas.merge.html",
            title="pandas.merge", media_type="text/markdown", source_type="official",
            content=captured, parser="provider", parser_version="1", capture_scope="provider_excerpt",
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )
    original = evidence.model_dump(mode="json")

    app = AppTest.from_function(_source_app, args=(original,)).run()

    assert not app.exception
    rendered = app.get("html")[0].proto.body
    assert "<h1><code>pandas.merge</code></h1>" in rendered
    assert "<p>pandas.merge(<em>left</em>, <em>right</em>, <em>how='inner'</em>)#. " in rendered
    assert ": Merge DataFrame or named Series objects with a database-style join." in rendered
    assert 'href="https://pandas.pydata.org/docs/reference/user_guide/merging.html"' in rendered
    assert "<h2>Examples</h2>" in rendered
    assert '<pre><code class="language-python">result = pd.merge(left, right)\n</code></pre>' in rendered
    assert app.code[0].value == captured
    assert evidence.model_dump(mode="json") == original


@pytest.mark.parametrize(("capture_scope", "captured", "expected_heading"), [
    ("provider_excerpt", "# Read `value#. literal` safely", "Read <code>value#. literal</code> safely"),
    ("provider_excerpt", "# Read [value#. literal](https://example.org) safely",
     'Read <a href="https://example.org">value#. literal</a> safely'),
    ("full_document", "# Heading#. ordinary text", "Heading#. ordinary text"),
])
def test_provider_heading_repair_preserves_literal_markdown_and_full_documents(capture_scope, captured, expected_heading):
    evidence = build_evidence(
        snapshot=build_snapshot(
            source_uri="https://docs.example.org/guide", title="Guide", media_type="text/markdown",
            source_type="official", content=captured, parser="provider", parser_version="1",
            capture_scope=capture_scope,
        ),
        element=DocumentElement(element_id="description", kind="paragraph", text=captured),
    )

    app = AppTest.from_function(_source_app, args=(evidence.model_dump(mode="json"),)).run()

    assert not app.exception
    assert f"<h1>{expected_heading}</h1>" in app.get("html")[0].proto.body
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


def test_official_parameter_definitions_are_readable_without_changing_captured_code():
    captured = (
        "Parameters:\n"
        ":   **a**array\\_like\n"
        "    :   Array to be reshaped.\n\n"
        "    **shape**int or tuple of ints\n"
        "    :   The new shape should be compatible with the original shape.\n\n"
        "    **order**{'C', 'F', 'A'}, optional\n"
        "    :   Read the elements using this index order.\n\n"
        "```python\nresult = values.reshape((2, 3))\n```\n\n"
        "An indented code example:\n\n"
        "    if values.size == 6:\n"
        "        result = values.reshape((2, 3))\n"
    )
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
    rendered = app.get("html")[0].proto.body
    assert "<p><strong>a</strong> array_like</p>" in rendered
    assert "<p>Array to be reshaped.</p>" in rendered
    assert "<p><strong>shape</strong> int or tuple of ints</p>" in rendered
    assert "<p>The new shape should be compatible with the original shape.</p>" in rendered
    assert "**shape**" not in rendered
    assert '<pre><code class="language-python">result = values.reshape((2, 3))\n</code></pre>' in rendered
    assert "<pre><code>if values.size == 6:\n    result = values.reshape((2, 3))\n</code></pre>" in rendered
    assert app.code[0].value == captured


@pytest.mark.parametrize(("captured", "expected_html"), [
    (
        "- **shape**int\n  :   Target shape.\n\n- Keep this next item.",
        "<ul>\n<li>\n<p><strong>shape</strong>int\n:   Target shape.</p>\n</li>\n"
        "<li>\n<p>Keep this next item.</p>\n</li>\n</ul>",
    ),
    (
        "> **shape**int\n> :   Target shape.",
        "<blockquote>\n<p><strong>shape</strong>int\n:   Target shape.</p>\n</blockquote>",
    ),
])
def test_official_parameter_formatting_preserves_list_and_quote_containers(captured, expected_html):
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
    assert expected_html in app.get("html")[0].proto.body
    assert app.code[0].value == captured


@pytest.mark.parametrize(("term", "rendered_term"), [
    ("**shape**int", "<strong>shape</strong> int"),
    ("**shape** int", "<strong>shape</strong> int"),
    ("**shape**: int", "<strong>shape</strong>: int"),
])
def test_official_parameter_formatting_preserves_term_separators_and_adjacent_code(term, rendered_term):
    captured = (
        f"    {term}\n"
        "    :   Target shape.\n\n"
        "    result = values.reshape((2, 3))\n"
    )
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
    rendered = app.get("html")[0].proto.body
    assert f"<p>{rendered_term}</p>" in rendered
    assert "<p>Target shape.</p>" in rendered
    assert "<pre><code>result = values.reshape((2, 3))\n</code></pre>" in rendered
    assert app.code[0].value == captured


def test_official_numpy_retained_parameter_excerpt_renders_nested_indentation_as_prose():
    captured = (
        "numpy.reshape(*a*, */*, *shape*, *order='C'*, *\\**, *copy=None*)[[source]]"
        "(https://github.com/numpy/numpy/blob/v2.5.0/numpy/_core/fromnumeric.py#L207-L299)"
        "[#](#numpy.reshape \"Link to this definition\")\n"
        ":   Returns a reshaped ndarray without changing data.\n\n"
        "    Parameters:\n"
        "    :   **a**array\\_like\n"
        "        :   Array to be reshaped.\n\n"
        "        **shape**int or tuple of ints\n"
        "        :   The new shape should be compatible with the original shape. If an integer, "
        "then the result will be a 1-D array of that length. One shape dimension can be -1. "
        "In this case, the value is inferred from the length of the array and remaining dimensions.\n\n"
        "        **order**{‘C’, ‘F’, ‘A’}, optional\n"
        "        :   Read the elements of `a` using this index order, and place the elements into "
        "the reshaped array using this index order. ‘C’ means to read / write the elements using "
        "C-like index order, with the last axis index changing fastest, back to the first axis "
        "index changing slowest. ‘F’ means to read / write the elements using Fortran-like index "
        "order, with the first index changing fastest, and the last index changing slowest. "
        "Note that the ‘C’ and ‘F’ options take no account of the memory layout of the underlying "
        "array, and only refer to the order of indexing. ‘A’ means to read / write the elements "
        "in Fortran-like index order if `a` is Fortran *contiguous* in memory, C-like order otherwise.\n\n"
        "        **copy**bool, optional\n"
        "        :   If `True`, then the array data is copied. If `None`, a copy will only be made "
        "if it’s required by `order`. For `False` it raises a `ValueError` if a copy cannot be "
        "avoided. Default: `None`."
    )
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
    rendered = app.get("html")[0].proto.body
    assert "<p><strong>a</strong> array_like</p>" in rendered
    assert "<p><strong>shape</strong> int or tuple of ints</p>" in rendered
    assert "<p><strong>order</strong> {‘C’, ‘F’, ‘A’}, optional</p>" in rendered
    assert "<p><strong>copy</strong> bool, optional</p>" in rendered
    assert "<p>Read the elements of <code>a</code>" in rendered
    assert "<p>If <code>True</code>, then the array data is copied." in rendered
    assert "<pre>" not in rendered
    assert app.code[0].value == captured
