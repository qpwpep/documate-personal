from __future__ import annotations

from streamlit.testing.v1 import AppTest


def test_citation_preserves_snapshot_code_and_source_location_without_original_file():
    """A citation remains readable from its captured source, without the upload file."""
    app = AppTest.from_string('''
from src.app.web.streamlit_sources import render_evidence
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence

text = "def example():\\n    return 3\\n"
snapshot = build_snapshot(source_uri="upload:///deleted.py", title="deleted.py", source_type="upload", media_type="text/x-python", content=text, parser="python", parser_version="1")
element = DocumentElement(element_id="code", kind="code", text=text, language="python", anchors=[SourceAnchor(kind="code", line_start=8, line_end=9)])
render_evidence(build_evidence(snapshot=snapshot, element=element, start=15, end=27))
''').run()

    assert not app.exception
    assert any(code.value == "def example():\n    return 3\n" for code in app.code)
    assert any("8–9행" in caption.value for caption in app.caption)
    assert any("deleted.py" in markdown.value for markdown in app.markdown)


def test_notebook_citation_shows_native_cell_and_page_precision_without_invented_coordinates():
    """Evidence details distinguish available cell/page locations from missing regions."""
    app = AppTest.from_string('''
from src.app.web.streamlit_sources import render_evidence
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import build_evidence

snapshot = build_snapshot(source_uri="upload:///book.ipynb", title="book.ipynb", source_type="upload", media_type="application/x-ipynb+json", content="x = 1", parser="notebook", parser_version="1")
element = DocumentElement(element_id="cell", kind="code", text="x = 1", language="python", anchors=[SourceAnchor(kind="notebook", cell_id="native-abc", cell_index=2, line_start=1, line_end=1), SourceAnchor(kind="page", page_no=4, precision="page")])
render_evidence(build_evidence(snapshot=snapshot, element=element))
''').run()

    assert not app.exception
    captions = "\n".join(item.value for item in app.caption)
    assert "셀 3" in captions
    assert "native-abc" in captions
    assert "4페이지" in captions
    assert "페이지 안의 정확한 영역은 제공되지 않았습니다" in captions


def test_official_citation_exposes_original_link_and_capture_limit():
    """A search excerpt is labelled as an excerpt and links only to a web source."""
    app = AppTest.from_string('''
from src.app.web.streamlit_sources import render_evidence
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence

snapshot = build_snapshot(source_uri="https://docs.example.com/api", title="API Docs", source_type="official", media_type="text/plain", content="Captured passage", parser="search", parser_version="1", capture_scope="provider_excerpt")
element = DocumentElement(element_id="p", kind="paragraph", text="Captured passage")
render_evidence(build_evidence(snapshot=snapshot, element=element))
''').run()

    assert not app.exception
    assert any("원문 열기" in item.value and "https://docs.example.com/api" in item.value for item in app.markdown)
    assert any("일부만 수집" in item.value for item in app.caption)
