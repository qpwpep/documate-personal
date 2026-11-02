"""Official async declarations keep their owned prose and original ranges."""

import pytest

from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import RetrievalRequirement
from src.infra.tools.docs_search.ranking import select_docs_candidates
from src.infra.tools.docs_search.sections import parse_doc_sections, scan_declarations


def test_official_async_declaration_owns_timeout_conditions_and_candidates():
    condition = "If a timeout occurs, it cancels aw and raises TimeoutError."
    waiting = "The function waits until cancellation completes, so the wait may exceed timeout."
    text = (
        "asyncio.timeout_at(when)¶\n\nAn absolute deadline context manager.\n\n"
        "async asyncio.wait_for(aw, timeout)¶\n\nWait for an awaitable.\n\n"
        + condition + "\n\n" + waiting + "\n\n"
        "```python\nasync def eternity():\n    await asyncio.sleep(3600)\n```\n\n"
        "async asyncio.wait(aws, *, timeout=None)¶\n\nReturn pending tasks without cancelling them.\n"
    )
    uri = "https://docs.python.org/3/library/asyncio-task.html"
    title = "Coroutines and tasks — Python 3.15.2 documentation"
    sections = parse_doc_sections(text, title=title, url=uri)

    owner, = [section for section in sections if section.start <= text.index(condition) < section.end]

    assert owner.owner == "asyncio.wait_for"
    assert text[owner.start:owner.end].startswith("async asyncio.wait_for(aw, timeout)¶")
    assert waiting in text[owner.start:owner.end]
    assert "async def eternity():" in text[owner.start:owner.end]
    assert "async asyncio.wait(" not in text[owner.start:owner.end]
    assert not any(section.owner.endswith("eternity") for section in sections)

    evidence = build_evidence(
        snapshot=build_snapshot(source_uri=uri, title=title, media_type="text/html", source_type="official",
            content=text, parser="official_html", parser_version="2"),
        element=DocumentElement(element_id="body", kind="paragraph", text=text),
    )
    candidates = select_docs_candidates("asyncio.wait_for timeout cancellation", [SearchHit(
        evidence=evidence, rank=1, score={"metric": "fixture", "direction": "higher"},
    )], RetrievalRequirement(library="Python", symbols=["asyncio.wait_for"], evidence_needs=["explanation"]),
        k=1, allowed_domains=["docs.python.org"])

    assert any("async asyncio.wait_for(aw, timeout)¶" in hit.evidence.excerpt
               and condition in hit.evidence.excerpt and waiting in hit.evidence.excerpt for hit in candidates)
    assert all(hit.evidence.excerpt == text[hit.evidence.selection.start:hit.evidence.selection.end]
               for hit in candidates)


@pytest.mark.parametrize("declaration, name", [
    ("async Reader.read(path)¶", "Reader.read"),
    ("async def read(path):", "read"),
])
def test_async_document_and_async_def_declarations_preserve_complete_original_ranges(declaration, name):
    text = "  " + declaration + "\n"

    found, = scan_declarations(text)

    assert found.name == name
    assert text[found.start:found.end] == declaration


def test_incomplete_async_declaration_remains_unrecognized():
    assert scan_declarations("async Reader.read(\n    path,\n") == []
