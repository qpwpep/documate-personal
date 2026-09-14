"""Actual, deterministic DOCX parsing; no inference models or external APIs needed."""

from __future__ import annotations

import hashlib

import pytest

pytest.importorskip("docling")

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE

from src.core.uploads import UploadRecord
from src.infra.docling_adapter import convert_file
from src.infra.document_ingestion import ConversionPolicy


def convert_word(word, tmp_path):
    path = tmp_path / "formatted.docx"
    word.save(path)
    raw = path.read_bytes()
    file = UploadRecord(file_id="formatted", name=path.name, path=str(path), size_bytes=len(raw),
                        content_hash="sha256:" + hashlib.sha256(raw).hexdigest(),
                        source_uri="upload://local/formatted.docx")
    # The Word pipeline is declarative and does not load the PDF/OCR models.
    return convert_file(file, ConversionPolicy(artifacts_path=str(tmp_path)))


@pytest.mark.parametrize("parts", [
    ("Retry ", "maximum", " is  24 attempts.\tKeep\nline break."),
    ("DOCU", "MATE", "-2026"),
    ("문단", "공백", "보존"),
    ("  Leading\t", "middle", "\u00a0trailing  "),
])
def test_format_changes_preserve_one_paragraph_and_exact_whitespace(tmp_path, parts):
    """Run boundaries preserve complete text, whitespace and independent paragraph ancestry."""
    word = Document()
    word.add_heading("Formatting check", 1)
    paragraph = word.add_paragraph(parts[0])
    paragraph.add_run(parts[1]).bold = True
    paragraph.add_run(parts[2]).italic = True
    word.add_paragraph("Next paragraph stays separate.")
    parsed = convert_word(word, tmp_path)
    assert [(e.kind, e.text, e.heading_path) for e in parsed.elements] == [
        ("heading", "Formatting check", []),
        ("paragraph", "".join(parts), ["Formatting check"]),
        ("paragraph", "Next paragraph stays separate.", ["Formatting check"]),
    ]
    assert all(not element.anchors for element in parsed.elements)


def test_hyperlink_preserves_visible_text_and_its_whitespace(tmp_path):
    """A hyperlink inside a paragraph neither removes spaces nor inserts its target URL."""
    word = Document()
    paragraph = word.add_paragraph("Read ")
    link = OxmlElement("w:hyperlink")
    link.set(qn("r:id"), paragraph.part.relate_to("https://example.invalid/docs", RELATIONSHIP_TYPE.HYPERLINK, is_external=True))
    run = OxmlElement("w:r")
    text = OxmlElement("w:t")
    text.set(qn("xml:space"), "preserve")
    text.text = "the  docs"
    run.append(text)
    link.append(run)
    paragraph._p.append(link)
    paragraph.add_run(" before retrying.").italic = True
    parsed = convert_word(word, tmp_path)
    assert [(e.kind, e.text) for e in parsed.elements] == [("paragraph", "Read the  docs before retrying.")]


def test_formatted_list_keeps_its_complete_text_and_list_role(tmp_path):
    """Formatted list runs remain one searchable list item with the original spaces."""
    word = Document()
    paragraph = word.add_paragraph("Keep ", style="List Bullet")
    paragraph.add_run("24").bold = True
    paragraph.add_run(" samples.")
    word.add_paragraph("Following body paragraph.")
    parsed = convert_word(word, tmp_path)
    assert [(e.kind, e.text) for e in parsed.elements] == [
        ("list", "Keep 24 samples."), ("paragraph", "Following body paragraph."),
    ]


def test_code_keeps_indentation_and_blank_lines_separate_from_body(tmp_path):
    """Code formatting preserves its block and indentation without absorbing the following paragraph."""
    word = Document()
    word.styles.add_style("Code", WD_STYLE_TYPE.PARAGRAPH)
    word.add_heading("Usage", 1)
    word.add_paragraph("if flag:", style="Code")
    paragraph = word.add_paragraph("    use(", style="Code")
    paragraph.add_run("24").bold = True
    paragraph.add_run(")")
    word.add_paragraph("", style="Code")
    word.add_paragraph("\tfinish()", style="Code")
    word.add_paragraph("Following body paragraph.")
    parsed = convert_word(word, tmp_path)
    assert [(e.kind, e.text, e.heading_path) for e in parsed.elements] == [
        ("heading", "Usage", []),
        ("code", "if flag:\n    use(24)\n\n\tfinish()", ["Usage"]),
        ("paragraph", "Following body paragraph.", ["Usage"]),
    ]


def test_rich_table_cell_preserves_literal_source_text_without_markdown(tmp_path):
    """Rich cells retain visible text and paragraph breaks without duplicate body elements or markup."""
    word = Document()
    table = word.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Setting"
    table.cell(0, 1).text = "Value"
    table.cell(1, 0).text = "Limit"
    paragraph = table.cell(1, 1).paragraphs[0]
    paragraph.add_run("cache_")
    paragraph.add_run("key").bold = True
    paragraph.add_run(" <  24")
    table.cell(1, 1).add_paragraph("Keep this paragraph.")
    parsed = convert_word(word, tmp_path)
    assert [e.kind for e in parsed.elements] == ["table"]
    assert [cell.text for cell in parsed.elements[0].table.cells] == [
        "Setting", "Value", "Limit", "cache_key <  24\n\nKeep this paragraph.",
    ]
