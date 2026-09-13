"""Build small, original bilingual fixtures for document ingestion tests.

Run with the bundled artifact Python via ``uv run --no-project --python ...``.
The source text is deliberately owned by this repository; no customer documents
or copyrighted example papers are downloaded for the OCR benchmark.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


TITLE = "문서 처리 실험 Document processing study"
HEADING = "학습 자료 Learning notes"
LINES = [
    "문서 검색은 원본 페이지와 표의 구조를 함께 보존합니다.",
    "Document retrieval preserves source pages and table structure.",
    "실험 식별자는 DOCUMATE-2026이며 표본 수는 24개입니다.",
    "The experiment identifier is DOCUMATE-2026 and the sample size is 24.",
    "한국어와 English text를 같은 문단에서 읽고 검증합니다.",
]
TABLE = [
    ["항목 Item", "수량 Count", "결과 Result"],
    ["학습 Training", "16", "완료 Complete"],
    ["평가 Evaluation", "8", "대기 Pending"],
]
PAGE_TWO = [
    "추가 검증 Additional validation",
    "두 번째 페이지의 정답은 42입니다.",
    "The answer on the second page is 42.",
]


def generate(output: Path, qa: Path, font_path: Path) -> None:
    from docx import Document
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Inches, Pt, RGBColor
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfgen import canvas

    poppler = shutil.which("pdftoppm")
    if poppler is None:
        raise RuntimeError("pdftoppm is required to generate and verify raster fixtures")
    if not font_path.is_file():
        raise FileNotFoundError(f"A Korean TrueType font is required: {font_path}")
    output.mkdir(parents=True, exist_ok=True)
    qa.mkdir(parents=True, exist_ok=True)
    pdfmetrics.registerFont(TTFont("FixtureKorean", str(font_path)))
    pdf_path = output / "bilingual.pdf"
    pdf = canvas.Canvas(str(pdf_path), pagesize=(612, 792), invariant=True)
    pdf.setTitle(TITLE)
    pdf.setFont("FixtureKorean", 17)
    pdf.drawString(42, 742, TITLE)
    pdf.setFont("FixtureKorean", 14)
    pdf.drawString(42, 700, HEADING)
    pdf.setFont("FixtureKorean", 11)
    for index, line in enumerate(LINES):
        pdf.drawString(42, 670 - index * 25, line)
    top, heights, widths = 506, 34, [205, 110, 205]
    for row_no, row in enumerate(TABLE):
        x = 42
        for width, value in zip(widths, row, strict=True):
            if row_no == 0:
                pdf.setFillColorRGB(0.9, 0.94, 0.97)
                pdf.rect(x, top - (row_no + 1) * heights, width, heights, fill=1, stroke=0)
            pdf.setStrokeColorRGB(0.65, 0.65, 0.65)
            pdf.rect(x, top - (row_no + 1) * heights, width, heights, fill=0, stroke=1)
            pdf.setFillColorRGB(0, 0, 0)
            pdf.drawString(x + 9, top - row_no * heights - 22, value)
            x += width
    pdf.setFont("FixtureKorean", 10)
    pdf.drawString(42, 53, "1")
    pdf.showPage()
    pdf.setFont("FixtureKorean", 17)
    pdf.drawString(42, 742, PAGE_TWO[0])
    pdf.setFont("FixtureKorean", 11)
    for index, line in enumerate(PAGE_TWO[1:]):
        pdf.drawString(42, 700 - index * 25, line)
    pdf.drawString(42, 53, "2")
    pdf.save()

    # A raster-only PDF and its identical PNG allow engine comparisons without
    # native PDF text accidentally bypassing OCR.
    subprocess.run([poppler, "-f", "1", "-l", "1", "-r", "150", "-singlefile",
                    "-png", str(pdf_path), str(output / "bilingual_scan")], check=True)
    scan = canvas.Canvas(str(output / "bilingual_scan.pdf"), pagesize=(612, 792), invariant=True)
    scan.setTitle(TITLE)
    scan.drawImage(str(output / "bilingual_scan.png"), 0, 0, width=612, height=792)
    scan.save()
    subprocess.run([poppler, "-r", "90", "-png", str(pdf_path), str(qa / "bilingual")], check=True)

    word = Document()
    section = word.sections[0]
    section.top_margin = section.bottom_margin = Inches(0.7)
    for name in ("Normal", "Title", "Heading 1"):
        style = word.styles[name]
        style.font.name = "Malgun Gothic"
        style.font.color.rgb = RGBColor(0, 0, 0)
        style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Malgun Gothic")
    word.styles["Normal"].font.size = Pt(10)
    word.add_paragraph(TITLE, style="Title")
    word.add_heading(HEADING, level=1)
    for line in LINES:
        word.add_paragraph(line)
    table = word.add_table(rows=len(TABLE), cols=3)
    table.style = "Table Grid"
    for row_no, values in enumerate(TABLE):
        for cell, value in zip(table.rows[row_no].cells, values, strict=True):
            cell.text = value
            if row_no == 0:
                shade = OxmlElement("w:shd")
                shade.set(qn("w:fill"), "E5EFF7")
                cell._tc.get_or_add_tcPr().append(shade)
        if row_no == 0:
            header = OxmlElement("w:tblHeader")
            table.rows[0]._tr.get_or_add_trPr().append(header)
    word.save(output / "bilingual.docx")
    expectation = {
        "source": "Original synthetic repository fixture; generated by script/generate_document_fixtures.py",
        "title": TITLE, "heading": HEADING, "lines": LINES, "table": TABLE,
        "second_page": PAGE_TWO, "scan_dpi": 150,
        "reference_text": "\n".join([TITLE, HEADING, *LINES, *(" ".join(row) for row in TABLE)]),
    }
    (output / "bilingual_expected.json").write_text(json.dumps(expectation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tests/fixtures/documents"))
    parser.add_argument("--qa", type=Path, default=Path("output/docling/fixture_qa"))
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    args = parser.parse_args()
    generate(args.output, args.qa, args.font)
