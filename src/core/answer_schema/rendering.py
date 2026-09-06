from __future__ import annotations

import re
from src.core.evidence import EvidenceRef
from src.core.answer_schema.models import (
    ActionReceipt, AnswerDocument, AnswerResponse, CodeBlock, ContentUnit,
    HeadingBlock, ListBlock, ParagraphBlock, ResponseIssue, TableBlock,
    derive_checks_and_citations, document_hash, iter_content_units,
)


def text_document(text: str, *, basis: str = "interaction", refs: list[str] | None = None) -> AnswerDocument:
    if not str(text or "").strip():
        return AnswerDocument()
    return AnswerDocument(blocks=[ParagraphBlock(content=[ContentUnit(text=text, basis=basis, refs=refs or [])])])


def finalize_answer(document: AnswerDocument, evidence: list[EvidenceRef], *, retrieval_required: bool = False, actions: list[ActionReceipt] | None = None, issues: list[ResponseIssue] | None = None) -> AnswerResponse:
    """Resolve the actual packet references without claiming semantic verification."""
    document = AnswerDocument.model_validate(document.model_dump(mode="json"))
    by_id: dict[str, EvidenceRef] = {}
    for item in evidence:
        item = EvidenceRef.model_validate(item.model_dump(mode="json"))
        if item.id in by_id and by_id[item.id] != item:
            raise ValueError("evidence id resolves to different source versions or ranges")
        by_id[item.id] = item
    checks, citations = derive_checks_and_citations(document, list(by_id.values()), retrieval_required=retrieval_required)
    return AnswerResponse(content=document, citations=citations, checks=checks, issues=issues or [], actions=actions or [], content_hash=document_hash(document), retrieval_required=retrieval_required)


def filter_document_units(document: AnswerDocument, keep_paths: set[str]) -> AnswerDocument:
    blocks = []
    for i, block in enumerate(document.blocks):
        prefix = f"b{i}"
        if isinstance(block, ParagraphBlock):
            units = [u for j, u in enumerate(block.content) if f"{prefix}.content.{j}" in keep_paths]
            if units:
                blocks.append(block.model_copy(update={"content": units}))
        elif isinstance(block, ListBlock):
            units = [u for j, u in enumerate(block.items) if f"{prefix}.items.{j}" in keep_paths]
            if units:
                blocks.append(block.model_copy(update={"items": units}))
        elif isinstance(block, (CodeBlock, HeadingBlock)):
            if f"{prefix}.content" in keep_paths:
                blocks.append(block)
        elif isinstance(block, TableBlock):
            if not all(f"{prefix}.columns.{j}" in keep_paths for j in range(len(block.columns))):
                continue
            rows = [row for r, row in enumerate(block.rows) if all(f"{prefix}.rows.{r}.{c}" in keep_paths for c in range(len(row)))]
            if rows:
                blocks.append(block.model_copy(update={"rows": rows}))
    return AnswerDocument(blocks=blocks)


def citation_labels(unit: ContentUnit, response: AnswerResponse) -> str:
    numbers = {citation.evidence.id: citation.number for citation in response.citations}
    return " ".join(f"[{numbers[ref]}]" for ref in unit.refs if ref in numbers)


def _source_location_details(evidence: EvidenceRef) -> str:
    selection = evidence.selection
    details = [f"snapshot {evidence.snapshot.snapshot_id}", f"element {evidence.element.element_id}"]
    if evidence.element.heading_path:
        details.append(" > ".join(evidence.element.heading_path))
    if selection.cell_ids:
        details.append("table cells " + ", ".join(selection.cell_ids))
    else:
        details.append(f"range [{selection.start}, {selection.end})")
    anchors = list(evidence.element.anchors)
    if evidence.element.table is not None and selection.cell_ids:
        anchors.extend(anchor for cell in evidence.element.table.cells if cell.cell_id in selection.cell_ids for anchor in cell.anchors)
    for anchor in anchors:
        location = []
        if anchor.cell_id is not None:
            location.append(f"cell ID {anchor.cell_id}")
        if anchor.cell_index is not None:
            location.append(f"cell {anchor.cell_index + 1}")
        if anchor.line_start is not None:
            location.append(f"lines {anchor.line_start}-{anchor.line_end or anchor.line_start}")
        if anchor.start is not None:
            location.append(f"source range [{anchor.start}, {anchor.end})")
        if anchor.page_no is not None:
            location.append(f"page {anchor.page_no}")
        if anchor.bbox is not None:
            location.append(f"bbox {anchor.bbox}, {anchor.coordinate_space}, origin {anchor.coordinate_origin}")
        if anchor.page_width is not None and anchor.page_height is not None:
            location.append(f"page size {anchor.page_width} x {anchor.page_height}")
        if location:
            location.append(f"precision {anchor.precision}")
            details.append(", ".join(location))
    return "; ".join(dict.fromkeys(details))


def export_answer_text(response: AnswerResponse, *, include_sources: bool = False) -> str:
    """Render the body, or a delivery document with sources and limitations.

    ``include_sources`` retains provenance and uncertainty for saved files and
    messages. Action receipts remain separate in both modes.
    """
    response = AnswerResponse.model_validate(response.model_dump(mode="json"))

    def render(unit: ContentUnit) -> str:
        labels = citation_labels(unit, response)
        text = f"{unit.text} {labels}" if labels else unit.text
        if include_sources:
            basis = {"inference": "해석", "example": "예시", "excerpt": "원문 발췌"}.get(unit.basis)
            if basis:
                text += f" ({basis})"
        return text

    def cell(unit: ContentUnit) -> str:
        return render(unit).replace("|", "\\|").replace("\n", "<br>")

    blocks: list[str] = []
    for block in response.content.blocks:
        if isinstance(block, ParagraphBlock):
            blocks.append(" ".join(render(unit) for unit in block.content))
        elif isinstance(block, ListBlock):
            blocks.append("\n".join(f"{str(i) + '.' if block.ordered else '-'} {render(unit)}" for i, unit in enumerate(block.items, 1)))
        elif isinstance(block, HeadingBlock):
            blocks.append(f"{'#' * block.level} {render(block.content)}")
        elif isinstance(block, CodeBlock):
            longest = max((len(m.group()) for m in re.finditer(r"`+", block.content.text)), default=0)
            fence = "`" * max(3, longest + 1)
            code = block.content.text
            newline = "" if code.endswith("\n") else "\n"
            labels = citation_labels(block.content, response)
            rendered_code = f"{fence}{block.language}\n{code}{newline}{fence}" + (f"\n{labels}" if labels else "")
            if include_sources:
                if block.content.basis == "excerpt":
                    description = "원문 코드 발췌"
                elif block.content.basis == "example":
                    description = "코드 예시 · 실행 확인 안 됨"
                else:
                    description = "코드 · 실행 확인 안 됨"
                rendered_code = f"{description}\n\n{rendered_code}"
            blocks.append(rendered_code)
        elif isinstance(block, TableBlock):
            rows = ["| " + " | ".join(cell(u) for u in block.columns) + " |", "| " + " | ".join("---" for _ in block.columns) + " |"]
            rows.extend("| " + " | ".join(cell(u) for u in row) + " |" for row in block.rows)
            blocks.append("\n".join(rows))
    text = "\n\n".join(blocks)
    if include_sources and response.citations:
        sources = []
        for citation in response.citations:
            source = citation.evidence.snapshot
            sources.append(f"[{citation.number}] {source.title or source.source_uri} — {source.source_uri} ({_source_location_details(citation.evidence)})")
        text += "\n\n출처\n" + "\n".join(sources)
    if include_sources:
        notes = [issue.message for issue in response.issues]
        for citation in response.citations:
            source = citation.evidence.snapshot
            notes.extend(f"[{citation.number}] {issue}" for issue in source.quality_issues)
            if source.capture_scope == "provider_excerpt":
                notes.append(f"[{citation.number}] 검색 제공자가 반환한 문서 발췌를 보관한 자료입니다.")
        if any(check.reference_status == "resolved" and check.support_status == "not_evaluated" for check in response.checks):
            notes.append("인용의 원문 위치 연결을 확인했으며, 요약과 해석의 의미적 지지는 별도로 검증하지 않았습니다.")
        if any(check.reference_status == "missing" for check in response.checks):
            notes.append("일부 내용은 필요한 원문 근거를 연결하지 못했습니다.")
        if any(check.support_status == "unsupported" for check in response.checks):
            notes.append("일부 발췌 내용이 연결된 원문과 일치하지 않습니다.")
        if notes:
            text += "\n\n참고 및 제한\n" + "\n".join(f"- {note}" for note in dict.fromkeys(notes))
    return text
