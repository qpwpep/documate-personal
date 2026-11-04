"""Source position views that refer to one unchanged excerpt in model input."""
from __future__ import annotations

from src.core.evidence import EvidenceRef, selected_source_anchors, source_line_records


def source_scope_prompt_record(evidence: EvidenceRef) -> dict[str, object]:
    """Preserve captured file identity and selection boundaries without adding text."""
    record: dict[str, object] = {
        "file_id": evidence.element.metadata.get("file_id") if evidence.route == "upload" else None,
        "selection": evidence.selection.model_dump(mode="json"),
    }
    if evidence.element.table is not None:
        record["is_partial"] = len(evidence.selection.cell_ids) < len(evidence.element.table.cells)
        return record

    length = len(evidence.element.text)
    end = length if evidence.selection.end is None else evidence.selection.end
    omitted_prefix = evidence.selection.start > 0
    omitted_suffix = end < length
    # Native Python parsing retains one whole-file element; notebook elements
    # retain individual cells. Generic code blocks establish neither boundary.
    kind = "document_element"
    if (evidence.snapshot.parser == "python-ast" and evidence.element.element_id == "python-source"
            and evidence.snapshot.capture_scope == "full_document"):
        kind = "python_file"
    elif evidence.snapshot.parser == "notebook-source":
        kind = "notebook_cell"
    record["is_partial"] = omitted_prefix or omitted_suffix
    record["element_extent"] = {
        "kind": kind, "text_length": length,
        "omitted_prefix": omitted_prefix, "omitted_suffix": omitted_suffix,
    }
    return record


def source_location_prompt_records(evidence: EvidenceRef) -> list[dict[str, object]]:
    """Use human cell numbers at model boundaries without changing source anchors."""
    records = []
    for anchor in selected_source_anchors(evidence):
        record = anchor.model_dump(mode="json", exclude={"cell_index"})
        if anchor.cell_index is not None:
            record["cell_number"] = anchor.cell_index + 1
        records.append(record)
    return records


def source_line_prompt_records(evidence: EvidenceRef) -> list[dict[str, object]]:
    """Keep physical positions; text is already present in the selected excerpt."""
    records = []
    for line in source_line_records(evidence):
        record = {key: value for key, value in line.items() if key != "text"}
        # AST definition ranges exclude the final line ending, not code content.
        record["partial_end"] = (line["partial_end"]
                                 and evidence.element.text[line["end"]:line["end"] + 1] not in {"\r", "\n"})
        records.append(record)
    return records


def table_cell_prompt_records(evidence: EvidenceRef) -> list[dict[str, object]]:
    """Map cells into table_excerpt without duplicating or searching cell text."""
    if evidence.element.table is None or not evidence.selection.cell_ids:
        return []
    selected_ids = set(evidence.selection.cell_ids)
    records = []
    offset = 0
    previous_row = None
    for cell in sorted(evidence.element.table.cells, key=lambda cell: (cell.row, cell.col)):
        if cell.cell_id not in selected_ids:
            continue
        if previous_row is not None:
            offset += 3 if cell.row == previous_row else 1
        records.append({
            "cell_id": cell.cell_id, "row": cell.row, "col": cell.col,
            **({"row_span": cell.row_span} if cell.row_span != 1 else {}),
            **({"col_span": cell.col_span} if cell.col_span != 1 else {}),
            **({"is_header": True} if cell.is_header else {}),
            "excerpt_range": [offset, offset + len(cell.text)],
        })
        offset += len(cell.text)
        previous_row = cell.row
    return records
