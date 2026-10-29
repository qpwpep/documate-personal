"""Source position views that refer to one unchanged excerpt in model input."""
from __future__ import annotations

from src.core.evidence import EvidenceRef, source_line_records


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
