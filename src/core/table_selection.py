"""Complete table row selections shared by indexing and evidence budgeting."""

from __future__ import annotations

from collections.abc import Collection

from src.core.documents import DocumentElement, TableCell, TableData


def table_excerpt(table: TableData, cell_ids: Collection[str]) -> str:
    """Render selected original cells identically for indexing and citations."""
    selected = set(cell_ids)
    rows: dict[int, list[str]] = {}
    for cell in sorted(table.cells, key=lambda cell: (cell.row, cell.col)):
        if cell.cell_id in selected:
            rows.setdefault(cell.row, []).append(cell.text)
    return "\n".join(" | ".join(values) for values in rows.values())


def _header_ids(element: DocumentElement) -> dict[str, set[str]]:
    cells = element.table.cells
    roles = element.metadata.get("table_header_cell_ids")
    if roles is None:
        # Older, parser-independent table sources only distinguish header cells.
        return {"column": {cell.cell_id for cell in cells if cell.is_header}, "row": set(), "section": set()}
    known = {cell.cell_id for cell in cells}
    if not isinstance(roles, dict):
        raise ValueError("Table header roles must map roles to cell IDs")
    result: dict[str, set[str]] = {}
    for role in ("column", "row", "section"):
        values = roles.get(role, [])
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ValueError("Table header roles must contain lists of cell IDs")
        if not set(values).issubset(known):
            raise ValueError("Table header role points to an unknown cell")
        result[role] = set(values)
    return result


def _overlaps_columns(cell: TableCell, columns: set[int]) -> bool:
    return any(column in columns for column in range(cell.col, cell.col + cell.col_span))


def table_row_units(
    element: DocumentElement, *, allowed_cell_ids: Collection[str] | None = None,
) -> list[list[str]]:
    """Return data rows with their column, row and nearest section headers.

    A merged cell stays whole even when its origin is on an earlier row. A
    bounded source selection may only yield complete units already inside it;
    missing headers or values are never recovered from unselected source cells.
    """
    if element.table is None:
        raise ValueError("Table row selection requires structured table data")
    cells = sorted(element.table.cells, key=lambda cell: (cell.row, cell.col))
    known = {cell.cell_id for cell in cells}
    allowed = known if allowed_cell_ids is None else set(allowed_cell_ids)
    if not allowed.issubset(known):
        raise ValueError("Allowed table selection contains an unknown cell")
    headers = _header_ids(element)
    body = [cell for cell in cells if cell.cell_id not in headers["column"] | headers["section"]]
    if not body:
        return [[cell.cell_id for cell in cells]] if cells and known.issubset(allowed) else []
    units: list[list[str]] = []
    seen: set[frozenset[str]] = set()
    for row in sorted({cell.row for cell in body}):
        row_cells = [cell for cell in body if cell.row <= row < cell.row + cell.row_span]
        columns = {col for cell in row_cells for col in range(cell.col, cell.col + cell.col_span)}
        selected = {cell.cell_id for cell in row_cells}
        selected.update(cell.cell_id for cell in cells
                        if cell.cell_id in headers["column"] and cell.row <= row and _overlaps_columns(cell, columns))
        sections = [cell for cell in cells
                    if cell.cell_id in headers["section"] and cell.row <= row and _overlaps_columns(cell, columns)]
        if sections:
            nearest_row = max(cell.row for cell in sections)
            selected.update(cell.cell_id for cell in sections if cell.row == nearest_row)
        key = frozenset(selected)
        if key not in seen and selected.issubset(allowed):
            units.append([cell.cell_id for cell in cells if cell.cell_id in selected])
            seen.add(key)
    return units
