from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nbformat import NotebookNode, from_dict


@dataclass(frozen=True)
class LoadedNotebook:
    notebook: NotebookNode
    raw_notebook: dict[str, Any]
    added_cell_id_count: int
    source_content: bytes


def load_canonical_notebook(path: str | Path) -> LoadedNotebook:
    source_content = Path(path).read_bytes()
    raw_notebook = json.loads(source_content.decode("utf-8"))
    if not isinstance(raw_notebook, dict):
        raise ValueError(f"Notebook payload must be an object: {path}")
    canonical_notebook, added_cell_id_count = canonicalize_notebook_payload(raw_notebook)
    return LoadedNotebook(
        notebook=from_dict(canonical_notebook),
        raw_notebook=canonical_notebook,
        added_cell_id_count=added_cell_id_count,
        source_content=source_content,
    )


def canonicalize_notebook_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    notebook = copy.deepcopy(payload)
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        notebook["cells"] = []
        return notebook, 0

    added = 0
    for cell_index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        if str(cell.get("id") or "").strip():
            continue
        cell["id"] = _build_deterministic_cell_id(
            cell_index=cell_index,
            cell_type=str(cell.get("cell_type") or ""),
            source=normalize_cell_source(cell.get("source")),
        )
        added += 1
    return notebook, added


def _build_deterministic_cell_id(*, cell_index: int, cell_type: str, source: str) -> str:
    digest = hashlib.sha1(
        f"{cell_index}:{cell_type}:{source}".encode("utf-8")
    ).hexdigest()
    return digest[:8]


def normalize_cell_source(source: Any) -> str:
    if isinstance(source, list):
        text = "".join(str(part) for part in source)
    else:
        text = str(source or "")
    return text.replace("\r\n", "\n").replace("\r", "\n")
