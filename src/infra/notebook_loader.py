from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nbformat import NotebookNode, from_dict, validate


@dataclass(frozen=True)
class LoadedNotebook:
    notebook: NotebookNode
    raw_notebook: dict[str, Any]
    added_cell_id_count: int
    source_content: bytes


def load_canonical_notebook(path: str | Path) -> LoadedNotebook:
    return parse_canonical_notebook(Path(path).read_bytes())


def parse_canonical_notebook(source_content: bytes) -> LoadedNotebook:
    raw_notebook = json.loads(source_content.decode("utf-8-sig"))
    if not isinstance(raw_notebook, dict):
        raise ValueError("Notebook payload must be an object")
    canonical_notebook, added_cell_id_count = canonicalize_notebook_payload(raw_notebook)
    return LoadedNotebook(
        notebook=from_dict(canonical_notebook),
        raw_notebook=canonical_notebook,
        added_cell_id_count=added_cell_id_count,
        source_content=source_content,
    )


def canonicalize_notebook_payload(payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
    notebook = copy.deepcopy(payload)
    minor_version = notebook.get("nbformat_minor")
    validate_before_ids = (
        notebook.get("nbformat") == 4 and isinstance(minor_version, int) and minor_version < 5
    )
    # Earlier v4 schemas forbid cell IDs; validate before adding internal anchors.
    if validate_before_ids:
        validate(notebook)
    cells = notebook.get("cells")
    added = 0
    cell_ids: set[str] = set()
    if isinstance(cells, list):
        for cell_index, cell in enumerate(cells):
            if not isinstance(cell, dict):
                continue
            if "id" not in cell:
                source = cell.get("source")
                if not isinstance(source, str) and not (
                    isinstance(source, list) and all(isinstance(part, str) for part in source)
                ):
                    raise ValueError("Notebook cell source must be a string or a list of strings")
                cell["id"] = _build_deterministic_cell_id(
                    cell_index=cell_index,
                    cell_type=str(cell.get("cell_type") or ""),
                    source=normalize_cell_source(source),
                )
                added += 1
            cell_id = cell["id"]
            if isinstance(cell_id, str):
                # nbformat otherwise repairs duplicates with random IDs.
                if cell_id in cell_ids:
                    raise ValueError(f"duplicate Notebook cell ID: {cell_id}")
                cell_ids.add(cell_id)
    if not validate_before_ids:
        validate(notebook)
    return notebook, added


def _build_deterministic_cell_id(*, cell_index: int, cell_type: str, source: str) -> str:
    digest = hashlib.sha1(
        f"{cell_index}:{cell_type}:{source}".encode("utf-8")
    ).hexdigest()
    return digest[:8]


def normalize_cell_source(source: str | list[str]) -> str:
    text = "".join(source) if isinstance(source, list) else source
    return text.replace("\r\n", "\n").replace("\r", "\n")
