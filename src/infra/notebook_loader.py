from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nbformat import NotebookNode, from_dict


CANONICAL_UPLOAD_DIRNAME = ".canonical"


@dataclass(frozen=True)
class LoadedNotebook:
    notebook: NotebookNode
    raw_notebook: dict[str, Any]
    added_cell_id_count: int


def load_canonical_notebook(path: str | Path) -> LoadedNotebook:
    raw_notebook = read_notebook_json(path)
    canonical_notebook, added_cell_id_count = canonicalize_notebook_payload(raw_notebook)
    return LoadedNotebook(
        notebook=from_dict(canonical_notebook),
        raw_notebook=canonical_notebook,
        added_cell_id_count=added_cell_id_count,
    )


def read_notebook_json(path: str | Path) -> dict[str, Any]:
    notebook_path = Path(path)
    payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Notebook payload must be an object: {notebook_path}")
    return payload


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
            source=_normalize_cell_source(cell.get("source")),
        )
        added += 1
    return notebook, added


def ensure_canonical_upload_copy(path: str | Path) -> Path:
    source_path = Path(path)
    canonical_dir = source_path.parent / CANONICAL_UPLOAD_DIRNAME
    canonical_dir.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_dir / source_path.name

    loaded = load_canonical_notebook(source_path)
    canonical_path.write_text(
        json.dumps(loaded.raw_notebook, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return canonical_path


def is_internal_canonical_path(path: str | Path) -> bool:
    return CANONICAL_UPLOAD_DIRNAME in Path(path).parts


def _build_deterministic_cell_id(*, cell_index: int, cell_type: str, source: str) -> str:
    digest = hashlib.sha1(
        f"{cell_index}:{cell_type}:{source}".encode("utf-8")
    ).hexdigest()
    return digest[:8]


def _normalize_cell_source(source: Any) -> str:
    if isinstance(source, list):
        text = "".join(str(part) for part in source)
    else:
        text = str(source or "")
    return text.replace("\r\n", "\n").replace("\r", "\n")
