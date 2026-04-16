from __future__ import annotations

import math
from typing import Any


def _truncate_prompt_snippet(value: str, *, max_chars: int) -> str:
    snippet = str(value or "").strip()
    if not snippet or max_chars < 1 or len(snippet) <= max_chars:
        return snippet
    if max_chars <= 3:
        return snippet[:max_chars]
    bridge = " ... "
    if max_chars <= len(bridge) + 2:
        return snippet[: max_chars - 3].rstrip() + "..."

    available = max_chars - len(bridge)
    head_chars = max(1, available // 2)
    tail_chars = max(1, available - head_chars)
    head = snippet[:head_chars].rstrip()
    tail = snippet[-tail_chars:].lstrip()
    if head and tail:
        return f"{head}{bridge}{tail}"
    return snippet[: max_chars - 3].rstrip() + "..."


def _normalize_cell_id(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _normalize_relevance_score(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _is_local_evidence(item: dict[str, Any]) -> bool:
    return str(item.get("kind") or "").strip().lower() == "local"


def format_evidence_for_prompt(
    items: list[dict[str, Any]],
    *,
    max_snippet_chars: int = 280,
) -> str:
    if not items:
        return "No retrieved evidence."

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "unknown")
        source = str(item.get("url_or_path") or "unknown-source")
        source_id = str(item.get("source_id") or "").strip()
        title = str(item.get("title") or "").strip()
        raw_snippet = str(item.get("snippet") or "").strip()
        snippet = raw_snippet if _is_local_evidence(item) else _truncate_prompt_snippet(
            raw_snippet,
            max_chars=max_snippet_chars,
        )
        score = _normalize_relevance_score(item.get("score"))
        chunk_id = item.get("chunk_id")
        cell_id = _normalize_cell_id(item.get("cell_id"))
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        header = f"{index}. [{kind}] {title} - {source}" if title else f"{index}. [{kind}] {source}"
        lines.append(header)
        if source_id:
            lines.append(f"   source_id: {source_id}")
        if score is not None:
            lines.append(f"   relevance_score: {score:.3f}")
        if cell_id is not None or chunk_id is not None:
            location_parts: list[str] = []
            if cell_id is not None:
                location_parts.append(f"cell_id={cell_id}")
            if chunk_id is not None:
                location_parts.append(f"chunk_id={chunk_id}")
            if start_offset is not None and end_offset is not None:
                location_parts.append(f"offsets={start_offset}-{end_offset}")
            if location_parts:
                lines.append(f"   location: {', '.join(location_parts)}")
        if snippet:
            lines.append(f"   snippet: {snippet}")
    return "\n".join(lines)
