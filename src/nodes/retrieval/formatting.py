from __future__ import annotations

from typing import Any


def format_evidence_for_prompt(items: list[dict[str, Any]]) -> str:
    if not items:
        return "No retrieved evidence."

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "unknown")
        source = str(item.get("url_or_path") or "unknown-source")
        source_id = str(item.get("source_id") or "").strip()
        title = str(item.get("title") or "").strip()
        snippet = str(item.get("snippet") or "").strip()
        chunk_id = item.get("chunk_id")
        cell_id = item.get("cell_id")
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        header = f"{index}. [{kind}] {title} - {source}" if title else f"{index}. [{kind}] {source}"
        lines.append(header)
        if source_id:
            lines.append(f"   source_id: {source_id}")
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
