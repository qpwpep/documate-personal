from __future__ import annotations

from typing import Any


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def slice_from_index(items: list[Any], start_index: int) -> list[Any]:
    if start_index < 0:
        start_index = 0
    if start_index >= len(items):
        return []
    return items[start_index:]
