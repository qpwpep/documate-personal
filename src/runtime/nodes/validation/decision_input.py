"""Normalize trusted evidence-preparation records for the Decisions payload."""
from __future__ import annotations

import json
from typing import cast


def encode_decision_input(payload: dict[str, object]) -> str:
    """Share source, task and requirement records across preparation predicates.

    These records are built by the validation nodes, not decoded provider input.
    The casts describe their existing JSON contracts without revalidating domain
    objects. Source IDs retain snapshot and selection identity, never text alone.
    """
    sources: dict[str, dict[str, object]] = {}
    tasks: dict[str, dict[str, object]] = {}

    def source_id(source: dict[str, object]) -> str:
        identity = cast(str, source["id"])
        sources.setdefault(identity, source)
        return identity

    def task_id(task: dict[str, object]) -> str:
        identity = cast(str, task["requirement_id"])
        tasks.setdefault(identity, task)
        return identity

    normalized = dict(payload)
    if "evidence" in payload:
        normalized["evidence"] = [{
            "name": entry["name"],
            "task_id": task_id(cast(dict[str, object], entry["task"])),
            "source_id": source_id(cast(dict[str, object], entry["source"])),
            **({"supporting_source_ids": [source_id(source) for source in
                                          cast(list[dict[str, object]], entry["supporting_sources"])]}
               if "supporting_sources" in entry else {}),
            **({"context_source_ids": [source_id(source) for source in
                                       cast(list[dict[str, object]], entry["context_sources"])]}
               if "context_sources" in entry else {}),
        } for entry in cast(list[dict[str, object]], payload["evidence"])]
    if "requirements" in payload:
        normalized["requirements"] = [{
            "name": entry["name"],
            "task_id": task_id(cast(dict[str, object], entry["task"])),
            "source_ids": [source_id(source) for source in cast(list[dict[str, object]], entry["sources"])],
        } for entry in cast(list[dict[str, object]], payload["requirements"])]
    return json.dumps({
        **normalized, "sources": sources, "tasks": tasks,
    }, ensure_ascii=False)
