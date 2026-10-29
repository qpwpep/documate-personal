"""Normalize trusted evidence-preparation records for the Decisions payload."""
from __future__ import annotations

import json
from typing import cast

from src.core.decisions import decision_fingerprint


def encode_decision_input(
    payload: dict[str, object], *, policy: str, rubrics: dict[str, str],
) -> str:
    """Share source, task and requirement records across preparation predicates.

    These records are built by the validation nodes, not decoded provider input.
    The casts describe their existing JSON contracts without revalidating domain
    objects. Source IDs retain snapshot and selection identity, never text alone.
    """
    sources: dict[str, dict[str, object]] = {}
    tasks: dict[str, dict[str, object]] = {}
    requirements: list[dict[str, object]] = []
    requirement_names: dict[str, str] = {}

    def source_id(source: dict[str, object]) -> str:
        identity = cast(str, source["id"])
        sources.setdefault(identity, source)
        return identity

    def task_id(task: dict[str, object]) -> str:
        identity = cast(str, task["requirement_id"])
        tasks.setdefault(identity, task)
        return identity

    def requirement_name(entry: dict[str, object]) -> str:
        normalized = {
            "task_id": task_id(cast(dict[str, object], entry["task"])),
            "source_ids": [source_id(source) for source in cast(list[dict[str, object]], entry["sources"])],
        }
        fingerprint = decision_fingerprint(normalized)
        if fingerprint not in requirement_names:
            name = cast(str, entry.get("name", f"packet_requirement_{len(requirements)}"))
            requirement_names[fingerprint] = name
            requirements.append({"name": name, **normalized})
        return requirement_names[fingerprint]

    normalized = dict(payload)
    if "evidence" in payload:
        normalized["evidence"] = [{
            "name": entry["name"],
            "task_id": task_id(cast(dict[str, object], entry["task"])),
            "source_id": source_id(cast(dict[str, object], entry["source"])),
        } for entry in cast(list[dict[str, object]], payload["evidence"])]
    for entry in cast(list[dict[str, object]], payload.get("requirements", [])):
        requirement_name(entry)
    if "packets" in payload:
        normalized["packets"] = [{
            "name": packet["name"],
            "requirement_names": [requirement_name(entry)
                                  for entry in cast(list[dict[str, object]], packet["requirements"])],
            "source_ids": [source_id(source) for source in cast(list[dict[str, object]], packet["sources"])],
        } for packet in cast(list[dict[str, object]], payload["packets"])]
    if "requirements" in payload or requirements:
        normalized["requirements"] = requirements
    if payload.get("bound_source_response") is not None:
        bound = cast(dict[str, object], payload["bound_source_response"])
        normalized["bound_source_response"] = {
            **{key: value for key, value in bound.items() if key != "sources"},
            "source_ids": [source_id(source) for source in cast(list[dict[str, object]], bound["sources"])],
        }
    return json.dumps({
        "evaluation_policy": policy, "evaluation_rubrics": rubrics,
        **normalized, "sources": sources, "tasks": tasks,
    }, ensure_ascii=False, separators=(",", ":"))
