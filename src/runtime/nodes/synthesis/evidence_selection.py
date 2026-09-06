from __future__ import annotations

import json
import re

from src.core.evidence import EvidenceRef, SearchHit
from src.core.planner_schema import PlannerOutput

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{1,}|[가-힣]{2,}")
_STOPWORDS = {"uploaded", "upload", "file", "the", "this", "with", "from", "official", "docs", "code"}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN_PATTERN.findall(text) if token.lower() not in _STOPWORDS}


def route_for_evidence(item: EvidenceRef) -> str:
    return item.route


def select_evidence_hits(
    *, user_input: str, hits: list[SearchHit], planner_output: PlannerOutput,
) -> list[SearchHit]:
    """Rank within a source, then preserve route coverage before filling the budget."""
    unique: dict[str, SearchHit] = {}
    for hit in hits:
        unique.setdefault(hit.evidence.id, hit)
    queries = {task.route: task.query or user_input for task in planner_output.tasks}

    def rank_key(hit: SearchHit) -> tuple[int, int]:
        evidence = hit.evidence
        text = " ".join([
            evidence.snapshot.title, evidence.excerpt,
            json.dumps(evidence.element.metadata, ensure_ascii=False),
        ])
        matches = len(_tokens(queries.get(evidence.route, user_input)).intersection(_tokens(text)))
        return (-matches, hit.rank)

    ranked = sorted(unique.values(), key=rank_key)
    first_per_route: list[SearchHit] = []
    seen_routes: set[str] = set()
    for task in planner_output.tasks:
        if task.route in seen_routes:
            continue
        match = next((hit for hit in ranked if hit.evidence.route == task.route), None)
        if match is not None:
            first_per_route.append(match)
        seen_routes.add(task.route)
    first_ids = {hit.evidence.id for hit in first_per_route}
    return first_per_route + [hit for hit in ranked if hit.evidence.id not in first_ids]
