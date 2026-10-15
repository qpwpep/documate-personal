from __future__ import annotations

import math

from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, dedupe_search_hits
from src.infra.chroma_store import normalize_l2_distance


def build_local_hit_bundle(
    docs_with_scores: list[tuple[EvidenceRef, float | None]],
) -> tuple[list[SearchHit], list[float], list[float]]:
    hits: list[SearchHit] = []
    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    for rank, (evidence, raw_score) in enumerate(docs_with_scores, start=1):
        if raw_score is not None and not math.isfinite(raw_score):
            raw_score = None
        score = normalize_l2_distance(raw_score)
        hits.append(SearchHit(evidence=evidence, score=RetrievalScore(
            metric="l2", raw=raw_score, normalized=score, direction="lower"), rank=rank))
        if raw_score is not None:
            raw_scores.append(raw_score)
        if score is not None:
            normalized_scores.append(score)
    return dedupe_search_hits(hits), normalized_scores, raw_scores
