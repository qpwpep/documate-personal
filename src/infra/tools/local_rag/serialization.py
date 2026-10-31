from __future__ import annotations

import math

from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, dedupe_search_hits
from src.infra.chroma_store import normalize_l2_distance


def build_local_hits(
    docs_with_scores: list[tuple[EvidenceRef, float | None]],
) -> list[SearchHit]:
    hits: list[SearchHit] = []
    for rank, (evidence, raw_score) in enumerate(docs_with_scores, start=1):
        if raw_score is not None and not math.isfinite(raw_score):
            raw_score = None
        score = normalize_l2_distance(raw_score)
        hits.append(SearchHit(evidence=evidence, score=RetrievalScore(
            metric="l2", raw=raw_score, normalized=score, direction="lower"), rank=rank))
    return dedupe_search_hits(hits)
