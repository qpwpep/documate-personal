from __future__ import annotations

from typing import Any
import math

from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, dedupe_search_hits
from src.infra.chroma_store import normalize_l2_distance
from src.infra.tools._common import to_float_or_none


def score_ranked_rows(
    docs_with_scores: list[tuple[Any, float | None]],
) -> list[tuple[Any, float | None, float | None]]:
    scored_rows: list[tuple[Any, float | None, float | None]] = []
    for doc, score in docs_with_scores:
        raw_score = to_float_or_none(score)
        if raw_score is not None and not math.isfinite(raw_score):
            raw_score = None
        scored_rows.append((doc, normalize_l2_distance(raw_score), raw_score))
    return scored_rows


def build_local_hit_bundle(
    docs_with_scores: list[tuple[Any, float | None]],
) -> tuple[list[SearchHit], list[float], list[float], list[str]]:
    hits: list[SearchHit] = []
    retrieval_warnings: list[str] = []
    raw_scores: list[float] = []
    normalized_scores: list[float] = []
    for rank, (doc, score, raw_score) in enumerate(score_ranked_rows(docs_with_scores), start=1):
        metadata = getattr(doc, "metadata", {})
        try:
            evidence = EvidenceRef.model_validate_json(metadata["evidence_ref"])
            if str(doc.page_content or "") != evidence.excerpt:
                raise ValueError("Indexed text differs from its source range")
            # The indexed source range is the retrieval result. Synthesis selects
            # the final model-visible excerpt within this range and its budget.
        except (KeyError, TypeError, ValueError) as exc:
            retrieval_warnings.append(f"invalid_source_reference: {exc}")
            continue
        hits.append(SearchHit(evidence=evidence, score=RetrievalScore(
            metric="l2", raw=raw_score, normalized=score, direction="lower"), rank=rank))
        if raw_score is not None:
            raw_scores.append(raw_score)
        if score is not None:
            normalized_scores.append(score)
    return dedupe_search_hits(hits), normalized_scores, raw_scores, retrieval_warnings
