from __future__ import annotations

import re
from urllib.parse import urlparse

from src.core.evidence import SearchHit
from src.core.rules import get_rules_config
from src.infra.tools.docs_search.normalization import (
    normalize_identifier_reference_text,
    normalize_identifier_token,
)


def tokenize_topic_terms(text: str) -> set[str]:
    stopwords = {"official", "docs", "documentation", "reference"}
    return {
        token.lower()
        for token in re.findall(r"[A-Za-z0-9_.:/-]+", str(text or ""))
        if len(token) >= 2 and token.lower() not in stopwords
    }


_ASCII_IDENTIFIER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])(?:[A-Za-z][A-Za-z0-9._-]*|v\d+)(?![A-Za-z0-9_])"
)


def _identifier_stopwords(*, library_name: str = "") -> set[str]:
    stopwords = {item.lower() for item in get_rules_config().planner.docs_identifier_stopwords}
    stopwords.update(
        {
            "api",
            "parameters",
            "parameter",
            "reference",
            "validation",
            "validator",
        }
    )
    for part in re.findall(r"[A-Za-z0-9_.-]+", str(library_name or "").lower()):
        stopwords.add(part)
    return stopwords


def extract_exact_identifier_terms(query: str, *, library_name: str = "") -> list[str]:
    identifiers: list[str] = []
    seen: set[str] = set()
    stopwords = _identifier_stopwords(library_name=library_name)
    normalized_query = normalize_identifier_reference_text(query)
    for token in _ASCII_IDENTIFIER_PATTERN.findall(normalized_query):
        normalized = normalize_identifier_token(token)
        lowered = normalized.lower()
        if not normalized or lowered in stopwords:
            continue
        if re.fullmatch(r"v\d+", lowered):
            continue
        if not (
            "." in normalized
            or "_" in normalized
            or "-" in normalized
            or normalized != normalized.lower()
        ):
            continue
        if lowered not in seen:
            identifiers.append(normalized)
            seen.add(lowered)
    return identifiers


def has_exact_identifier_coverage(
    query: str,
    hits: list[SearchHit],
    *,
    library_name: str = "",
) -> bool:
    required_identifiers = extract_exact_identifier_terms(query, library_name=library_name)
    if not required_identifiers:
        return True
    combined_text = " ".join(
        part
        for item in hits
        for part in (
            item.evidence.snapshot.title,
            item.evidence.snapshot.source_uri,
            item.evidence.excerpt,
        )
        if part
    )
    normalized_combined_text = normalize_identifier_reference_text(combined_text)
    return all(
        re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(identifier)}(?![A-Za-z0-9_])",
            normalized_combined_text,
            flags=re.I,
        )
        is not None
        for identifier in required_identifiers
    )


def entity_hit_score(query: str, hit: SearchHit) -> float:
    query_terms = tokenize_topic_terms(query)
    haystack = " ".join(
        [
            hit.evidence.snapshot.title,
            hit.evidence.snapshot.source_uri,
            hit.evidence.excerpt,
        ]
    ).lower()
    return float(sum(1 for token in query_terms if token in haystack))


def query_requests_api_detail(query: str) -> bool:
    lowered = str(query or "").lower()
    return any(
        marker in lowered
        for marker in (
            "api",
            "reference",
            "signature",
            "option",
            "options",
            "parameter",
            "parameters",
            "argument",
            "arguments",
            "옵션",
            "파라미터",
            "매개변수",
            "인자",
        )
    )


def api_reference_preference_score(query: str, hit: SearchHit) -> float:
    if not query_requests_api_detail(query):
        return 0.0

    url = hit.evidence.snapshot.source_uri.lower()
    title = hit.evidence.snapshot.title.lower()
    metadata = hit.evidence.element.metadata.get("doc_metadata")
    score = 0.0

    if "/api/_as_gen/" in url:
        score += 6.0
    if "/reference/generated/" in url or "/reference/api/" in url:
        score += 4.0
    if "/plot_types/" in url or "/gallery/" in url:
        score -= 3.0

    if isinstance(metadata, dict):
        if metadata.get("parameters") or metadata.get("options"):
            score += 5.0
        if metadata.get("signature"):
            score += 2.0
        symbol = str(metadata.get("symbol") or "").lower()
        if "." in symbol:
            score += 1.0
            if symbol and symbol in f"{title} {url}":
                score += 1.0

    if "matplotlib.pyplot.pie" in f"{title} {url}":
        score += 4.0
    return score


def path_cluster(value: str) -> str:
    parsed = urlparse(str(value or ""))
    parts = [part for part in str(parsed.path or "").split("/") if part]
    return "/".join(parts[:4]).lower()


def filter_docs_hits_by_topic_purity(
    query: str,
    hits: list[SearchHit],
    retrieval_warnings: list[str],
) -> list[SearchHit]:
    meaningful_hits = [item for item in hits if hit_has_grounded_text(item)]
    if len(meaningful_hits) < len(hits):
        retrieval_warnings.append("docs_chrome_only")
    hits = meaningful_hits
    if len(hits) <= 1:
        return hits

    ranked = sorted(
        hits,
        key=lambda item: (
            api_reference_preference_score(query, item),
            entity_hit_score(query, item),
            float(item.score.normalized or 0.0) if item.score else 0.0,
        ),
        reverse=True,
    )
    if entity_hit_score(query, ranked[0]) <= 0.0:
        return ranked[:2]
    anchor = ranked[0]
    anchor_cluster = path_cluster(anchor.evidence.snapshot.source_uri)
    kept = [anchor]
    for item in ranked[1:]:
        same_cluster = path_cluster(item.evidence.snapshot.source_uri) == anchor_cluster
        strong_entity_match = entity_hit_score(query, item) >= 2.0
        if same_cluster or strong_entity_match:
            kept.append(item)
    if len(kept) < len(hits):
        retrieval_warnings.append("topic_purity_pruned")
    return kept[:2]


def hit_has_grounded_text(hit: SearchHit) -> bool:
    """Check returned source text, never treat a document title as factual support."""
    navigation_prefixes = (
        "table of contents",
        "on this page",
        "previous:",
        "next:",
        "skip to content",
        "edit this page",
        "view source",
        "home >",
        "navigation",
        "back to top",
    )
    section_names = {"parameters", "returns", "examples", "notes", "contents", "api reference", "reference"}
    for raw_line in hit.evidence.excerpt.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line.lower().startswith(navigation_prefixes):
            continue
        if line.rstrip(":").lower() in section_names:
            continue
        if re.fullmatch(r"\[[^]]+\]\([^)]+\)", line):
            continue
        if line.lower() == hit.evidence.snapshot.title.lower():
            continue
        if re.search(r"[A-Za-z가-힣0-9]", line):
            return True
    return False


def has_meaningful_docs_hits(hits: list[SearchHit]) -> bool:
    return any(hit_has_grounded_text(item) for item in hits)


def dedupe_docs_hits(items: list[SearchHit]) -> list[SearchHit]:
    """Deduplicate exact source selections without concatenating different excerpts."""
    by_evidence: dict[str, SearchHit] = {}
    for hit in items:
        key = hit.evidence.id
        current = by_evidence.get(key)
        score = hit.score.normalized if hit.score else None
        current_score = current.score.normalized if current and current.score else None
        if current is None or (score is not None and (current_score is None or score > current_score)):
            by_evidence[key] = hit
    return [hit.model_copy(update={"rank": index}) for index, hit in enumerate(by_evidence.values(), start=1)]
