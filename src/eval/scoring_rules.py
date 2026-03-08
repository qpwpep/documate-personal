from __future__ import annotations

import math
import re
from typing import Any
from urllib.parse import urlparse

from ..domain_docs import DEFAULT_DOCS
from ..evidence import normalize_source_id
from .schemas import (
    BenchmarkCase,
    CaseWeightOverride,
    EvidenceItem,
    Pricing,
    ScoreWeights,
)


_FAILURE_TEXT_PATTERNS = [
    r"Agent 호출 실패",
    r"FastAPI 서버에 연결할 수 없습니다",
    r"요청이 타임아웃되었습니다",
]


def _contains_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


def _normalize_domain(url_or_domain: str) -> str:
    parsed = urlparse(url_or_domain if "://" in url_or_domain else f"https://{url_or_domain}")
    domain = (parsed.netloc or parsed.path).strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


_ALLOWED_OFFICIAL_DOMAINS = {_normalize_domain(value) for value in DEFAULT_DOCS.values()}


def _is_valid_official_source(url_or_path: str) -> bool:
    parsed = urlparse(str(url_or_path or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    return _normalize_domain(parsed.netloc) in _ALLOWED_OFFICIAL_DOMAINS


def _is_valid_local_source(url_or_path: str) -> bool:
    raw = str(url_or_path or "").strip()
    if not raw:
        return False
    normalized = raw.replace("\\", "/").lower()
    return (
        normalized.endswith(".py")
        or normalized.endswith(".ipynb")
        or "/uploads/" in normalized
        or normalized.startswith("uploads/")
        or normalized.startswith("data/")
    )


def _expected_local_citation_tool(case: BenchmarkCase) -> str:
    return "upload_search" if case.upload_fixture else "rag_search"


def _collect_valid_source_ids(
    evidence: list[EvidenceItem],
    *,
    required_kind: str,
    required_tool: str,
    source_validator: Any,
) -> set[str]:
    valid_ids: set[str] = set()
    for item in evidence:
        if item.kind != required_kind:
            continue
        if item.tool != required_tool:
            continue
        source_id = str(item.source_id or "").strip()
        document_id = str(item.document_id or normalize_source_id(item.url_or_path)).strip()
        if not source_id or not document_id:
            continue
        if document_id != normalize_source_id(item.url_or_path):
            continue
        if not source_validator(item.url_or_path):
            continue
        valid_ids.add(source_id)
    return valid_ids


def resolve_effective_weights(
    *,
    base_weights: ScoreWeights,
    case_override: CaseWeightOverride | None,
) -> tuple[ScoreWeights, str | None]:
    merged = base_weights.as_dict()

    if case_override is not None:
        merged.update(case_override.as_partial_dict())

    for key, value in merged.items():
        if value < 0.0 or not math.isfinite(float(value)):
            return base_weights, f"invalid weight '{key}': {value}"

    total = float(sum(merged.values()))
    if total <= 0.0 or not math.isfinite(total):
        return base_weights, "weight sum must be a positive finite number"

    normalized = {key: float(value) / total for key, value in merged.items()}
    try:
        return ScoreWeights(**normalized), None
    except Exception as exc:
        return base_weights, f"failed to build normalized weights: {exc}"


def score_tool_match(case: BenchmarkCase, called_tools: list[str]) -> float:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)

    if not expected and not forbidden:
        return 1.0

    expected_score = 1.0
    if expected:
        expected_score = len(expected.intersection(called)) / len(expected)

    forbidden_penalty = 0.0
    if forbidden:
        forbidden_penalty = len(forbidden.intersection(called)) / len(forbidden)

    return max(0.0, expected_score * (1.0 - forbidden_penalty))


def score_content_constraints(case: BenchmarkCase, response_text: str) -> float:
    text = response_text or ""
    if not case.must_include and not case.must_not_include:
        return 1.0

    include_score = 1.0
    if case.must_include:
        include_hits = sum(1 for needle in case.must_include if needle.lower() in text.lower())
        include_score = include_hits / len(case.must_include)

    exclude_score = 1.0
    if case.must_not_include:
        exclude_violations = sum(1 for needle in case.must_not_include if needle.lower() in text.lower())
        exclude_score = 1.0 - (exclude_violations / len(case.must_not_include))

    return max(0.0, min(1.0, (include_score + exclude_score) / 2.0))


def score_citation_compliance(
    case: BenchmarkCase,
    response_evidence: list[EvidenceItem],
    observed_evidence: list[EvidenceItem],
    called_tools: list[str],
) -> float:
    checks: list[bool] = []

    if case.require_official_citation:
        response_ids = _collect_valid_source_ids(
            response_evidence,
            required_kind="official",
            required_tool="tavily_search",
            source_validator=_is_valid_official_source,
        )
        observed_ids = _collect_valid_source_ids(
            observed_evidence,
            required_kind="official",
            required_tool="tavily_search",
            source_validator=_is_valid_official_source,
        )
        checks.append(("tavily_search" in called_tools) and bool(response_ids.intersection(observed_ids)))

    if case.require_local_citation:
        expected_local_tool = _expected_local_citation_tool(case)
        response_ids = _collect_valid_source_ids(
            response_evidence,
            required_kind="local",
            required_tool=expected_local_tool,
            source_validator=_is_valid_local_source,
        )
        observed_ids = _collect_valid_source_ids(
            observed_evidence,
            required_kind="local",
            required_tool=expected_local_tool,
            source_validator=_is_valid_local_source,
        )
        checks.append((expected_local_tool in called_tools) and bool(response_ids.intersection(observed_ids)))

    if not checks:
        return 1.0

    return sum(1 for check in checks if check) / len(checks)


def score_safety_format(
    *,
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
    response_text: str,
) -> float:
    if runtime_errors or response_errors or judge_errors:
        return 0.0

    text = (response_text or "").strip()
    if not text:
        return 0.0
    if _contains_any_pattern(text, _FAILURE_TEXT_PATTERNS):
        return 0.0
    return 1.0


def compute_rule_scores(
    *,
    case: BenchmarkCase,
    response_text: str,
    called_tools: list[str],
    response_evidence: list[EvidenceItem],
    observed_evidence: list[EvidenceItem],
    runtime_errors: list[str],
    response_errors: list[str],
    judge_errors: list[str],
) -> dict[str, float]:
    return {
        "tool_match": score_tool_match(case, called_tools),
        "content_constraints": score_content_constraints(case, response_text),
        "citation_compliance": score_citation_compliance(
            case=case,
            response_evidence=response_evidence,
            observed_evidence=observed_evidence,
            called_tools=called_tools,
        ),
        "safety_format": score_safety_format(
            runtime_errors=runtime_errors,
            response_errors=response_errors,
            judge_errors=judge_errors,
            response_text=response_text,
        ),
    }


def compute_rule_weighted_score(
    component_scores: dict[str, float],
    weights: ScoreWeights,
) -> float:
    weight_map = weights.as_dict()
    score = 0.0
    for key, value in component_scores.items():
        score += value * float(weight_map.get(key, 0.0))
    return max(0.0, min(1.0, score))


def compute_final_score(
    rule_weighted_score: float,
    llm_judge_score: float | None,
    weights: ScoreWeights,
) -> float:
    llm_weight = float(weights.llm_judge)
    if llm_judge_score is None:
        denominator = max(1e-9, 1.0 - llm_weight)
        normalized = rule_weighted_score / denominator
        return max(0.0, min(1.0, normalized))
    return max(0.0, min(1.0, rule_weighted_score + llm_judge_score * llm_weight))


def compute_cost_usd(token_usage: Any, pricing: Pricing) -> float | None:
    if token_usage is None:
        return None
    prompt_tokens = int(getattr(token_usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(token_usage, "completion_tokens", 0) or 0)
    prompt_cost = (prompt_tokens / 1000.0) * float(pricing.prompt_per_1k_usd)
    completion_cost = (completion_tokens / 1000.0) * float(pricing.completion_per_1k_usd)
    return round(prompt_cost + completion_cost, 8)


def tool_confusion_counts(case: BenchmarkCase, called_tools: list[str]) -> tuple[int, int, int]:
    expected = set(case.expected_tools)
    forbidden = set(case.forbidden_tools)
    called = set(called_tools)

    tp = len(expected.intersection(called))
    fn = len(expected.difference(called))
    fp = 0
    if forbidden:
        fp += len(forbidden.intersection(called))
    if expected:
        fp += len(called.difference(expected).difference(forbidden))
    return tp, fp, fn
