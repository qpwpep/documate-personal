from __future__ import annotations

import re
from typing import Any

from .schemas import BenchmarkCase, Pricing, ScoreWeights


_OFFICIAL_CITATION_PATTERNS = [
    r"\[◆\s*공식 문서\]",
    r"\[◆\s*official",
    r"https?://",
]
_LOCAL_CITATION_PATTERNS = [
    r"\[◆\s*로컬 예제\]",
    r"\[local example\]",
    r"\.ipynb",
    r"uploads/",
]
_FAILURE_TEXT_PATTERNS = [
    r"Agent 호출 실패",
    r"FastAPI 서버에 연결할 수 없습니다",
    r"요청이 타임아웃되었습니다",
]


def _contains_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, flags=re.I) for pattern in patterns)


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


def score_citation_compliance(case: BenchmarkCase, response_text: str) -> float:
    text = response_text or ""
    required_checks: list[bool] = []

    if case.require_official_citation:
        required_checks.append(_contains_any_pattern(text, _OFFICIAL_CITATION_PATTERNS))
    if case.require_local_citation:
        required_checks.append(_contains_any_pattern(text, _LOCAL_CITATION_PATTERNS))

    if not required_checks:
        return 1.0

    return sum(1 for check in required_checks if check) / len(required_checks)


def score_safety_format(errors: list[str], response_text: str) -> float:
    if errors:
        return 0.0
    text = (response_text or "").strip()
    if not text:
        return 0.0
    if _contains_any_pattern(text, _FAILURE_TEXT_PATTERNS):
        return 0.0
    return 1.0


def compute_rule_scores(
    case: BenchmarkCase,
    response_text: str,
    called_tools: list[str],
    errors: list[str],
) -> dict[str, float]:
    return {
        "tool_match": score_tool_match(case, called_tools),
        "content_constraints": score_content_constraints(case, response_text),
        "citation_compliance": score_citation_compliance(case, response_text),
        "safety_format": score_safety_format(errors=errors, response_text=response_text),
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
        # Keep comparable [0,1] range even when LLM judge is disabled.
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
