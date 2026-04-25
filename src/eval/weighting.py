from __future__ import annotations

import math

from .config_models import BenchmarkCase, CaseWeightOverride, ScoreWeights


def resolve_effective_weights(
    *,
    case: BenchmarkCase | None = None,
    base_weights: ScoreWeights,
    case_override: CaseWeightOverride | None,
) -> tuple[ScoreWeights, str | None]:
    if (
        case is not None
        and case.category == "tool_action"
        and not case.require_official_citation
        and not case.require_local_citation
    ):
        merged = {
            "answer_quality": 0.25,
            "criteria_coverage": 0.20,
            "groundedness": 0.025,
            "citation_traceability": 0.025,
            "tool_choice": 0.25,
            "uncertainty_handling": 0.10,
            "format_language": 0.05,
            "llm_judge": 0.10,
        }
    else:
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


def resolve_base_weights_for_case(
    *,
    case: BenchmarkCase,
    base_weights: ScoreWeights,
) -> ScoreWeights:
    if case.category != "tool_action":
        return base_weights
    return ScoreWeights(
        answer_quality=0.25,
        criteria_coverage=0.20,
        groundedness=0.08,
        citation_traceability=0.04,
        tool_choice=0.22,
        uncertainty_handling=0.10,
        format_language=0.04,
        llm_judge=0.07,
    )


def compute_rule_weighted_score(
    component_scores: dict[str, float],
    weights: ScoreWeights,
) -> float:
    weight_map = weights.as_dict()
    score = 0.0
    for key, value in component_scores.items():
        score += value * float(weight_map.get(key, 0.0))
    return max(0.0, min(1.0, score))


def compute_composite_quality_score(
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


def compute_final_score(
    rule_weighted_score: float,
    llm_judge_score: float | None,
    weights: ScoreWeights,
) -> float:
    return compute_composite_quality_score(
        rule_weighted_score=rule_weighted_score,
        llm_judge_score=llm_judge_score,
        weights=weights,
    )
