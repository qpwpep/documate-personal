from __future__ import annotations

from dataclasses import dataclass

from .config_models import BenchmarkCase, ScoreWeights, WeightProfileId, WeightProfiles


@dataclass(frozen=True)
class ResolvedWeights:
    profile_id: WeightProfileId
    values: ScoreWeights


def resolve_case_weights(*, case: BenchmarkCase, profiles: WeightProfiles) -> ResolvedWeights:
    """Choose once, merge relative values, validate, then normalize once."""
    if case.category != "tool_action":
        profile_id: WeightProfileId = "general"
    elif case.require_official_citation or case.require_local_citation:
        profile_id = "action_with_citations"
    else:
        profile_id = "action_without_citations"

    merged = getattr(profiles, profile_id).as_dict()
    merged.update(case.weight_override or {})
    context = f"case {case.case_id!r}, weights.{profile_id}"
    if profile_id == "action_without_citations":
        for axis in ("reference_coverage", "citation_traceability"):
            if merged[axis] != 0:
                raise ValueError(f"{context}.{axis}: must be zero; citation weights cannot be enabled by override")
    try:
        validated = ScoreWeights.model_validate(merged)
    except ValueError as exc:
        raise ValueError(f"{context}: {exc}") from exc
    total = sum(validated.as_dict().values())
    normalized = {key: value / total for key, value in validated.as_dict().items()}
    return ResolvedWeights(profile_id=profile_id, values=ScoreWeights.model_validate(normalized))


def compute_rule_weighted_score(
    component_scores: dict[str, float],
    weights: ResolvedWeights,
) -> float:
    weight_map = weights.values.as_dict()
    score = 0.0
    for key, value in component_scores.items():
        score += value * float(weight_map.get(key, 0.0))
    return max(0.0, min(1.0, score))


def compute_composite_quality_score(
    rule_weighted_score: float,
    llm_judge_score: float | None,
    weights: ResolvedWeights,
) -> float | None:
    """A composite only exists when the required judge score exists.

    Missing or failed judge evaluations return None instead of renormalizing
    rule scores into a manufactured full score.
    """
    if llm_judge_score is None:
        return None
    llm_weight = float(weights.values.llm_judge)
    return max(0.0, min(1.0, rule_weighted_score + llm_judge_score * llm_weight))
