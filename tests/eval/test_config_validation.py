from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.io import load_config
from src.eval.weighting import compute_rule_weighted_score, resolve_case_weights


def test_release_gate_typo_is_rejected_when_loading_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[hard_gates]\npass_raet = 0.99\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="pass_raet"):
        load_config(config_path)


@pytest.mark.parametrize(("field", "value"), [
    ("pass_rate", -0.1),
    ("pass_rate", 1.1),
    ("tool_precision", 1.1),
    ("tool_recall", -0.1),
    ("citation_compliance", 1.1),
    ("cost_gate_min_observation_rate", 2.0),
    ("pass_rate", float("nan")),
    ("pass_rate", True),
    ("pass_rate", "0.99"),
    ("p95_latency_ms", -1),
    ("p95_latency_ms", 1.5),
    ("p95_latency_ms", True),
    ("avg_cost_per_case_usd", -0.01),
    ("avg_cost_per_case_usd", float("inf")),
    ("avg_cost_per_case_usd", float("nan")),
])
def test_release_gates_reject_invalid_thresholds(field: str, value: object) -> None:
    with pytest.raises(ValidationError, match=field):
        BenchmarkConfig.model_validate({"hard_gates": {field: value}})


@pytest.mark.parametrize("rate", [0, 1])
def test_release_gates_accept_explicit_boundary_thresholds(rate: int) -> None:
    thresholds = {
        "pass_rate": rate,
        "tool_precision": rate,
        "tool_recall": rate,
        "citation_compliance": rate,
        "cost_gate_min_observation_rate": rate,
        "p95_latency_ms": 0,
        "avg_cost_per_case_usd": 0,
    }

    config = BenchmarkConfig.model_validate({"hard_gates": thresholds})

    assert config.hard_gates.model_dump() == thresholds


def test_response_availability_weight_is_applied_to_rule_score() -> None:
    config = BenchmarkConfig()
    case = BenchmarkCase(
        case_id="availability-weight", category="docs_only", query="Explain the API.",
        weight_override={
            "answer_quality": 0, "reference_coverage": 0, "citation_traceability": 0,
            "tool_choice": 0, "response_availability": 2, "llm_judge": 0,
        },
    )

    weights = resolve_case_weights(case=case, profiles=config.weights)

    assert weights.values.response_availability == 1.0
    assert compute_rule_weighted_score({"response_availability": 1.0}, weights) == 1.0


@pytest.mark.parametrize("boundary", ["profile", "case_override"])
def test_legacy_format_language_weight_is_rejected(boundary: str) -> None:
    with pytest.raises(ValidationError, match="format_language"):
        if boundary == "profile":
            profile = BenchmarkConfig().weights.general.model_dump()
            profile["format_language"] = profile.pop("response_availability")
            BenchmarkConfig.model_validate({"weights": {"general": profile}})
        else:
            BenchmarkCase.model_validate({
                "case_id": "old-weight", "category": "docs_only", "query": "Explain the API.",
                "weight_override": {"format_language": 0.05},
            })
