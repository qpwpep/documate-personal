from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval.config_models import BenchmarkConfig
from src.eval.io import load_config


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
