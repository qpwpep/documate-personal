"""Current weight inputs and saved scoring facts have separate contracts."""

import argparse
from copy import deepcopy
from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.history_loader import StoredRun, load_history_runs, select_comparable_runs
from src.eval.reporting.summary import build_summary
from src.eval.reporting.writer import load_report_inputs, write_run_outputs
from src.eval.result_models import CaseResult


# Captured with the pre-change scorer, including the original result fingerprint.
V4_RUN = Path(__file__).parents[1] / "fixtures" / "eval" / "weights_v4_run.json"


def _saved_run(path: Path, *, legacy_weight_keys: bool = False):
    payload = json.loads(V4_RUN.read_text(encoding="utf-8"))
    if legacy_weight_keys:
        payload["summary"]["weights"] = {
            "tool_match": 0.3, "content_constraints": 0.25,
            "citation_compliance": 0.2, "safety_format": 0.05, "llm_judge": 0.2,
        }
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(json.dumps(payload["summary"], ensure_ascii=False), encoding="utf-8")
    (path / "raw_results.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in payload["results"]), encoding="utf-8",
    )
    return payload


def _case(**changes):
    return BenchmarkCase.model_validate({
        "case_id": "case-0", "category": "docs_only",
        "query": "pandas merge는 어떻게 동작해?", "expected_tools": ["tavily_search"],
        "forbidden_tools": ["slack_notify"], "require_official_citation": True,
        **changes,
    })


def _new_run(*, config=None, case=None):
    previous = json.loads(V4_RUN.read_text(encoding="utf-8"))
    results = [CaseResult.model_validate(row) for row in previous["results"]]
    summary = build_summary(
        run_id="policy-run", endpoint="http://fixture", fixtures_path="cases.jsonl",
        config_path="config.toml", track="release", requested_limit=None,
        config=config or BenchmarkConfig(), cases=[case or _case()], results=results,
    )
    return summary, results


def _stored(summary):
    return StoredRun(summary, datetime.fromisoformat(summary.generated_at_utc), track_explicit=True)


def test_new_summary_records_raw_profiles_and_omits_legacy_weights(tmp_path):
    summary, results = _new_run()
    write_run_outputs(output_dir=tmp_path, summary=summary, results=results)

    saved = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    profiles = saved["weight_profiles"]
    assert set(profiles) == {"general", "action_with_citations", "action_without_citations"}
    assert profiles["general"]["reference_coverage"] == 0.2
    assert profiles["action_with_citations"]["reference_coverage"] == 0.1
    # Snapshot the raw profile; normalization happens only after a case override.
    assert profiles["action_without_citations"]["answer_quality"] == 0.4
    assert sum(profiles["action_without_citations"].values()) == pytest.approx(0.95)
    assert "weights" not in saved
    restored = load_report_inputs(tmp_path)
    assert restored.summary.weight_profiles == profiles
    assert restored.summary.weights is None
    assert restored.current_results[0].effective_weights == results[0].effective_weights


@pytest.mark.parametrize("legacy_weight_keys", [False, True])
def test_previous_scoring_results_keep_weights_scores_fingerprint_and_bytes(tmp_path, legacy_weight_keys):
    saved = _saved_run(tmp_path, legacy_weight_keys=legacy_weight_keys)
    original_bytes = {name: (tmp_path / name).read_bytes() for name in ("summary.json", "raw_results.jsonl")}

    inputs = load_report_inputs(tmp_path)

    assert inputs.summary.weights == saved["summary"]["weights"]
    assert inputs.summary.weight_profiles is None
    assert inputs.summary.audit_metrics["scoring_contract_version"] == "execution-policy-contract-v4"
    assert inputs.summary.results_fingerprint == saved["summary"]["results_fingerprint"]
    assert inputs.current_results[0].model_dump(mode="json") == saved["results"][0]
    from src.eval.main import command_report
    assert command_report(argparse.Namespace(run=tmp_path)) == 0
    assert {name: (tmp_path / name).read_bytes() for name in original_bytes} == original_bytes


def test_previous_scoring_raw_result_tampering_is_still_rejected(tmp_path):
    saved = _saved_run(tmp_path)
    saved["results"][0]["effective_weights"]["llm_judge"] = 0.0
    (tmp_path / "raw_results.jsonl").write_text(json.dumps(saved["results"][0]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="fingerprint"):
        load_report_inputs(tmp_path)


def test_saved_effective_weights_are_read_without_current_input_alias_conversion(tmp_path):
    saved = _saved_run(tmp_path)
    legacy_weights = {
        "tool_match": 0.3, "content_constraints": 0.25,
        "citation_compliance": 0.2, "safety_format": 0.05, "llm_judge": 0.2,
    }
    saved["results"][0]["effective_weights"] = legacy_weights
    canonical = json.dumps(saved["results"], ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    saved["summary"]["results_fingerprint"] = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    (tmp_path / "summary.json").write_text(json.dumps(saved["summary"]), encoding="utf-8")
    (tmp_path / "raw_results.jsonl").write_text(json.dumps(saved["results"][0]) + "\n", encoding="utf-8")
    original_bytes = {name: (tmp_path / name).read_bytes() for name in ("summary.json", "raw_results.jsonl")}

    inputs = load_report_inputs(tmp_path)

    assert inputs.current_results[0].effective_weights == legacy_weights
    assert inputs.current_results[0].composite_quality_score == saved["results"][0]["composite_quality_score"]
    assert {name: (tmp_path / name).read_bytes() for name in original_bytes} == original_bytes


def test_new_scoring_separates_history_without_relabeling_prior_runs(tmp_path):
    saved = _saved_run(tmp_path / "previous")
    historical = load_history_runs(tmp_path)[0]
    current, _ = _new_run()
    current = current.model_copy(update={"run_id": "new-scoring"})
    latest, comparable = select_comparable_runs(
        [historical, _stored(current)], track="release", latest_run_id="new-scoring",
    )

    assert latest.summary.audit_metrics["scoring_contract_version"] == "weight-profiles-v5"
    assert comparable == [latest]
    assert current.execution_contract_version == historical.summary.execution_contract_version
    assert current.measurement_contract_version == historical.summary.measurement_contract_version
    assert current.decision_contract_version == historical.summary.decision_contract_version
    assert current.suite_fingerprint == historical.summary.suite_fingerprint
    assert historical.summary.audit_metrics == saved["summary"]["audit_metrics"]
    assert historical.summary.weights == saved["summary"]["weights"]
    assert historical.summary.weight_profiles is None
    assert json.loads((tmp_path / "previous" / "summary.json").read_text(encoding="utf-8")) == saved["summary"]


@pytest.mark.parametrize("profile", ["general", "action_with_citations", "action_without_citations"])
def test_every_configured_profile_changes_the_evaluation_identity(profile):
    config = BenchmarkConfig()
    original, _ = _new_run(config=config)
    changed = config.model_dump(mode="json")
    changed["weights"][profile]["answer_quality"] = 0.6
    updated, _ = _new_run(config=BenchmarkConfig.model_validate(changed))

    assert updated.evaluation_fingerprint != original.evaluation_fingerprint
    assert updated.suite_fingerprint == original.suite_fingerprint


def test_case_override_changes_suite_identity_without_changing_config_identity():
    original, _ = _new_run()
    changed, _ = _new_run(case=_case(weight_override={"llm_judge": 0}))

    assert changed.suite_fingerprint != original.suite_fingerprint
    assert changed.evaluation_fingerprint == original.evaluation_fingerprint


def test_reading_prior_summary_never_injects_current_profile_defaults(tmp_path):
    saved = _saved_run(tmp_path)
    before = deepcopy(saved["summary"])
    inputs = load_report_inputs(tmp_path)

    assert inputs.summary.weight_profiles is None
    assert saved["summary"] == before
    assert "weight_profiles" not in json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
