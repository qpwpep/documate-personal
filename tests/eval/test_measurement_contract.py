from datetime import datetime
import hashlib
import json
from pathlib import Path

import pytest

from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.history_loader import StoredRun, select_comparable_runs
from src.eval.readme_renderer import build_history_readme_block
from src.eval.reporting import build_markdown_report, build_summary
from src.eval.result_models import CaseResult
from src.eval.summary_models import RunSummary
from src.eval.svg_renderer import build_history_svg


def _case() -> BenchmarkCase:
    return BenchmarkCase(case_id="timing", category="docs_only", query="Explain arrays")


def _result(**updates) -> CaseResult:
    return CaseResult.model_validate({
        "run_id": "run-timing", "case_id": "timing", "category": "docs_only",
        "query": "Explain arrays", "session_id": "session-timing",
        "endpoint": "http://localhost:8000/agent/stream", "request_payload": {},
        "http_status": 200, "created_at_utc": "2026-09-13T00:00:00+00:00",
        "latency_ms_e2e": 700, "question_response_ms": 700,
        "attachment_setup_ms": 300, "scenario_total_ms": 1800,
        **updates,
    })


def _summary(**updates) -> RunSummary:
    return build_summary(**{
        "run_id": "run-timing", "endpoint": "http://localhost:8000",
        "fixtures_path": "cases.jsonl", "config_path": "config.toml",
        "track": "release", "requested_limit": None,
        "config": BenchmarkConfig(judge_enabled=False), "cases": [_case()],
        "results": [_result()],
        **updates,
    })


def _stored(summary: RunSummary, run_id: str, **updates) -> StoredRun:
    changed = summary.model_copy(update={"run_id": run_id, **updates})
    return StoredRun(changed, datetime.fromisoformat(changed.generated_at_utc), track_explicit=True)


def test_summary_reports_distinct_latency_windows_and_gates_question_time() -> None:
    """Attachment preparation and setup turns cannot masquerade as final-question latency."""
    summary = _summary(results=[_result(), _result(
        attachment_setup_ms=500, question_response_ms=900,
        latency_ms_e2e=900, scenario_total_ms=2200,
    )])

    metrics = summary.metrics
    assert (metrics.p50_attachment_setup_ms, metrics.p95_attachment_setup_ms) == (400.0, 490.0)
    assert (metrics.p50_question_response_ms, metrics.p95_question_response_ms) == (800.0, 890.0)
    assert (metrics.p50_scenario_total_ms, metrics.p95_scenario_total_ms) == (2000.0, 2180.0)
    assert next(gate for gate in summary.gates if gate.name == "p95_latency_ms").actual == 890.0
    report = build_markdown_report(summary)
    assert "attachment_setup_ms" in report
    assert "question_response_ms" in report
    assert "scenario_total_ms" in report
    assert "judge and cleanup" in report


@pytest.mark.parametrize("field", [
    "execution_contract_version", "measurement_contract_version",
    "suite_fingerprint", "evaluation_fingerprint",
])
def test_history_excludes_runs_with_a_different_comparison_contract(field: str) -> None:
    """History deltas require matching execution, measurement, fixture and evaluation contracts."""
    summary = _summary()
    different = _stored(summary, "different", **{field: "changed"})
    compatible = _stored(summary, "compatible")
    latest = _stored(summary, "latest")

    selected, comparable = select_comparable_runs(
        [different, compatible, latest], track="release", latest_run_id="latest",
    )

    assert selected.run_id == "latest"
    assert [run.run_id for run in comparable] == ["compatible", "latest"]


def test_legacy_summary_stays_unversioned_and_cannot_join_a_new_baseline() -> None:
    """Loading old summaries never upgrades their unknown measurement contract."""
    summary = _summary()
    payload = summary.model_dump()
    for field in ("execution_contract_version", "measurement_contract_version", "suite_fingerprint", "evaluation_fingerprint"):
        payload.pop(field, None)
    legacy = RunSummary.model_validate(payload)

    assert legacy.measurement_contract_version is None
    assert legacy.execution_contract_version is None
    _, comparable = select_comparable_runs(
        [_stored(legacy, "legacy"), _stored(summary, "latest")],
        track="release", latest_run_id="latest",
    )
    assert [run.run_id for run in comparable] == ["latest"]


def test_incomplete_new_comparison_contract_does_not_claim_a_shared_baseline() -> None:
    """Matching missing fingerprints do not prove two versioned runs used the same content."""
    summary = _summary().model_copy(update={"suite_fingerprint": None})
    _, comparable = select_comparable_runs(
        [_stored(summary, "older"), _stored(summary, "latest")],
        track="release", latest_run_id="latest",
    )
    assert [run.run_id for run in comparable] == ["latest"]


def test_fingerprints_change_with_case_attachment_evaluation_and_live_options() -> None:
    """Editing content at existing paths or changing judge/live policy starts another baseline."""
    base = _summary(attachment_fingerprints={"input.md": "first"})
    same = _summary(attachment_fingerprints={"input.md": "first"})
    changed_case = _summary(cases=[_case().model_copy(update={"query": "Explain lists"})], attachment_fingerprints={"input.md": "first"})
    changed_attachment = _summary(attachment_fingerprints={"input.md": "second"})
    changed_judge = _summary(config=BenchmarkConfig(judge_enabled=True))
    live = _summary(slack_live_enabled=True)
    changed_destination = _summary(execution_options={"slack": {"channel_id": "C_TEST"}})

    assert base.suite_fingerprint == same.suite_fingerprint
    assert base.evaluation_fingerprint == same.evaluation_fingerprint
    assert base.suite_fingerprint != changed_case.suite_fingerprint
    assert base.suite_fingerprint != changed_attachment.suite_fingerprint
    assert base.evaluation_fingerprint != changed_judge.evaluation_fingerprint
    assert base.evaluation_fingerprint != live.evaluation_fingerprint
    assert base.evaluation_fingerprint != changed_destination.evaluation_fingerprint


def test_provenance_scoring_starts_a_new_baseline_without_relabeling_history() -> None:
    """The new citation provenance rules cannot claim a quality delta against older scoring."""
    config = BenchmarkConfig(judge_enabled=False)
    current = _summary(config=config)
    historical = current.model_dump(mode="json")
    historical["audit_metrics"].pop("scoring_contract_version", None)
    # This is the persisted evaluation identity before scoring was versioned.
    previous_identity = {
        "config": config.model_dump(mode="json"),
        "slack_live_enabled": False,
        "execution_options": {},
    }
    serialized = json.dumps(previous_identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    historical["evaluation_fingerprint"] = "sha256:" + hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    previous = RunSummary.model_validate_json(json.dumps(historical))

    _, comparable = select_comparable_runs(
        [_stored(previous, "previous-scoring"), _stored(current, "current-scoring")],
        track="release", latest_run_id="current-scoring",
    )

    assert [run.run_id for run in comparable] == ["current-scoring"]
    assert current.execution_contract_version == previous.execution_contract_version
    assert current.measurement_contract_version == previous.measurement_contract_version
    assert current.suite_fingerprint == previous.suite_fingerprint
    assert "scoring_contract_version" not in previous.audit_metrics


def test_summary_identifies_the_provenance_scoring_contract() -> None:
    """Persisted results identify the scoring semantics without changing timing contracts."""
    payload = json.loads(_summary().model_dump_json())

    assert payload["audit_metrics"]["scoring_contract_version"] == "answer-provenance-v1"


def test_history_artifacts_identify_new_timing_baseline(tmp_path: Path) -> None:
    """Published history identifies timing semantics and keeps all three timing windows visible."""
    latest = _stored(_summary(), "latest")
    readme_path = tmp_path / "README.md"
    readme_path.write_text("# Demo", encoding="utf-8")
    readme = build_history_readme_block(
        track="release", latest=latest, comparable_runs=[latest], readme_path=readme_path,
        output_root=tmp_path / "output", svg_path=tmp_path / "history.svg",
    )
    svg = build_history_svg([latest])

    assert latest.summary.measurement_contract_version in readme
    assert "p95_attachment_setup_ms" in readme
    assert "p95_question_response_ms" in readme
    assert "p95_scenario_total_ms" in readme
    assert "p95_attachment_setup_ms" in svg
    assert "p95_scenario_total_ms" in svg


def test_cost_coverage_accepts_observed_setup_llm_before_deterministic_final() -> None:
    """A save-only final turn retains the fully observed LLM cost of its preceding answer."""
    result = _result(scenario_turns=[
        {"query": "Explain arrays", "request_payload": {}, "request_id": "setup", "debug": {
            "model_usage_status": "llm_used", "llm_calls": [{
                "stage": "synthesis", "path": "structured",
                "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
            }],
        }},
        {"query": "Save that answer", "request_payload": {}, "request_id": "final", "debug": {
            "model_usage_status": "deterministic", "llm_calls": [],
        }},
    ], cost_usd=0.001)

    metrics = _summary(results=[result]).metrics

    assert metrics.llm_call_coverage_rate == 1.0
    assert metrics.request_id_coverage_rate == 1.0
    assert metrics.cost_gate_eligible is True


def test_missing_setup_usage_keeps_partial_scenario_cost_out_of_gate() -> None:
    """A well-observed final turn cannot hide unobserved setup usage or request identity."""
    result = _result(scenario_turns=[
        {"query": "Explain arrays", "request_payload": {}, "debug": {
            "model_usage_status": "llm_used", "llm_calls": [{"stage": "synthesis", "path": "structured"}],
        }},
        {"query": "Continue", "request_payload": {}, "request_id": "final", "debug": {
            "model_usage_status": "llm_used", "llm_calls": [{
                "stage": "synthesis", "path": "structured",
                "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
            }],
        }},
    ], llm_calls=[{
        "stage": "synthesis", "path": "structured",
        "usage_metadata": {"input_tokens": 10, "output_tokens": 5},
    }], request_id="final", cost_usd=0.001)

    metrics = _summary(results=[result]).metrics

    assert metrics.llm_call_coverage_rate == 0.0
    assert metrics.request_id_coverage_rate == 0.0
    assert metrics.cost_gate_eligible is False
