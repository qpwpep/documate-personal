"""Read retained v3/v4 records without adding aliases to current models.

The reader verifies original bytes first. These projections retain the saved
observations and verdicts; they never reprice or promote historical runs.
"""
from __future__ import annotations

from copy import deepcopy


def project_case_result(payload: dict[str, object]) -> dict[str, object]:
    projected = {
        key: value for key, value in payload.items()
        if key not in {"judge_gate_passed", "judge_score_total", "invalid_eval", "upload_fixture"}
    }
    projected = deepcopy(projected)
    legacy_latency = projected.pop("latency_ms_e2e", None)
    if projected.get("question_response_ms") is None:
        projected["question_response_ms"] = legacy_latency
    _project_execution_turn(projected)
    turns = projected.get("scenario_turns")
    if isinstance(turns, list):
        for turn in turns:
            if isinstance(turn, dict):
                _project_execution_turn(turn)
    policy = projected.get("policy_assessment")
    if isinstance(policy, dict):
        for field in ("violations", "evidence_issues"):
            issues = policy.get(field)
            if isinstance(issues, list):
                for issue in issues:
                    _rename_execution_id(issue)
    return projected


def _project_execution_turn(payload: dict[str, object]) -> None:
    _rename_execution_id(payload)
    _rename_execution_id(payload.get("turn_result"))
    _rename_execution_id(payload.get("execution_evidence"))
    debug = payload.get("debug")
    if isinstance(debug, dict):
        _rename_execution_id(debug.get("execution_evidence"))


def _rename_execution_id(payload: object) -> None:
    if isinstance(payload, dict) and "request_id" in payload:
        payload["execution_id"] = payload.pop("request_id")


def project_run_summary(payload: dict[str, object]) -> dict[str, object]:
    projected = deepcopy(payload)
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("historical summary metrics must be an object")
    projected_metrics = {
        key: value for key, value in metrics.items()
        if key not in {"passed_cases", "pass_rate", "judge_legacy_unknown_cases", "deterministic_direct_usage_rate"}
    }
    if "request_id_coverage_rate" in projected_metrics:
        projected_metrics["execution_id_coverage_rate"] = projected_metrics.pop("request_id_coverage_rate")
    projected["metrics"] = projected_metrics
    audit = projected.get("audit_metrics")
    if isinstance(audit, dict) and "request_id_coverage_rate" in audit:
        audit["execution_id_coverage_rate"] = audit.pop("request_id_coverage_rate")
    weights = projected.pop("weights", None)
    if projected.get("weight_profiles") is None:
        if not isinstance(weights, dict):
            raise ValueError("historical summary requires retained weights")
        projected["weight_profiles"] = {"historical": weights}
    return projected
