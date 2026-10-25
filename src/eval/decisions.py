"""The sole source of current case and run release decisions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from .tool_policy import ToolPolicyAssessment, ToolPolicySpec, assess_execution_policy
from .action_outcomes import assess_action_outcomes

if TYPE_CHECKING:
    from .config_models import BenchmarkCase
    from .result_models import CaseResult
    from .summary_models import GateResult, RunSummary


DECISION_CONTRACT_VERSION = 2
REQUIRED_RELEASE_GATES = frozenset({
    "tool_execution_policy", "save_outcome_contract", "evaluation_completeness", "release_pass_rate",
    "tool_precision", "tool_recall", "citation_compliance", "p95_latency_ms", "avg_cost_per_case_usd",
    "action_outcome_contract", "release_suite_completeness", "cost_observation_rate",
})


class CaseDecision(BaseModel):
    passed: bool
    failure_codes: list[str] = Field(default_factory=list)


class ReleaseDecision(CaseDecision):
    scope: Literal["release", "diagnostic"]


def policy_for_case(case: BenchmarkCase) -> ToolPolicySpec:
    outcomes = {
        tool: "required_success" for tool in case.expected_tools
        if tool in {"save_text", "slack_notify"}
    }
    outcomes.update({
        tool: "must_not_execute" for tool in case.forbidden_tools
        if tool in {"save_text", "slack_notify"}
    })
    if case.save_expectation is not None:
        outcomes["save_text"] = case.save_expectation.outcome
    return ToolPolicySpec(final_forbidden_tools=list(case.forbidden_tools), setup_turn_count=len(case.setup_turns),
                          setup_forbidden_tools=case.setup_forbidden_tools, final_action_outcomes=outcomes)


def refresh_case_decision(result: CaseResult, case: BenchmarkCase | None = None) -> CaseDecision:
    """Recompute from observations and assessments, never from a stored PASS."""
    if case is not None and result.decision_contract_version == DECISION_CONTRACT_VERSION:
        if result.policy_snapshot is not None and result.policy_snapshot != policy_for_case(case):
            raise ValueError("case policy does not match the retained execution policy snapshot")
    assessment, decision = evaluate_case_decision(result)
    result.action_assessments = assess_action_outcomes(result, assessment)
    if any(item.status == "failed" and item.outcome != "must_not_execute"
           for item in result.action_assessments.values()):
        result.product_pass = False
        assessment, decision = evaluate_case_decision(result)
    outcome_assessments = [
        *result.action_assessments.values(),
        *([result.save_assessment] if result.save_assessment is not None else []),
        *result.setup_save_assessments.values(),
    ]
    if any(item.status == "unverifiable" for item in outcome_assessments):
        # Unobserved actions cannot erase an already settled product failure.
        # Apply this once for initial results, run-end rechecks, and reloads.
        if result.eval_validity == "valid" and result.judge_status_reason != "missing_final_response":
            result.eval_validity = "incomplete"
            assessment, decision = evaluate_case_decision(result)
    result.policy_assessment = assessment
    result.decision = decision
    result.release_pass = result.passed = decision.passed
    result.gate_failures = decision.failure_codes
    result.final_score = result.composite_quality_score
    result.judge_gate_passed = result.judge_pass
    return decision


def evaluate_case_decision(result: CaseResult) -> tuple[ToolPolicyAssessment, CaseDecision]:
    """Evaluate retained decision evidence without changing stored measurements."""
    assessment = assess_execution_policy(policy=result.policy_snapshot, evidence=result.execution_evidence,
                                         request_id=result.request_id, prior_turns=result.scenario_turns,
                                         actions=result.actions, retrieval_diagnostics=result.retrieval_diagnostics,
                                         debug=result.debug, tool_calls=result.tool_calls,
                                         tool_call_count=result.tool_call_count)
    # New preflight failures have no quality observation. Keep prior reason
    # semantics intact so existing decision-contract records remain readable.
    preflight_quality_unobserved = (
        result.product_pass is None and result.judge_status == "not_run"
        and result.judge_status_reason in {"input_budget_exceeded", "input_invalid"}
    )
    quality_reasons = (
        {"product_quality_below_floor", "product_quality_unobserved"}
        if preflight_quality_unobserved else set()
    )
    reasons = [code for code in result.gate_failures
               if code not in quality_reasons
               and not code.startswith(("tool_execution_", "tool_policy_", "action_")) and code not in {
                   "forbidden_tool_execution", "tool_policy_snapshot_missing", "legacy_result_unverified",
               }]
    reasons.extend(assessment.failure_codes)
    actions = assess_action_outcomes(result, assessment)
    reasons.extend(code for item in actions.values() for code in item.failure_codes)
    if result.decision_contract_version != DECISION_CONTRACT_VERSION:
        reasons.append("legacy_result_unverified")
    if result.runtime_errors:
        reasons.append("runtime_error")
    if result.response_errors:
        reasons.append("response_contract_error")
    if result.eval_validity != "valid":
        reasons.append("invalid_eval" if result.eval_validity == "invalid" else "incomplete_eval")
    if preflight_quality_unobserved:
        reasons.append("product_quality_unobserved")
    elif result.product_pass is not True and not any(item.status == "failed" for item in actions.values()):
        reasons.append("product_quality_below_floor")
    if result.judge_pass is not True or result.judge_status != "succeeded":
        reasons.append("judge_not_passed")
    for item in [*([result.save_assessment] if result.save_assessment is not None else []),
                 *result.setup_save_assessments.values()]:
        if item.passed is False:
            reasons.extend(item.failure_codes)
    reasons = sorted(set(reasons), key=lambda code: (code != "forbidden_tool_execution", code))
    return assessment, CaseDecision(passed=not reasons, failure_codes=reasons)


def decide_release(*, gates: list[GateResult], track: str, policy_failure_codes: list[str]) -> ReleaseDecision:
    reasons = list(policy_failure_codes)
    for name in sorted(REQUIRED_RELEASE_GATES):
        matched = [gate for gate in gates if gate.name == name]
        if len(matched) != 1 or matched[0].gate_type != "release":
            reasons.append("release_gate_contract_invalid")
    reasons.extend(gate.name for gate in gates if gate.gate_type == "release" and not gate.passed)
    if track != "release":
        reasons.append("diagnostic_run")
    return ReleaseDecision(passed=not reasons, failure_codes=list(dict.fromkeys(reasons)),
                           scope="release" if track == "release" else "diagnostic")


def results_fingerprint(results: list[CaseResult]) -> str:
    payload = [result.model_dump(mode="json") for result in results]
    return raw_results_fingerprint(payload)


def raw_results_fingerprint(payload: list[dict[str, Any]]) -> str:
    """Hash the saved JSON values before any model defaults or coercion apply."""
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_run_outputs(summary: RunSummary, results: list[CaseResult]) -> None:
    """Validate persisted current outputs without external effects or an LLM."""
    from .reporting.summary import MEASUREMENT_CONTRACT_VERSION

    if summary.measurement_contract_version != MEASUREMENT_CONTRACT_VERSION:
        raise ValueError("historical results cannot establish current release eligibility (measurement contract)")
    validate_run_decision_evidence(summary, results)
    if summary.results_fingerprint != results_fingerprint(results):
        raise ValueError("run results fingerprint mismatch")


def validate_run_decision_evidence(summary: RunSummary, results: list[CaseResult]) -> None:
    """Validate the shared decision contract independently of measurement formats."""
    if summary.decision_contract_version != DECISION_CONTRACT_VERSION:
        raise ValueError("historical results cannot establish current release eligibility")
    # model_copy/direct assignment do not run Pydantic validators.
    type(summary).model_validate(summary.model_dump(mode="json"))
    planned_ids = set(summary.case_policy_snapshots)
    result_ids = [result.case_id for result in results]
    expected_counts = {
        "planned_cases": len(planned_ids), "total_cases": len(results),
        "missing_result_cases": len(planned_ids.difference(result_ids)),
        "unexpected_result_cases": len(set(result_ids).difference(planned_ids)),
        "duplicate_result_cases": len(result_ids) - len(set(result_ids)),
    }
    if any(getattr(summary.metrics, field) != value for field, value in expected_counts.items()):
        raise ValueError("run completeness does not match planned and observed case identities")
    for result in results:
        if result.run_id != summary.run_id:
            raise ValueError("case result belongs to a different run")
        if result.decision_contract_version != DECISION_CONTRACT_VERSION:
            raise ValueError("current run contains historical results")
        snapshot = summary.case_policy_snapshots.get(result.case_id)
        if snapshot is not None and result.policy_snapshot != snapshot:
            raise ValueError("case policy does not match the run policy snapshot")
        assessment, decision = evaluate_case_decision(result)
        if (result.decision != decision or result.release_pass != decision.passed
                or result.policy_assessment != assessment
                or result.action_assessments != assess_action_outcomes(result, assessment)):
            raise ValueError("stored case verdict disagrees with execution policy")
    expected = decide_release(gates=summary.gates, track=summary.track,
                              policy_failure_codes=summary.metrics.policy_failure_codes)
    if summary.release_decision != expected or summary.overall_passed != expected.passed:
        raise ValueError("stored release verdict disagrees with release gates")
    codes = list(dict.fromkeys(code for result in results for code in result.policy_assessment.failure_codes))
    if codes != summary.metrics.policy_failure_codes:
        raise ValueError("run policy failures disagree with case policy assessments")
    policy_counts = {
        "policy_compliant_cases": sum(result.policy_assessment.status == "compliant" for result in results),
        "policy_violating_cases": sum(result.policy_assessment.status == "violated" for result in results),
        "policy_indeterminate_cases": sum(result.policy_assessment.status == "indeterminate" for result in results),
        "policy_violations": sum(len(result.policy_assessment.violations) for result in results),
    }
    if any(getattr(summary.metrics, field) != value for field, value in policy_counts.items()):
        raise ValueError("run policy counts disagree with case policy assessments")
    passed = sum(result.decision.passed for result in results)
    rate = round(passed / len(planned_ids), 4) if planned_ids else 0.0
    if summary.metrics.release_passed_cases != passed or summary.metrics.release_pass_rate != rate:
        raise ValueError("run pass rate disagrees with case decisions")
    from .reporting.summary import cost_observation_metrics

    costs = cost_observation_metrics(results=results, planned_ids=planned_ids)
    if any(getattr(summary.metrics, field) != value for field, value in costs.items()):
        raise ValueError("run cost metrics disagree with raw cost observations")
    action_counts = {
        "action_contract_failures": sum(
            any(not item.passed for item in result.action_assessments.values()) for result in results
        ),
        "action_unverifiable_cases": sum(
            any(item.status == "unverifiable" for item in result.action_assessments.values()) for result in results
        ),
    }
    if any(getattr(summary.metrics, field) != value for field, value in action_counts.items()):
        raise ValueError("run action counts disagree with case action assessments")
