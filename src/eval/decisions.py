"""The sole source of current case and run release decisions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from .tool_policy import ToolPolicyAssessment, ToolPolicySpec, assess_execution_policy
from .action_outcomes import ActionAssessment, assess_action_outcomes

if TYPE_CHECKING:
    from .config_models import BenchmarkCase
    from .result_models import CaseResult, SaveAssessment
    from .summary_models import GateResult, RunSummary


DECISION_CONTRACT_VERSION = 3
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
    assessment, actions = _assess_case_outcomes(result)
    result.action_assessments = actions
    if any(item.status == "failed" and item.outcome != "must_not_execute"
           for item in result.action_assessments.values()):
        result.product_pass = False
    outcome_assessments = [
        *result.action_assessments.values(),
        *([result.save_assessment] if result.save_assessment is not None else []),
        *result.setup_save_assessments.values(),
    ]
    if any(item.status == "unverifiable" for item in outcome_assessments):
        # Unobserved actions cannot erase an already settled product failure.
        # Apply this once for initial results, run-end rechecks, and reloads.
        if result.eval_validity == "valid" and result.judge_status_reason != "product_failure":
            result.eval_validity = "incomplete"
    decision = _decide_case(result, assessment, actions)
    result.policy_assessment = assessment
    result.decision = decision
    result.release_pass = decision.passed
    result.gate_failures = decision.failure_codes
    return decision


def _assess_case_outcomes(result: CaseResult) -> tuple[ToolPolicyAssessment, dict[str, ActionAssessment]]:
    """Evaluate retained observations before deriving the case verdict."""
    assessment = assess_execution_policy(policy=result.policy_snapshot, evidence=result.execution_evidence,
                                         execution_id=result.execution_id, prior_turns=result.scenario_turns,
                                         actions=result.actions, retrieval_diagnostics=result.retrieval_diagnostics,
                                         debug=result.debug, tool_calls=result.tool_calls,
                                         tool_call_count=result.tool_call_count)
    return assessment, assess_action_outcomes(
        policy=result.policy_snapshot, execution_evidence=result.execution_evidence,
        response=result.response, actions=result.actions, save_assessment=result.save_assessment,
        policy_assessment=assessment,
    )


def _decide_case(
    result: CaseResult, assessment: ToolPolicyAssessment, actions: dict[str, ActionAssessment],
) -> CaseDecision:
    if result.judge_status_reason == "product_failure" and not product_failure_observed(
        runtime_errors=result.runtime_errors, response_errors=result.response_errors,
        policy_assessment=assessment, actions=actions,
        save_assessments=[*([result.save_assessment] if result.save_assessment is not None else []),
                          *result.setup_save_assessments.values()],
    ):
        raise ValueError("a product_failure judge skip requires an observed product failure")
    # Input budget/schema failures leave semantic quality unobserved.
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
                   "forbidden_tool_execution",
               }]
    reasons.extend(assessment.failure_codes)
    reasons.extend(code for item in actions.values() for code in item.failure_codes)
    if result.runtime_errors:
        reasons.append("runtime_error")
    if result.response_errors:
        reasons.append("response_contract_error")
    if result.eval_validity != "valid":
        reasons.append("invalid_eval" if result.eval_validity == "invalid" else "incomplete_eval")
    if preflight_quality_unobserved:
        reasons.append("product_quality_unobserved")
    elif (result.product_pass is not True and result.judge_status_reason != "product_failure"
          and not any(item.status == "failed" for item in actions.values())):
        reasons.append("product_quality_below_floor")
    if result.judge_status_reason != "product_failure" and (result.judge_pass is not True or result.judge_status != "succeeded"):
        reasons.append("judge_not_passed")
    for item in [*([result.save_assessment] if result.save_assessment is not None else []),
                 *result.setup_save_assessments.values()]:
        if item.passed is False:
            reasons.extend(item.failure_codes)
    reasons = sorted(set(reasons), key=lambda code: (code != "forbidden_tool_execution", code))
    return CaseDecision(passed=not reasons, failure_codes=reasons)


def product_failure_observed(
    *, runtime_errors: list[str], response_errors: list[str], policy_assessment: ToolPolicyAssessment,
    actions: dict[str, ActionAssessment], save_assessments: list[SaveAssessment],
) -> bool:
    """Only observed failures settle a verdict before semantic evaluation."""
    return bool(
        runtime_errors or response_errors or policy_assessment.violations
        or any(item.status == "failed" for item in [*save_assessments, *actions.values()])
    )


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
        assessment, actions = _assess_case_outcomes(result)
        decision = _decide_case(result, assessment, actions)
        if (result.decision != decision or result.release_pass != decision.passed
                or result.policy_assessment != assessment
                or result.action_assessments != actions):
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
    skipped = len({result.case_id for result in results if result.case_id in planned_ids
                   and result.judge_status == "not_run" and result.judge_status_reason == "product_failure"})
    required = len(planned_ids) - skipped if summary.judge_enabled else 0
    succeeded = sum(result.judge_status == "succeeded" for result in results)
    judge_counts = {
        "judge_skipped_product_failure_cases": skipped, "judge_required_cases": required,
        "judge_succeeded_cases": succeeded,
        "judge_failed_cases": sum(result.judge_status == "failed" for result in results),
        "judge_disabled_cases": sum(result.judge_status == "disabled" for result in results),
        "judge_not_run_cases": sum(result.judge_status == "not_run" for result in results),
        "judge_execution_rate": round(succeeded / required, 4) if required else None,
    }
    if any(getattr(summary.metrics, field) != value for field, value in judge_counts.items()):
        raise ValueError("run judge counts disagree with observed judge eligibility")
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
