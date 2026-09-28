"""The sole source of current case and run release decisions."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, TYPE_CHECKING

from pydantic import BaseModel, Field

from .tool_policy import ToolPolicySpec, assess_execution_policy

if TYPE_CHECKING:
    from .config_models import BenchmarkCase
    from .result_models import CaseResult
    from .summary_models import GateResult, RunSummary


DECISION_CONTRACT_VERSION = 1
REQUIRED_RELEASE_GATES = frozenset({
    "tool_execution_policy", "save_outcome_contract", "evaluation_completeness", "release_pass_rate",
    "tool_precision", "tool_recall", "citation_compliance", "p95_latency_ms", "avg_cost_per_case_usd",
})


class CaseDecision(BaseModel):
    passed: bool
    failure_codes: list[str] = Field(default_factory=list)


class ReleaseDecision(CaseDecision):
    scope: Literal["release", "diagnostic"]


def policy_for_case(case: BenchmarkCase) -> ToolPolicySpec:
    return ToolPolicySpec(final_forbidden_tools=list(case.forbidden_tools), setup_turn_count=len(case.setup_turns),
                          setup_forbidden_tools=case.setup_forbidden_tools)


def refresh_case_decision(result: CaseResult, case: BenchmarkCase | None = None) -> CaseDecision:
    """Recompute from observations and assessments, never from a stored PASS."""
    if case is not None and result.decision_contract_version == DECISION_CONTRACT_VERSION:
        if result.policy_snapshot is not None and result.policy_snapshot != policy_for_case(case):
            raise ValueError("case policy does not match the retained execution policy snapshot")
    assessment = assess_execution_policy(policy=result.policy_snapshot, evidence=result.execution_evidence,
                                         request_id=result.request_id, prior_turns=result.scenario_turns,
                                         actions=result.actions, retrieval_diagnostics=result.retrieval_diagnostics,
                                         debug=result.debug, tool_calls=result.tool_calls,
                                         tool_call_count=result.tool_call_count)
    result.policy_assessment = assessment
    # Preserve non-policy reasons such as judge quality thresholds. Policy
    # reasons are always derived anew from the retained execution evidence.
    reasons = [code for code in result.gate_failures
               if not code.startswith(("tool_execution_", "tool_policy_")) and code not in {
                   "forbidden_tool_execution", "tool_policy_snapshot_missing", "legacy_result_unverified",
               }]
    reasons.extend(assessment.failure_codes)
    if result.decision_contract_version != DECISION_CONTRACT_VERSION:
        reasons.append("legacy_result_unverified")
    if result.runtime_errors:
        reasons.append("runtime_error")
    if result.response_errors:
        reasons.append("response_contract_error")
    if result.eval_validity != "valid":
        reasons.append("invalid_eval" if result.eval_validity == "invalid" else "incomplete_eval")
    if result.product_pass is not True:
        reasons.append("product_quality_below_floor")
    if result.judge_pass is not True or result.judge_status != "succeeded":
        reasons.append("judge_not_passed")
    for item in [*([result.save_assessment] if result.save_assessment is not None else []),
                 *result.setup_save_assessments.values()]:
        if item.passed is False:
            reasons.extend(item.failure_codes)
    reasons = sorted(set(reasons), key=lambda code: (code != "forbidden_tool_execution", code))
    decision = CaseDecision(passed=not reasons, failure_codes=reasons)
    result.decision = decision
    result.release_pass = result.passed = decision.passed
    result.gate_failures = reasons
    result.final_score = result.composite_quality_score
    result.judge_gate_passed = result.judge_pass
    return decision


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
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
