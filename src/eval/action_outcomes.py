"""Mandatory action outcomes, independent of quality scores and live options."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts.tool_execution import ToolExecutionEvidence
from .tool_policy import ToolPolicyAssessment, executed_tool_names


class ActionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: Literal["save_text", "slack_notify"]
    outcome: Literal["required_success", "expected_failure", "must_not_execute"]
    status: Literal["verified", "failed", "unverifiable"]
    failure_codes: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "verified"


def assess_action_outcomes(result: Any, policy_assessment: ToolPolicyAssessment) -> dict[str, ActionAssessment]:
    """Recompute from retained receipts, lifecycle facts, and byte verification."""
    obligations = result.policy_snapshot.final_action_outcomes if result.policy_snapshot is not None else None
    assessments = {}
    for tool, outcome in (obligations or {}).items():
        status = "unverifiable"
        if outcome == "must_not_execute":
            executed = executed_tool_names(result.execution_evidence)
            if executed is not None:
                status = "failed" if tool in executed else "verified"
        elif tool == "save_text":
            saved = result.save_assessment
            if saved is not None and saved.outcome == outcome and saved.status != "not_applicable":
                status = saved.status
        else:
            receipts = [receipt for receipt in result.actions if receipt.kind == tool]
            if not receipts and outcome == "required_success" and policy_assessment.status == "compliant" and result.response is not None:
                # A complete, consistent request history can prove that the
                # required action was omitted. Missing/incomplete histories or
                # an attempted action without its receipt remain unverified.
                evidence = ToolExecutionEvidence.model_validate(result.execution_evidence)
                if not any(event.tool_name == tool for event in evidence.events):
                    status = "failed"
            elif len(receipts) == 1:
                receipt = receipts[0]
                if outcome == "required_success":
                    if receipt.status == "success" and receipt.slack is not None and receipt.slack.status == "sent":
                        status = "verified"
                    elif receipt.status in {"error", "skipped"}:
                        status = "failed"
                # A future expected-failure obligation needs an explicit error
                # oracle. An arbitrary failed send must never imply success.
        if status == "verified" and policy_assessment.status != "compliant":
            status = "unverifiable"
        assessments[tool] = ActionAssessment(
            tool_name=tool, outcome=outcome, status=status,
            failure_codes=[] if status == "verified" else [f"action_{tool}_{status}"],
        )
    return assessments
