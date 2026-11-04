"""Mandatory action outcomes, independent of quality scores and live options."""
from __future__ import annotations

from typing import Literal, TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from src.core.answer_schema import ActionReceipt, AnswerResponse
from src.core.contracts.tool_execution import ToolExecutionEvidence
from .tool_policy import ToolPolicyAssessment, ToolPolicySpec, executed_tool_names

if TYPE_CHECKING:
    from .result_models import SaveAssessment


class ActionAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tool_name: Literal["save_text", "slack_notify"]
    outcome: Literal["required_success", "expected_failure", "must_not_execute"]
    status: Literal["verified", "failed", "unverifiable"]
    failure_codes: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.status == "verified"


def assess_action_outcomes(
    *, policy: ToolPolicySpec | None, execution_evidence: ToolExecutionEvidence | dict[str, object] | None,
    response: AnswerResponse | None, actions: list[ActionReceipt], save_assessment: SaveAssessment | None,
    policy_assessment: ToolPolicyAssessment,
) -> dict[str, ActionAssessment]:
    """Recompute from retained receipts, lifecycle facts, and byte verification."""
    obligations = policy.final_action_outcomes if policy is not None else None
    assessments = {}
    for tool, outcome in (obligations or {}).items():
        status = "unverifiable"
        evidence_issues = [issue for issue in policy_assessment.evidence_issues
                           if issue.tool_name is None or issue.tool_name == tool]
        if outcome == "must_not_execute":
            executed = executed_tool_names(execution_evidence)
            if executed is not None:
                status = "failed" if tool in executed else "verified"
        elif tool == "save_text":
            saved = save_assessment
            if saved is not None and saved.outcome == outcome and saved.status != "not_applicable":
                status = saved.status
        else:
            receipts = [receipt for receipt in actions if receipt.kind == tool]
            if not receipts and outcome == "required_success" and not evidence_issues and response is not None:
                # A complete, consistent request history can prove that the
                # required action was omitted. Missing/incomplete histories or
                # an attempted action without its receipt remain unverified.
                evidence = ToolExecutionEvidence.model_validate(execution_evidence)
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
        # A prohibited action does not invalidate another action's observation.
        if status == "verified" and evidence_issues:
            status = "unverifiable"
        assessments[tool] = ActionAssessment(
            tool_name=tool, outcome=outcome, status=status,
            failure_codes=[] if status == "verified" else [f"action_{tool}_{status}"],
        )
    return assessments
