"""Deterministic execution policy, independent of quality scores and judges."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.contracts.boundary.retrieval import normalize_retrieval_diagnostic_observation
from src.core.contracts.tool_execution import EXECUTABLE_TOOL_NAMES, ToolExecutionEvidence, ToolExecutionEvent


class ToolPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    version: Literal[1] = 1
    scope: Literal["scenario_turns"] = "scenario_turns"
    final_forbidden_tools: list[str]
    setup_turn_count: int = Field(ge=0)
    setup_forbidden_tools: list[list[str]] | None = None

    @model_validator(mode="after")
    def complete_known_policy(self) -> "ToolPolicySpec":
        if self.setup_forbidden_tools is not None and len(self.setup_forbidden_tools) != self.setup_turn_count:
            raise ValueError("setup policies must cover exactly the declared preparation turns")
        names = [*self.final_forbidden_tools, *(name for names in self.setup_forbidden_tools or [] for name in names)]
        if any(name not in EXECUTABLE_TOOL_NAMES for name in names):
            raise ValueError("policy contains an unknown execution tool")
        return self


class PolicyIssue(BaseModel):
    code: str
    turn_index: int
    request_id: str | None = None
    detail: str | None = None


class PolicyViolation(PolicyIssue):
    tool_name: str
    invocation_id: str


class ToolPolicyAssessment(BaseModel):
    status: Literal["compliant", "violated", "indeterminate"]
    violations: list[PolicyViolation] = Field(default_factory=list)
    evidence_issues: list[PolicyIssue] = Field(default_factory=list)

    @property
    def failure_codes(self) -> list[str]:
        return list(dict.fromkeys(issue.code for issue in [*self.violations, *self.evidence_issues]))


def _raw(evidence: Any) -> Any:
    return evidence.model_dump(mode="json") if isinstance(evidence, BaseModel) else evidence


def _canonical_evidence(evidence: Any) -> Any:
    try:
        return ToolExecutionEvidence.model_validate(_raw(evidence)).model_dump(mode="json")
    except (TypeError, ValueError):
        return _raw(evidence)


def assess_execution_policy(*, policy: ToolPolicySpec | None, evidence: Any,
                            request_id: str | None, prior_turns: list[Any],
                            actions: list[Any] | None = None,
                            retrieval_diagnostics: list[Any] | None = None,
                            debug: dict[str, Any] | None = None,
                            tool_calls: list[str] | None = None,
                            tool_call_count: int | None = None) -> ToolPolicyAssessment:
    """Assess every required turn; final-question prohibitions remain final-only.

    Started events are examined before strict envelope validation. A malformed
    or truncated envelope cannot erase an already observed forbidden execution.
    """
    violations: list[PolicyViolation] = []
    issues: list[PolicyIssue] = []
    # Completion proves provenance, not successful task outcome. Retain the
    # terminal phase so a reused failure never becomes a successful execution.
    completed_invocations: dict[tuple[str, str], Literal["succeeded", "failed"]] = {}
    if policy is None:
        return ToolPolicyAssessment(status="indeterminate", evidence_issues=[
            PolicyIssue(code="tool_policy_snapshot_missing", turn_index=0),
        ])

    def assess(raw: Any, expected_id: str | None, index: int, forbidden: set[str],
               receipts: list[Any], diagnostics: Any, observations: list[dict[str, Any]],
               *, trust_origins: bool = True) -> None:
        initial_issue_count = len(issues)
        raw = _raw(raw)
        observed_id = raw.get("request_id") if isinstance(raw, dict) else None
        detail = {"turn_index": index, "request_id": expected_id}
        # Persisted turns retain raw debug values. Normalize every observation
        # through the same boundary as live responses, keeping discarded input
        # visible as an evidence issue rather than inventing an empty history.
        diagnostic_sources = [diagnostics]
        diagnostic_errors = []
        for observation in observations:
            if "retrieval_diagnostics" in observation and observation["retrieval_diagnostics"] != diagnostics:
                diagnostic_sources.append(observation["retrieval_diagnostics"])
            missing = observation.get("missing_required_debug_fields")
            if isinstance(missing, list) and "retrieval_diagnostics" in missing:
                diagnostic_errors.append("debug.retrieval_diagnostics was unavailable during response normalization")
        normalized_diagnostics = []
        for source in diagnostic_sources:
            normalized, errors = normalize_retrieval_diagnostic_observation(source)
            diagnostic_errors.extend(errors)
            for diagnostic in normalized:
                if diagnostic not in normalized_diagnostics:
                    normalized_diagnostics.append(diagnostic)
        issues.extend(PolicyIssue(code="tool_execution_retrieval_diagnostics_invalid", detail=error, **detail)
                      for error in dict.fromkeys(diagnostic_errors))
        if isinstance(raw, dict) and isinstance(raw.get("events"), list):
            seen = set()
            for item in raw["events"]:
                try:
                    event = ToolExecutionEvent.model_validate(item)
                except (TypeError, ValueError):
                    continue
                if event.phase == "started" and event.tool_name in forbidden:
                    key = (event.invocation_id, event.tool_name)
                    if key not in seen:
                        violations.append(PolicyViolation(code="forbidden_tool_execution", **detail,
                                                          tool_name=event.tool_name, invocation_id=event.invocation_id))
                        seen.add(key)
        if raw is None:
            issues.append(PolicyIssue(code="tool_execution_evidence_missing", **detail))
            return
        try:
            parsed = ToolExecutionEvidence.model_validate(raw)
        except (TypeError, ValueError) as exc:
            issues.append(PolicyIssue(code="tool_execution_evidence_invalid", detail=str(exc), **detail))
            return
        if not expected_id or observed_id != expected_id:
            issues.append(PolicyIssue(code="tool_execution_request_mismatch", **detail))
        if parsed.status != "complete":
            issues.append(PolicyIssue(code="tool_execution_evidence_incomplete", detail=parsed.status, **detail))
        for event in parsed.events:
            if event.tool_name not in EXECUTABLE_TOOL_NAMES:
                issues.append(PolicyIssue(code="tool_execution_unknown_tool", detail=event.tool_name, **detail))
        if parsed.status == "complete":
            started = [event for event in parsed.events if event.phase == "started"]
            names = {event.tool_name for event in started}
            for observation in observations:
                conflicting = False
                if "tool_calls" in observation and observation["tool_calls"] is not None:
                    declared = observation["tool_calls"]
                    conflicting |= (not isinstance(declared, list)
                                    or not all(isinstance(name, str) for name in declared)
                                    or set(declared) != names)
                if "tool_call_count" in observation and observation["tool_call_count"] is not None:
                    count = observation["tool_call_count"]
                    conflicting |= isinstance(count, bool) or not isinstance(count, int) or count != len(started)
                if "execution_evidence" in observation:
                    other = observation["execution_evidence"]
                    if _canonical_evidence(other) != _canonical_evidence(raw):
                        conflicting = True
                        assess(other, expected_id, index, forbidden, [], [], [], trust_origins=False)
                if conflicting:
                    issues.append(PolicyIssue(code="tool_execution_observation_conflict", **detail))
        # Only the selected, complete, consistent request history may establish
        # origins. Alternate conflicting copies still expose forbidden starts.
        trusted_history = trust_origins and len(issues) == initial_issue_count
        for event in parsed.events:
            if event.phase == "reused" and (event.origin_invocation_id, event.tool_name) not in completed_invocations:
                issues.append(PolicyIssue(code="tool_execution_reuse_unverifiable", detail=event.invocation_id, **detail))
            elif trusted_history and event.phase in {"succeeded", "failed"}:
                completed_invocations[(event.invocation_id, event.tool_name)] = event.phase
        starts = {(event.invocation_id, event.tool_name) for event in parsed.events if event.phase == "started"}
        reused = {(event.origin_invocation_id, event.tool_name) for event in parsed.events
                  if event.phase == "reused" and event.origin_invocation_id}
        blocked = {(event.invocation_id, event.tool_name) for event in parsed.events if event.phase == "blocked"}
        for receipt in receipts:
            receipt = _raw(receipt)
            if not isinstance(receipt, dict):
                continue
            if receipt.get("status") == "skipped" and not receipt.get("invocation_id"):
                continue  # A deferred action is not an execution assertion.
            key = (receipt.get("invocation_id"), receipt.get("kind"))
            allowed = starts | reused
            if receipt.get("status") in {"error", "skipped"}:
                allowed |= blocked
            if key not in allowed:
                issues.append(PolicyIssue(code="tool_execution_receipt_unmatched", detail=str(receipt.get("kind")), **detail))
        for diagnostic in normalized_diagnostics:
            if diagnostic.status in {"skipped", "not_run"}:
                continue
            key = (diagnostic.invocation_id, diagnostic.tool)
            if key not in starts | reused | blocked:
                issues.append(PolicyIssue(code="tool_execution_retrieval_unmatched", detail=diagnostic.tool, **detail))

    for index in range(policy.setup_turn_count):
        setup_forbidden = set()
        if policy.setup_forbidden_tools is None or index >= len(policy.setup_forbidden_tools):
            issues.append(PolicyIssue(code="tool_policy_setup_missing", turn_index=index))
        else:
            setup_forbidden = set(policy.setup_forbidden_tools[index])
        if index >= len(prior_turns):
            issues.append(PolicyIssue(code="tool_execution_evidence_missing", turn_index=index,
                                      detail="required setup turn is missing"))
        else:
            turn = prior_turns[index]
            assess(turn.execution_evidence, turn.request_id, index, setup_forbidden,
                   list(turn.response.actions) if turn.response is not None else [],
                   (turn.debug or {}).get("retrieval_diagnostics"),
                   [{"tool_calls": turn.tool_calls}, *([turn.debug] if turn.debug is not None else [])])
    final_turn = prior_turns[-1] if len(prior_turns) == policy.setup_turn_count + 1 else None
    final_turn_matches = (final_turn is not None and final_turn.request_id == request_id
                          and _canonical_evidence(final_turn.execution_evidence) == _canonical_evidence(evidence))
    final_observations = [
        {"tool_calls": tool_calls, "tool_call_count": tool_call_count,
         "retrieval_diagnostics": retrieval_diagnostics if retrieval_diagnostics is not None else []},
        *([debug] if debug is not None else []),
    ]
    if final_turn_matches:
        # A matching execution envelope does not prove the retained diagnostics
        # agree. Check both observations without replaying the same event history.
        final_observations.extend([
            {"tool_calls": final_turn.tool_calls,
             "retrieval_diagnostics": (final_turn.debug or {}).get("retrieval_diagnostics")},
            *([final_turn.debug] if final_turn.debug is not None else []),
        ])
    assess(evidence, request_id, policy.setup_turn_count, set(policy.final_forbidden_tools),
           actions or [], debug.get("retrieval_diagnostics") if debug is not None else retrieval_diagnostics or [],
           final_observations)
    if len(prior_turns) > policy.setup_turn_count + 1:
        issues.append(PolicyIssue(code="tool_execution_turn_coverage_invalid", turn_index=policy.setup_turn_count))
    elif final_turn is not None:
        if not final_turn_matches:
            issues.append(PolicyIssue(code="tool_execution_final_turn_mismatch", turn_index=policy.setup_turn_count))
            # Keep a forbidden start present in the retained actual final turn
            # even when the duplicated case-level envelope was changed.
            assess(final_turn.execution_evidence, final_turn.request_id, policy.setup_turn_count,
                   set(policy.final_forbidden_tools),
                   list(final_turn.response.actions) if final_turn.response is not None else [],
                   (final_turn.debug or {}).get("retrieval_diagnostics"),
                   [final_turn.debug] if final_turn.debug is not None else [], trust_origins=False)
    return ToolPolicyAssessment(status="violated" if violations else "indeterminate" if issues else "compliant",
                                violations=violations, evidence_issues=issues)


def executed_tool_names(evidence: Any) -> list[str] | None:
    """Only verified complete execution observations can replace legacy names."""
    try:
        parsed = ToolExecutionEvidence.model_validate(_raw(evidence))
    except (TypeError, ValueError):
        return None
    if parsed.status != "complete":
        return None
    return list(dict.fromkeys(event.tool_name for event in parsed.events if event.phase == "started"))
