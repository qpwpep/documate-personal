from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ...answer_schema import AgentResponsePayloadModel
from ...contracts import RetrievalDiagnostic
from ...contracts.debug import RetryReason
from ...evidence import EvidenceItem
from ...planner_schema import PlannerOutput


@dataclass(slots=True)
class ValidationSnapshot:
    user_input: str
    planner_output: PlannerOutput
    retrieval_required: bool
    parsed_evidence: list[EvidenceItem]
    current_attempt_retrieval_errors: list[str]
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic]
    response_payload: AgentResponsePayloadModel | None
    evidence_by_route: dict[str, list[EvidenceItem]] = field(default_factory=dict)
    diagnostics_by_route: dict[str, list[RetrievalDiagnostic]] = field(default_factory=dict)
    required_routes: list[str] = field(default_factory=list)


@dataclass(slots=True)
class ValidationAssessment:
    blocked_missing_upload: bool
    tool_error_routes: set[str]
    route_failures: dict[str, RetryReason]
    valid_claims: list[Any]
    invalid_claims: list[Any]
    missing_route_coverage: list[str]
    missing_sections: list[str]
    has_grounded_response_payload: bool
    unsupported_claims: bool
    retry_reason: RetryReason | None
    failed_routes: set[str]
    score_avg: float | None
