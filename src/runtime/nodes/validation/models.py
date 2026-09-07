from __future__ import annotations

from dataclasses import dataclass, field

from src.core.answer_schema import AnswerResponse
from src.core.contracts import RetrievalDiagnostic
from src.core.contracts.debug import ErrorCode, RetryReason
from src.core.evidence import EvidenceRef, SearchHit
from src.core.planner_schema import PlannerOutput


@dataclass(slots=True)
class ValidationSnapshot:
    user_input: str
    planner_output: PlannerOutput
    retrieval_required: bool
    parsed_hits: list[SearchHit]
    current_attempt_retrieval_errors: list[str]
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic]
    response_result: AnswerResponse | None
    evidence_packet: list[EvidenceRef]
    evidence_by_route: dict[str, list[EvidenceRef]] = field(default_factory=dict)
    diagnostics_by_route: dict[str, list[RetrievalDiagnostic]] = field(default_factory=dict)
    required_routes: list[str] = field(default_factory=list)
    evidence_requirement_map: dict[str, list[str]] = field(default_factory=dict)


@dataclass(slots=True)
class ValidationAssessment:
    blocked_missing_upload: bool = False
    tool_error_routes: set[str] = field(default_factory=set)
    route_failures: dict[str, RetryReason] = field(default_factory=dict)
    valid_unit_paths: set[str] = field(default_factory=set)
    invalid_unit_paths: set[str] = field(default_factory=set)
    missing_route_coverage: list[str] = field(default_factory=list)
    missing_content: list[str] = field(default_factory=list)
    checked_result: AnswerResponse | None = None
    retry_reason: RetryReason | None = None
    failed_routes: set[str] = field(default_factory=set)
    failed_requirement_ids: set[str] = field(default_factory=set)
    score_avg: float | None = None
    error_codes: list[ErrorCode] = field(default_factory=list)
