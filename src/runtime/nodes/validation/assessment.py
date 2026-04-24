from __future__ import annotations

from typing import Any

from src.core.answer_schema import filter_claims_by_evidence
from src.core.contracts.debug import RetryReason
from src.core.evidence import EvidenceItem
from src.core.request_contracts import infer_answer_contract, missing_required_sections
from src.runtime.nodes.retry import contains_tool_error
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot
from src.runtime.nodes.validation.route_policy import route_error_statuses, route_score_avg
from src.runtime.nodes.validation.snapshot import detect_missing_route_coverage, route_for_item_tool


def score_avg_for_failed_routes(
    failed_routes: set[str],
    evidence_by_route: dict[str, list[EvidenceItem]],
) -> float | None:
    if not failed_routes:
        return None
    scores = [
        float(item.score)
        for route in failed_routes
        for item in evidence_by_route.get(route, [])
        if item.score is not None
    ]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _empty_validation_assessment(
    snapshot: ValidationSnapshot,
    *,
    blocked_missing_upload: bool = False,
    tool_error_routes: set[str] | None = None,
    route_failures: dict[str, RetryReason] | None = None,
    valid_claims: list[Any] | None = None,
    invalid_claims: list[Any] | None = None,
    missing_route_coverage: list[str] | None = None,
    missing_sections: list[str] | None = None,
    has_grounded_response_payload: bool = False,
    unsupported_claims: bool = False,
    retry_reason: RetryReason | None = None,
    failed_routes: set[str] | None = None,
    score_avg: float | None = None,
) -> ValidationAssessment:
    return ValidationAssessment(
        blocked_missing_upload=blocked_missing_upload,
        tool_error_routes=tool_error_routes or set(),
        route_failures=route_failures or {},
        valid_claims=valid_claims or [],
        invalid_claims=invalid_claims or [],
        missing_route_coverage=missing_route_coverage or [],
        missing_sections=missing_sections or [],
        has_grounded_response_payload=has_grounded_response_payload,
        unsupported_claims=unsupported_claims,
        retry_reason=retry_reason,
        failed_routes=failed_routes or set(),
        score_avg=score_avg if score_avg is not None else route_score_avg(snapshot.parsed_evidence),
    )


def assess_retrieval_quality(snapshot: ValidationSnapshot) -> ValidationAssessment:
    if not snapshot.retrieval_required:
        return _empty_validation_assessment(snapshot)

    blocked_missing_upload = bool(
        "upload" in snapshot.required_routes
        and any(
            str(item.status or "") == "unavailable"
            for item in snapshot.diagnostics_by_route.get("upload", [])
        )
    )

    tool_error_routes: set[str] = set()
    route_failures: dict[str, RetryReason] = {}
    for route in snapshot.required_routes:
        route_items = snapshot.evidence_by_route.get(route, [])
        route_diagnostics = snapshot.diagnostics_by_route.get(route, [])
        statuses = route_error_statuses(route_diagnostics)
        if "error" in statuses or ("unavailable" in statuses and route != "upload"):
            tool_error_routes.add(route)
            continue
        if not route_items:
            route_failures[route] = "no_evidence"

    if contains_tool_error(snapshot.current_attempt_retrieval_errors) and not tool_error_routes:
        tool_error_routes = set(snapshot.required_routes)

    retry_reason: RetryReason | None = None
    failed_routes: set[str] = set()
    if blocked_missing_upload:
        retry_reason = "blocked_missing_upload"
        failed_routes = {"upload"}
    elif tool_error_routes:
        retry_reason = "tool_error"
        failed_routes = set(tool_error_routes)
    elif route_failures:
        retry_reason = "no_evidence"
        failed_routes = set(route_failures)

    score_avg = score_avg_for_failed_routes(failed_routes, snapshot.evidence_by_route)
    if score_avg is None:
        score_avg = route_score_avg(snapshot.parsed_evidence)

    return _empty_validation_assessment(
        snapshot,
        blocked_missing_upload=blocked_missing_upload,
        tool_error_routes=tool_error_routes,
        route_failures=route_failures,
        retry_reason=retry_reason,
        failed_routes=failed_routes,
        score_avg=score_avg,
    )


def assess_validation(snapshot: ValidationSnapshot) -> ValidationAssessment:
    valid_claims: list[Any] = []
    invalid_claims: list[Any] = []
    if snapshot.retrieval_required and snapshot.response_payload is not None:
        valid_claims, invalid_claims = filter_claims_by_evidence(
            claims=snapshot.response_payload.claims,
            evidence_items=snapshot.parsed_evidence,
        )

    route_by_source_id = {
        str(item.source_id or "").strip(): route_for_item_tool(item.tool)
        for item in snapshot.parsed_evidence
        if str(item.source_id or "").strip()
    }
    missing_route_coverage = detect_missing_route_coverage(
        required_routes=snapshot.required_routes,
        valid_claims=valid_claims,
        route_by_source_id=route_by_source_id,
    ) if snapshot.retrieval_required else []

    answer_contract = infer_answer_contract(snapshot.user_input, snapshot.required_routes)
    missing_sections = missing_required_sections(answer_contract, snapshot.response_payload)

    has_grounded_response_payload = bool(
        snapshot.response_payload is not None
        and snapshot.response_payload.answer.strip()
        and valid_claims
        and not invalid_claims
        and not missing_route_coverage
        and not missing_sections
    )

    unsupported_claims = bool(
        snapshot.retrieval_required
        and snapshot.response_payload is not None
        and (
            (snapshot.response_payload.answer.strip() and not snapshot.response_payload.claims)
            or bool(invalid_claims)
        )
    )

    retry_reason: RetryReason | None = None
    failed_routes: set[str] = set()
    if unsupported_claims:
        retry_reason = "unsupported_claims"
        failed_routes = set(missing_route_coverage)
    elif missing_route_coverage:
        retry_reason = "missing_route_coverage"
        failed_routes = set(missing_route_coverage)
    elif missing_sections:
        retry_reason = "missing_sections"

    score_avg = score_avg_for_failed_routes(failed_routes, snapshot.evidence_by_route)
    if score_avg is None:
        score_avg = route_score_avg(snapshot.parsed_evidence)

    return _empty_validation_assessment(
        snapshot,
        valid_claims=valid_claims,
        invalid_claims=invalid_claims,
        missing_route_coverage=missing_route_coverage,
        missing_sections=missing_sections,
        has_grounded_response_payload=has_grounded_response_payload,
        unsupported_claims=unsupported_claims,
        retry_reason=retry_reason,
        failed_routes=failed_routes,
        score_avg=score_avg,
    )
