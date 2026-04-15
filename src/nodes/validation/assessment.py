from __future__ import annotations

from typing import Any

from ...answer_schema import filter_claims_by_evidence
from ...contracts.debug import RetryReason
from ...evidence import EvidenceItem
from ...request_contracts import infer_answer_contract, missing_required_sections
from ..retry import contains_tool_error
from .models import ValidationAssessment, ValidationSnapshot
from .route_policy import (
    route_error_statuses,
    route_has_strong_lexical_match,
    route_has_warning,
    route_passes_validation,
    route_query_for_validation,
    route_score_avg,
    resolve_validation_query,
)
from .snapshot import detect_missing_route_coverage, route_for_item_tool


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


def assess_validation(snapshot: ValidationSnapshot) -> ValidationAssessment:
    blocked_missing_upload = bool(
        snapshot.retrieval_required
        and "upload" in snapshot.required_routes
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
            continue
        route_query = route_query_for_validation(route, route_diagnostics, snapshot.user_input)
        validation_query = resolve_validation_query(
            route=route,
            route_query=route_query,
            user_input=snapshot.user_input,
            required_routes=snapshot.required_routes,
        )
        if not route_passes_validation(
            route,
            validation_query,
            route_items,
            required_routes=snapshot.required_routes,
            diagnostics=route_diagnostics,
        ):
            route_failures[route] = "low_score"
            continue
        if (
            route == "docs"
            and route_has_warning(route_diagnostics)
            and not route_has_strong_lexical_match(validation_query, route_items)
        ):
            route_failures[route] = "low_score"

    if contains_tool_error(snapshot.current_attempt_retrieval_errors) and not tool_error_routes:
        tool_error_routes = set(snapshot.required_routes)

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
        and not route_failures
        and not tool_error_routes
        and snapshot.response_payload is not None
        and (
            (snapshot.response_payload.answer.strip() and not snapshot.response_payload.claims)
            or bool(invalid_claims)
        )
    )

    retry_reason: RetryReason | None = None
    failed_routes: set[str] = set()
    if blocked_missing_upload:
        retry_reason = "blocked_missing_upload"
        failed_routes = {"upload"}
    elif tool_error_routes:
        retry_reason = "tool_error"
        failed_routes = set(tool_error_routes)
    elif route_failures:
        failed_routes = set(route_failures)
        retry_reason = "no_evidence" if any(
            reason == "no_evidence" for reason in route_failures.values()
        ) else "low_score"
    elif unsupported_claims:
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

    return ValidationAssessment(
        blocked_missing_upload=blocked_missing_upload,
        tool_error_routes=tool_error_routes,
        route_failures=route_failures,
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
