from __future__ import annotations

from dataclasses import dataclass, field
import re
from functools import lru_cache
from typing import Any

from ...answer_schema import AgentResponsePayloadModel, filter_claims_by_evidence
from ...request_contracts import infer_answer_contract, missing_required_sections
from ...contracts import RetrievalDiagnostic
from ...contracts.debug import RetryReason
from ...contracts.routes import route_for_tool
from ...evidence import EvidenceItem
from ...planner_schema import PlannerOutput
from ...rules import get_rules_config
from ..retry import contains_tool_error


def _validation_rules():
    return get_rules_config().validation


@lru_cache(maxsize=1)
def _code_identifier_pattern():
    return re.compile(_validation_rules().code_identifier_pattern)


@lru_cache(maxsize=1)
def _embedded_code_identifier_pattern():
    return re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}")


@lru_cache(maxsize=1)
def _keyword_pattern():
    return re.compile(_validation_rules().keyword_pattern)


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


def coerce_evidence_list(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if isinstance(item, EvidenceItem)]


def route_for_item_tool(tool_name: str) -> str:
    return route_for_tool(str(tool_name or ""))


def detect_missing_route_coverage(
    *,
    required_routes: list[str],
    valid_claims: list[Any],
    route_by_source_id: dict[str, str],
) -> list[str]:
    covered_routes: set[str] = set()
    for claim in valid_claims:
        for source_id in getattr(claim, "evidence_ids", []) or []:
            route = route_by_source_id.get(str(source_id or "").strip(), "")
            if route:
                covered_routes.add(route)
    return [route for route in required_routes if route not in covered_routes]


def extract_code_identifiers(text: str) -> set[str]:
    matches = set(_code_identifier_pattern().findall(str(text or "")))
    matches.update(_embedded_code_identifier_pattern().findall(str(text or "")))
    return {
        token.lower()
        for token in matches
        if token and token.lower() not in set(_validation_rules().keyword_stopwords)
    }


def extract_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    stopwords = set(_validation_rules().keyword_stopwords)
    for token in _keyword_pattern().findall(str(text or "").lower()):
        normalized = token.strip().lower()
        if len(normalized) < 2 or normalized in stopwords:
            continue
        keywords.add(normalized)
    return keywords


def combine_evidence_text(items: list[EvidenceItem]) -> str:
    return " ".join(
        part.strip().lower()
        for item in items
        for part in (
            item.title or "",
            item.snippet or "",
            item.url_or_path or "",
            item.document_id or "",
            item.source_id or "",
        )
        if part and part.strip()
    )


def route_max_score(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return max(scores)


def route_score_avg(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def has_exact_identifier_hit(query: str, items: list[EvidenceItem]) -> bool:
    identifiers = extract_code_identifiers(query)
    if not identifiers:
        return False
    combined_text = combine_evidence_text(items)
    return any(identifier in combined_text for identifier in identifiers)


def keyword_overlap_count(query: str, items: list[EvidenceItem]) -> int:
    query_keywords = extract_keywords(query)
    if not query_keywords:
        return 0
    evidence_keywords = extract_keywords(combine_evidence_text(items))
    return len(query_keywords.intersection(evidence_keywords))


def route_has_strong_lexical_match(query: str, items: list[EvidenceItem]) -> bool:
    return has_exact_identifier_hit(query, items) or keyword_overlap_count(query, items) >= 2


def route_passes_validation(route: str, query: str, items: list[EvidenceItem]) -> bool:
    if not items:
        return False
    max_score = route_max_score(items)
    if route == "docs":
        return bool((max_score is not None and max_score >= 0.5) or route_has_strong_lexical_match(query, items))
    query_identifiers = extract_code_identifiers(query)
    query_keywords = extract_keywords(query)
    if not query_identifiers and len(query_keywords) < 2:
        return True
    return bool(
        has_exact_identifier_hit(query, items)
        or keyword_overlap_count(query, items) >= 2
        or (max_score is not None and max_score > 0.0)
    )


def route_query_for_validation(
    route: str,
    diagnostics: list[RetrievalDiagnostic],
    fallback_query: str,
) -> str:
    for item in diagnostics:
        if str(item.route or "").strip() != route:
            continue
        query = str(item.query or "").strip()
        if query:
            return query
    return fallback_query


def route_error_statuses(diagnostics: list[RetrievalDiagnostic]) -> set[str]:
    return {
        str(item.status or "").strip()
        for item in diagnostics
        if str(item.status or "").strip()
    }


def route_has_warning(diagnostics: list[RetrievalDiagnostic]) -> bool:
    return any(item.warnings for item in diagnostics)


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


def build_validation_snapshot(
    *,
    user_input: str,
    planner_output: PlannerOutput,
    parsed_evidence: list[EvidenceItem],
    current_attempt_retrieval_errors: list[str],
    current_attempt_retrieval_diagnostics: list[RetrievalDiagnostic],
    response_payload: AgentResponsePayloadModel | None,
) -> ValidationSnapshot:
    retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
    evidence_by_route: dict[str, list[EvidenceItem]] = {"docs": [], "upload": [], "local": []}
    for item in parsed_evidence:
        route = route_for_item_tool(item.tool)
        if route:
            evidence_by_route.setdefault(route, []).append(item)

    diagnostics_by_route: dict[str, list[RetrievalDiagnostic]] = {"docs": [], "upload": [], "local": []}
    for item in current_attempt_retrieval_diagnostics:
        route = str(item.route or "").strip()
        if route:
            diagnostics_by_route.setdefault(route, []).append(item)

    required_routes = [task.route for task in planner_output.tasks] if retrieval_required else []
    return ValidationSnapshot(
        user_input=user_input,
        planner_output=planner_output,
        retrieval_required=retrieval_required,
        parsed_evidence=parsed_evidence,
        current_attempt_retrieval_errors=current_attempt_retrieval_errors,
        current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics,
        response_payload=response_payload,
        evidence_by_route=evidence_by_route,
        diagnostics_by_route=diagnostics_by_route,
        required_routes=required_routes,
    )


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
        if not route_passes_validation(route, route_query, route_items):
            route_failures[route] = "low_score"
            continue
        if route_has_warning(route_diagnostics) and not route_has_strong_lexical_match(route_query, route_items):
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
