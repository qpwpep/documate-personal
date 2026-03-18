from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import AIMessage

from ..answer_schema import (
    AgentResponsePayloadModel,
    SynthesisOutput,
    average_claim_confidence,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    filter_claims_by_evidence,
    render_payload_from_claims,
)
from ..contracts.debug import RetryReason
from ..contracts.graph_state import (
    GraphState,
    ResponseState,
    coerce_planner_output,
    debug_state,
    normalize_state_updates,
    planner_state,
    response_state,
    retrieval_state,
    retry_state,
    runtime_state,
    slice_from_index,
)
from ..contracts.routes import route_for_tool
from ..evidence import EvidenceItem, parse_evidence_payload
from ..logging_utils import log_event
from ..rules import get_rules_config
from .retry import build_followup_from_routes, build_retry_update, contains_tool_error


logger = logging.getLogger(__name__)


def _validation_rules():
    return get_rules_config().validation


@lru_cache(maxsize=1)
def _code_identifier_pattern():
    return re.compile(_validation_rules().code_identifier_pattern)


@lru_cache(maxsize=1)
def _keyword_pattern():
    return re.compile(_validation_rules().keyword_pattern)


def _build_response_payload_updates(
    payload: AgentResponsePayloadModel,
    *,
    attempt: int,
) -> GraphState:
    synthesis_output = SynthesisOutput(
        answer=payload.answer,
        claims=payload.claims,
        confidence=payload.confidence,
    )
    return {
        "messages": [AIMessage(content=payload.answer)],
        "response": ResponseState(
            final_answer=payload.answer,
            payload=payload,
            synthesis_output=synthesis_output,
            synthesis_attempt=attempt,
        ),
    }


def _coerce_evidence_list(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if isinstance(item, EvidenceItem)]


def _build_followup_updates(answer: str, *, attempt: int) -> GraphState:
    return _build_response_payload_updates(
        build_empty_response_payload(answer=answer),
        attempt=attempt,
    )


def _route_for_tool(tool_name: str) -> str:
    return route_for_tool(str(tool_name or ""))


def _extract_code_identifiers(text: str) -> set[str]:
    return {
        token.lower()
        for token in _code_identifier_pattern().findall(str(text or ""))
        if token and token.lower() not in set(_validation_rules().keyword_stopwords)
    }


def _extract_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    stopwords = set(_validation_rules().keyword_stopwords)
    for token in _keyword_pattern().findall(str(text or "").lower()):
        normalized = token.strip().lower()
        if len(normalized) < 2 or normalized in stopwords:
            continue
        keywords.add(normalized)
    return keywords


def _combine_evidence_text(items: list[EvidenceItem]) -> str:
    return " ".join(
        part.strip().lower()
        for item in items
        for part in (item.title or "", item.snippet or "", item.url_or_path or "")
        if part and part.strip()
    )


def _route_max_score(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return max(scores)


def _route_score_avg(items: list[EvidenceItem]) -> float | None:
    scores = [float(item.score) for item in items if item.score is not None]
    if not scores:
        return None
    return sum(scores) / len(scores)


def _has_exact_identifier_hit(query: str, items: list[EvidenceItem]) -> bool:
    identifiers = _extract_code_identifiers(query)
    if not identifiers:
        return False
    combined_text = _combine_evidence_text(items)
    return any(identifier in combined_text for identifier in identifiers)


def _keyword_overlap_count(query: str, items: list[EvidenceItem]) -> int:
    query_keywords = _extract_keywords(query)
    if not query_keywords:
        return 0
    evidence_keywords = _extract_keywords(_combine_evidence_text(items))
    return len(query_keywords.intersection(evidence_keywords))


def _route_has_strong_lexical_match(query: str, items: list[EvidenceItem]) -> bool:
    return _has_exact_identifier_hit(query, items) or _keyword_overlap_count(query, items) >= 2


def _route_passes_validation(route: str, query: str, items: list[EvidenceItem]) -> bool:
    if not items:
        return False
    max_score = _route_max_score(items)
    if route == "docs":
        return bool((max_score is not None and max_score >= 0.5) or _route_has_strong_lexical_match(query, items))
    query_identifiers = _extract_code_identifiers(query)
    query_keywords = _extract_keywords(query)
    if not query_identifiers and len(query_keywords) < 2:
        return True
    return bool(
        _has_exact_identifier_hit(query, items)
        or _keyword_overlap_count(query, items) >= 2
        or (max_score is not None and max_score > 0.0)
    )


def _route_query_for_validation(route: str, diagnostics: list[dict[str, Any]], fallback_query: str) -> str:
    for item in diagnostics:
        if str(item.get("route") or "").strip() != route:
            continue
        query = str(item.get("query") or "").strip()
        if query:
            return query
    return fallback_query


def _route_error_statuses(diagnostics: list[dict[str, Any]]) -> set[str]:
    return {
        str(item.get("status") or "").strip()
        for item in diagnostics
        if str(item.get("status") or "").strip()
    }


def _score_avg_for_failed_routes(
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


def make_validate_evidence_node(verbose: bool):
    def validate_evidence(state: GraphState) -> GraphState:
        local_errors: list[str] = []
        parse_errors: list[str] = []
        runtime = runtime_state(state)
        planner = planner_state(state)
        retrieval = retrieval_state(state)
        response = response_state(state)
        debug = debug_state(state)

        planner_output = coerce_planner_output(planner.output, local_errors)
        retry_context = retry_state(state)
        guided_followup = str(planner.guided_followup or "").strip()
        if guided_followup:
            needs_retry, next_retry_context, _ = build_retry_update(
                retry_context=retry_context,
                retry_reason="blocked_missing_upload",
                planner_output=planner_output,
                retrieval_errors=[],
                score_avg=None,
            )
            updates: GraphState = {
                "retry": next_retry_context,
            }
            updates.update(_build_followup_updates(guided_followup, attempt=response.synthesis_attempt))
            return normalize_state_updates(updates)

        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        retrieval_error_start_index = int(retry_context.get("retrieval_error_start_index", 0))
        retrieval_diagnostic_start_index = int(
            retry_context.get("retrieval_diagnostic_start_index", 0)
        )

        current_attempt_evidence_payload = slice_from_index(
            retrieval.evidence_log,
            evidence_start_index,
        )
        parsed_evidence = _coerce_evidence_list(
            parse_evidence_payload(
                current_attempt_evidence_payload,
                context="retrieved_evidence",
                errors=parse_errors,
            )
        )
        local_errors.extend(parse_errors)

        current_attempt_retrieval_errors = [
            str(error)
            for error in slice_from_index(
                debug.retrieval_errors,
                retrieval_error_start_index,
            )
            if str(error).strip()
        ]
        current_attempt_retrieval_diagnostics = [
            item
            for item in slice_from_index(
                debug.retrieval_diagnostics,
                retrieval_diagnostic_start_index,
            )
            if item is not None
        ]

        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
        response_payload = response.payload

        evidence_by_route: dict[str, list[EvidenceItem]] = {"docs": [], "upload": [], "local": []}
        for item in parsed_evidence:
            route = _route_for_tool(item.tool)
            if route:
                evidence_by_route.setdefault(route, []).append(item)

        diagnostics_by_route: dict[str, list[dict[str, Any]]] = {"docs": [], "upload": [], "local": []}
        for item in current_attempt_retrieval_diagnostics:
            route = str(item.get("route") or "").strip()
            if route:
                diagnostics_by_route.setdefault(route, []).append(item)

        required_routes = [task.route for task in planner_output.tasks] if retrieval_required else []
        blocked_missing_upload = bool(
            retrieval_required
            and "upload" in required_routes
            and any(
                str(item.get("status") or "") == "unavailable"
                for item in diagnostics_by_route.get("upload", [])
            )
        )

        tool_error_routes: set[str] = set()
        route_failures: dict[str, RetryReason] = {}
        for route in required_routes:
            route_items = evidence_by_route.get(route, [])
            route_diagnostics = diagnostics_by_route.get(route, [])
            route_statuses = _route_error_statuses(route_diagnostics)
            if "error" in route_statuses or ("unavailable" in route_statuses and route != "upload"):
                tool_error_routes.add(route)
                continue
            if not route_items:
                route_failures[route] = "no_evidence"
                continue
            route_query = _route_query_for_validation(route, route_diagnostics, runtime.user_input)
            if not _route_passes_validation(route, route_query, route_items):
                route_failures[route] = "low_score"

        if contains_tool_error(current_attempt_retrieval_errors + parse_errors) and not tool_error_routes:
            tool_error_routes = set(required_routes)

        valid_claims: list[Any] = []
        invalid_claims: list[Any] = []
        if retrieval_required and response_payload is not None:
            valid_claims, invalid_claims = filter_claims_by_evidence(
                claims=response_payload.claims,
                evidence_items=parsed_evidence,
            )
        has_grounded_response_payload = bool(
            response_payload is not None
            and response_payload.answer.strip()
            and valid_claims
            and not invalid_claims
        )

        unsupported_claims = bool(
            retrieval_required
            and not route_failures
            and not tool_error_routes
            and response_payload is not None
            and (
                (response_payload.answer.strip() and not response_payload.claims)
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

        score_avg = _score_avg_for_failed_routes(failed_routes, evidence_by_route)
        if score_avg is None:
            score_avg = _route_score_avg(parsed_evidence)

        needs_retry, next_retry_context, retrieval_feedback = build_retry_update(
            retry_context=retry_context,
            retry_reason=retry_reason,
            planner_output=planner_output,
            retrieval_errors=current_attempt_retrieval_errors + parse_errors,
            score_avg=score_avg,
            failed_routes=failed_routes,
            current_attempt_evidence=parsed_evidence,
            current_attempt_retrieval_diagnostics=current_attempt_retrieval_diagnostics,
        )

        if retry_reason is not None:
            local_errors.append(
                "validate_evidence: retry_reason="
                f"{retry_reason}, failed_routes={sorted(failed_routes)}, "
                f"score_avg={score_avg}, feedback={retrieval_feedback}"
            )

        if verbose:
            log_event(
                logger,
                logging.INFO,
                "validate_evidence",
                retrieval_required=retrieval_required,
                evidence_count=len(parsed_evidence),
                needs_retry=needs_retry,
                retry_reason=retry_reason,
            )

        updates: GraphState = {
            "retry": next_retry_context,
        }
        if retry_reason is not None and not needs_retry:
            if has_grounded_response_payload and response_payload is not None:
                updates.update(
                    _build_response_payload_updates(
                        response_payload,
                        attempt=response.synthesis_attempt,
                    )
                )
            elif retry_reason == "unsupported_claims" and valid_claims:
                filtered_confidence = average_claim_confidence(valid_claims)
                filtered_payload = render_payload_from_claims(
                    claims=valid_claims,
                    evidence_items=parsed_evidence,
                    confidence=filtered_confidence,
                )
                filtered_payload.confidence = filtered_confidence
                updates.update(
                    _build_response_payload_updates(
                        filtered_payload,
                        attempt=response.synthesis_attempt,
                    )
                )
            elif retrieval_required and parsed_evidence:
                grounded_payload = build_deterministic_grounded_payload(
                    evidence_items=parsed_evidence,
                    fallback_answer="",
                )
                updates.update(
                    _build_response_payload_updates(
                        grounded_payload,
                        attempt=response.synthesis_attempt,
                    )
                )
            else:
                followup_answer = build_followup_from_routes(planner_output, retry_reason)
                updates.update(
                    _build_followup_updates(
                        followup_answer,
                        attempt=response.synthesis_attempt,
                    )
                )
        if local_errors:
            updates["debug"] = debug.model_copy(
                update={"validation_errors": [*debug.validation_errors, *local_errors]}
            )
        return normalize_state_updates(updates)

    return validate_evidence
