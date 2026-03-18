from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage

from ..answer_schema import (
    AgentResponsePayloadModel,
    average_claim_confidence,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    filter_claims_by_evidence,
    render_payload_from_claims,
)
from ..evidence import EvidenceItem, parse_evidence_payload
from ..logging_utils import log_event
from .retry import build_followup_from_routes, build_retry_update, contains_tool_error
from .state import (
    RetryReason,
    State,
    coerce_planner_output,
    coerce_retry_context,
    safe_list,
    slice_from_index,
)


logger = logging.getLogger(__name__)

_CODE_IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]{1,}\b")
_KEYWORD_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]{1,}|[가-힣]{2,}")
_KEYWORD_STOPWORDS = {
    "official",
    "docs",
    "documentation",
    "reference",
    "latest",
    "upload",
    "uploaded",
    "current",
    "file",
    "notebook",
    "code",
    "example",
    "examples",
    "local",
    "find",
    "show",
    "tell",
    "explain",
    "describe",
    "how",
    "what",
    "where",
    "using",
    "used",
    "usage",
    "based",
    "공식",
    "문서",
    "최신",
    "업로드",
    "업로드한",
    "파일",
    "노트북",
    "코드",
    "예제",
    "설명",
    "문법",
    "사용",
    "위치",
    "찾아줘",
    "알려줘",
    "기준",
    "실제",
    "부분",
}


def _coerce_response_payload(raw_payload: object) -> AgentResponsePayloadModel | None:
    if raw_payload is None:
        return None
    try:
        return AgentResponsePayloadModel.model_validate(raw_payload)
    except Exception:
        return None


def _payload_to_state_dict(payload: AgentResponsePayloadModel) -> dict[str, object]:
    return payload.model_dump(mode="json")


def _build_followup_updates(answer: str) -> State:
    payload = build_empty_response_payload(answer=answer)
    return {
        "messages": [AIMessage(content=answer)],
        "final_answer": answer,
        "response_payload": _payload_to_state_dict(payload),
    }


def _coerce_evidence_list(items: list[EvidenceItem]) -> list[EvidenceItem]:
    return [item for item in items if isinstance(item, EvidenceItem)]


def _build_response_payload_updates(payload: AgentResponsePayloadModel) -> State:
    return {
        "messages": [AIMessage(content=payload.answer)],
        "final_answer": payload.answer,
        "response_payload": _payload_to_state_dict(payload),
    }


def _route_for_tool(tool_name: str) -> str:
    return {
        "tavily_search": "docs",
        "upload_search": "upload",
        "rag_search": "local",
    }.get(str(tool_name or ""), "")


def _extract_code_identifiers(text: str) -> set[str]:
    return {
        token.lower()
        for token in _CODE_IDENTIFIER_PATTERN.findall(str(text or ""))
        if token and token.lower() not in _KEYWORD_STOPWORDS
    }


def _extract_keywords(text: str) -> set[str]:
    keywords: set[str] = set()
    for token in _KEYWORD_PATTERN.findall(str(text or "").lower()):
        normalized = token.strip().lower()
        if len(normalized) < 2 or normalized in _KEYWORD_STOPWORDS:
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
    def validate_evidence(state: State) -> State:
        local_errors: list[str] = []
        parse_errors: list[str] = []

        planner_output = coerce_planner_output(state.get("planner_output"), local_errors)
        retry_context = coerce_retry_context(state.get("retry_context"))
        guided_followup = str(state.get("guided_followup") or "").strip()
        if guided_followup:
            needs_retry, next_retry_context, _ = build_retry_update(
                retry_context=retry_context,
                retry_reason="blocked_missing_upload",
                planner_output=planner_output,
                retrieval_errors=[],
                score_avg=None,
            )
            updates: State = {
                "needs_retry": needs_retry,
                "retry_context": next_retry_context,
            }
            updates.update(_build_followup_updates(guided_followup))
            return updates

        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        retrieval_error_start_index = int(retry_context.get("retrieval_error_start_index", 0))
        retrieval_diagnostic_start_index = int(
            retry_context.get("retrieval_diagnostic_start_index", 0)
        )

        current_attempt_evidence_payload = slice_from_index(
            safe_list(state.get("retrieved_evidence")),
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
                safe_list(state.get("retrieval_errors")),
                retrieval_error_start_index,
            )
            if str(error).strip()
        ]
        current_attempt_retrieval_diagnostics = [
            item
            for item in slice_from_index(
                safe_list(state.get("retrieval_diagnostics")),
                retrieval_diagnostic_start_index,
            )
            if isinstance(item, dict)
        ]

        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)
        response_payload = _coerce_response_payload(state.get("response_payload"))

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
            route_query = _route_query_for_validation(route, route_diagnostics, state.get("user_input", ""))
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

        updates: State = {
            "needs_retry": needs_retry,
            "retry_context": next_retry_context,
        }
        if retry_reason is not None and not needs_retry:
            if has_grounded_response_payload and response_payload is not None:
                updates.update(_build_response_payload_updates(response_payload))
            elif retry_reason == "unsupported_claims" and valid_claims:
                filtered_confidence = average_claim_confidence(valid_claims)
                filtered_payload = render_payload_from_claims(
                    claims=valid_claims,
                    evidence_items=parsed_evidence,
                    confidence=filtered_confidence,
                )
                filtered_payload.confidence = filtered_confidence
                updates.update(_build_response_payload_updates(filtered_payload))
            elif retrieval_required and parsed_evidence:
                grounded_payload = build_deterministic_grounded_payload(
                    evidence_items=parsed_evidence,
                    fallback_answer="",
                )
                updates.update(_build_response_payload_updates(grounded_payload))
            else:
                followup_answer = build_followup_from_routes(planner_output, retry_reason)
                updates.update(_build_followup_updates(followup_answer))
        if local_errors:
            updates["validation_errors"] = local_errors
        return updates

    return validate_evidence
