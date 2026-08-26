from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from langchain_core.messages import AIMessage

from src.core.contracts import GraphState, PlannerState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.debug import (
    LLMCallMetadata,
    RetryState,
    build_llm_call_metadata,
    empty_planner_diagnostic,
)
from src.core.planner_schema import PlannerOutput, normalize_planner_output_input
from src.infra.logging_utils import log_event
from src.runtime.nodes.actions import is_action_only_request
from src.runtime.nodes.planner.deterministic import build_deterministic_planner_decision
from src.runtime.nodes.planner.guardrails import (
    apply_required_route_guardrail,
    sanitize_planner_output,
)
from src.runtime.nodes.planner.heuristic import build_heuristic_planner_decision
from src.runtime.nodes.planner.models import (
    PlannerDecision,
    normalize_planner_diagnostics,
)
from src.runtime.nodes.planner.prompt_builder import build_planner_messages
from src.runtime.nodes.planner.query_sanitizer import sanitize_planner_output_queries

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlannerRunContext:
    user_input: str
    has_retriever: bool
    planner_attempt: int


def _coerce_planner_payload(raw: Any) -> Any:
    if isinstance(raw, PlannerOutput):
        return raw
    return normalize_planner_output_input(raw)


def _content_text_candidates(content: Any) -> list[str]:
    if isinstance(content, str):
        return [content]
    if not isinstance(content, list):
        return []

    candidates: list[str] = []
    for part in content:
        if isinstance(part, str):
            candidates.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text") or part.get("content")
        if isinstance(text, str):
            candidates.append(text)
    return candidates


def _json_payload_from_text(text: str) -> Any | None:
    stripped = str(text or "").strip()
    if not stripped:
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except json.JSONDecodeError:
        return None


def _looks_like_planner_payload(value: Any) -> bool:
    return isinstance(value, dict) and ("use_retrieval" in value or "tasks" in value)


def _find_planner_payload(value: Any) -> Any | None:
    if _looks_like_planner_payload(value):
        return value

    if isinstance(value, str):
        parsed = _json_payload_from_text(value)
        if _looks_like_planner_payload(parsed):
            return parsed
        return None

    if isinstance(value, dict):
        for item in value.values():
            found = _find_planner_payload(item)
            if found is not None:
                return found
        return None

    if isinstance(value, (list, tuple)):
        for item in value:
            found = _find_planner_payload(item)
            if found is not None:
                return found
    return None


def _coerce_planner_payload_from_raw_message(raw_message: AIMessage | None) -> Any | None:
    if raw_message is None:
        return None
    for text in _content_text_candidates(raw_message.content):
        payload = _json_payload_from_text(text)
        if _looks_like_planner_payload(payload):
            return payload
    for attribute_name in ("additional_kwargs", "response_metadata"):
        payload = _find_planner_payload(getattr(raw_message, attribute_name, None))
        if payload is not None:
            return payload
    return None


def _validate_planner_payload(payload: Any) -> tuple[PlannerOutput | None, list[str], Exception | None]:
    warnings: list[str] = []
    try:
        return PlannerOutput.validate_input(_coerce_planner_payload(payload), warnings=warnings), warnings, None
    except Exception as exc:
        return None, warnings, exc


def _coerce_structured_planner_result(
    result: Any,
) -> tuple[PlannerOutput | None, AIMessage | None, Exception | None, list[str]]:
    if isinstance(result, PlannerOutput):
        return result, None, None, []

    if not isinstance(result, dict):
        planner_output, warnings, error = _validate_planner_payload(result)
        return planner_output, None, error, warnings

    if "use_retrieval" in result or "tasks" in result:
        planner_output, warnings, error = _validate_planner_payload(result)
        return planner_output, None, error, warnings

    raw_message = result.get("raw")
    parsed = _coerce_planner_payload(result.get("parsed"))
    parsing_error = result.get("parsing_error")

    if not isinstance(raw_message, AIMessage):
        raw_message = None

    if parsed is not None:
        planner_output, warnings, error = _validate_planner_payload(parsed)
        if planner_output is not None:
            return planner_output, raw_message, None, warnings
    else:
        error = None

    if parsing_error is not None and isinstance(parsing_error, Exception):
        raw_payload = _coerce_planner_payload_from_raw_message(raw_message)
        if raw_payload is not None:
            planner_output, warnings, raw_error = _validate_planner_payload(raw_payload)
            if planner_output is not None:
                return planner_output, raw_message, None, warnings
            error = raw_error
        return None, raw_message, parsing_error if error is None else error, []
    if parsing_error is not None:
        raw_payload = _coerce_planner_payload_from_raw_message(raw_message)
        if raw_payload is not None:
            planner_output, warnings, raw_error = _validate_planner_payload(raw_payload)
            if planner_output is not None:
                return planner_output, raw_message, None, warnings
            error = raw_error
        return None, raw_message, RuntimeError(str(parsing_error) if error is None else str(error)), []

    if isinstance(parsed, PlannerOutput):
        return parsed, raw_message, None, []

    return None, raw_message, error, []


def _resolve_planner_strategy(
    *,
    llm_planner: Any,
    state: GraphState,
    context: PlannerRunContext,
    max_turns: int,
) -> tuple[PlannerDecision, list[str], list[LLMCallMetadata]]:
    planner_errors: list[str] = []
    llm_calls: list[LLMCallMetadata] = []

    try:
        planner_messages = build_planner_messages(state, max_turns=max_turns)
        planner_raw = llm_planner.invoke(planner_messages)
        planner_output, raw_message, parse_error, planner_warnings = _coerce_structured_planner_result(planner_raw)
        if raw_message is not None:
            llm_calls.append(
                build_llm_call_metadata(
                    stage="planner",
                    attempt=context.planner_attempt,
                    path="structured",
                    message=raw_message,
                )
            )
        if planner_output is not None:
            return (
                PlannerDecision(
                    output=planner_output,
                    diagnostics=normalize_planner_diagnostics(
                        status="llm",
                        reason=None,
                        fallback_routes=[],
                        planner_warnings=planner_warnings,
                    ),
                    status="llm",
                ),
                planner_errors,
                llm_calls,
            )
        planner_errors.append(f"planner: output validation failed ({parse_error})")
    except Exception as exc:
        planner_errors.append(f"planner: structured output invocation failed ({exc})")

    deterministic = build_deterministic_planner_decision(
        user_input=context.user_input,
        has_retriever=context.has_retriever,
    )
    if deterministic is not None:
        return deterministic, planner_errors, llm_calls

    decision = build_heuristic_planner_decision(
        user_input=context.user_input,
        has_retriever=context.has_retriever,
    )
    return decision, planner_errors, llm_calls


def _error_codes_from_planner_errors(errors: list[str]) -> list[str]:
    codes: list[str] = []
    for error in errors:
        lowered = str(error or "").lower()
        if (
            "output validation failed" in lowered
            or "schema" in lowered
        ) and "PLANNER_SCHEMA_INVALID" not in codes:
            codes.append("PLANNER_SCHEMA_INVALID")
        if ("timeout" in lowered or "timed out" in lowered) and "PLANNER_TIMEOUT" not in codes:
            codes.append("PLANNER_TIMEOUT")
    return codes


def _apply_planner_guardrail(
    *,
    decision: PlannerDecision,
    context: PlannerRunContext,
    retry_context: RetryState,
    planner_errors: list[str],
) -> PlannerDecision:
    planner_output = sanitize_planner_output(
        decision.output,
        has_retriever=context.has_retriever,
        errors=planner_errors,
    )
    if is_action_only_request(context.user_input) and planner_output.use_retrieval:
        return PlannerDecision(
            output=PlannerOutput.fallback(),
            diagnostics=normalize_planner_diagnostics(
                status=decision.status,
                reason="action_only",
                fallback_routes=[],
                planner_warnings=decision.diagnostics.planner_warnings,
            ),
            guided_followup=decision.guided_followup,
            status=decision.status,
        )
    planner_output = sanitize_planner_output_queries(
        planner_output,
        user_input=context.user_input,
        retry_context=retry_context,
    )
    planner_output, planner_diagnostics, guardrail_followup = apply_required_route_guardrail(
        planner_output=planner_output,
        planner_status=decision.status,
        planner_diagnostics=decision.diagnostics,
        user_input=context.user_input,
        has_retriever=context.has_retriever,
    )
    return PlannerDecision(
        output=planner_output,
        diagnostics=planner_diagnostics,
        guided_followup=guardrail_followup or decision.guided_followup,
        status=decision.status,
    )


def _reset_retry_window(
    *,
    existing_retry_context: RetryState,
    retrieval_evidence_count: int,
    retrieval_error_count: int,
    retrieval_diagnostic_count: int,
) -> RetryState:
    retry_context = existing_retry_context.model_copy(
        update={
            "needs_retry": False,
            "max_retries": int(existing_retry_context.max_retries),
            "evidence_start_index": retrieval_evidence_count,
            "retrieval_error_start_index": retrieval_error_count,
            "retrieval_diagnostic_start_index": retrieval_diagnostic_count,
        }
    )
    if int(retry_context.attempt) <= 0:
        retry_context = retry_context.model_copy(
            update={
                "retrieval_feedback": "",
                "score_avg": None,
                "retry_reason": None,
                "failed_routes": [],
                "preserved_evidence": [],
                "preserved_retrieval_diagnostics": [],
            }
        )
    return retry_context


def make_planner_node(
    llm_planner: Any,
    verbose: bool,
    max_turns: int = 6,
):
    def planner(state: GraphState) -> GraphState:
        runtime = get_runtime_state(state)
        retrieval = get_retrieval_state(state)
        debug = get_debug_state(state)
        existing_retry_context = get_retry_state(state)
        context = PlannerRunContext(
            user_input=runtime.user_input,
            has_retriever=bool(runtime.retriever),
            planner_attempt=int(existing_retry_context.attempt) + 1,
        )

        decision, planner_errors, llm_calls = _resolve_planner_strategy(
            llm_planner=llm_planner,
            state=state,
            context=context,
            max_turns=max_turns,
        )
        decision = _apply_planner_guardrail(
            decision=decision,
            context=context,
            retry_context=existing_retry_context,
            planner_errors=planner_errors,
        )

        if verbose:
            log_event(
                logger,
                logging.INFO,
                "planner",
                status=decision.status,
                use_retrieval=decision.output.use_retrieval,
                task_count=len(decision.output.tasks),
                required_routes=decision.diagnostics.required_routes,
                override=decision.diagnostics.override_applied,
            )

        retry_context = _reset_retry_window(
            existing_retry_context=existing_retry_context,
            retrieval_evidence_count=len(retrieval.evidence_log),
            retrieval_error_count=len(debug.retrieval_errors),
            retrieval_diagnostic_count=len(debug.retrieval_diagnostics),
        )

        updates: GraphState = {
            "planner": PlannerState(
                output=decision.output,
                status=decision.status,
                diagnostics=decision.diagnostics or empty_planner_diagnostic(status=decision.status),
                guided_followup=decision.guided_followup,
            ),
            "retry": retry_context,
        }
        if planner_errors or llm_calls:
            planner_error_codes = _error_codes_from_planner_errors(planner_errors)
            updates["debug"] = debug.model_copy(
                update={
                    "planner_errors": [*debug.planner_errors, *planner_errors],
                    "error_codes": [
                        *debug.error_codes,
                        *[
                            code
                            for code in planner_error_codes
                            if code not in debug.error_codes
                        ],
                    ],
                    "llm_calls": [*debug.llm_calls, *llm_calls],
                }
            )
        return updates

    return planner
