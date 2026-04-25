from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

from langchain_core.messages import AIMessage

from src.core.contracts import GraphState, PlannerState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.debug import LLMCallMetadata, RetryState, build_llm_call_metadata, empty_planner_diagnostic
from src.infra.logging_utils import log_event
from src.core.planner_schema import PlannerOutput, normalize_planner_output_input
from src.runtime.nodes.planner.deterministic import build_deterministic_planner_decision
from src.runtime.nodes.planner.guardrails import apply_required_route_guardrail, sanitize_planner_output
from src.runtime.nodes.planner.heuristic import build_heuristic_planner_decision
from src.runtime.nodes.planner.models import PlannerDecision, normalize_planner_diagnostics
from src.runtime.nodes.planner.prompt_builder import build_planner_messages
from src.runtime.nodes.planner.query_sanitizer import sanitize_planner_output_queries


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlannerRunContext:
    user_input: str
    has_retriever: bool
    planner_attempt: int
    planner_mode: str = "auto"


def _coerce_planner_payload(raw: Any) -> Any:
    if isinstance(raw, PlannerOutput):
        return raw
    return normalize_planner_output_input(raw)


def _coerce_structured_planner_result(
    result: Any,
) -> tuple[PlannerOutput | None, AIMessage | None, Exception | None]:
    if isinstance(result, PlannerOutput):
        return result, None, None

    if not isinstance(result, dict):
        try:
            return PlannerOutput.validate_input(_coerce_planner_payload(result)), None, None
        except Exception as exc:
            return None, None, exc

    raw_message = result.get("raw")
    parsed = _coerce_planner_payload(result.get("parsed"))
    parsing_error = result.get("parsing_error")

    if not isinstance(raw_message, AIMessage):
        raw_message = None

    if parsing_error is not None and isinstance(parsing_error, Exception):
        return None, raw_message, parsing_error
    if parsing_error is not None:
        return None, raw_message, RuntimeError(str(parsing_error))

    if isinstance(parsed, PlannerOutput):
        return parsed, raw_message, None

    try:
        return PlannerOutput.validate_input(parsed), raw_message, None
    except Exception as exc:
        return None, raw_message, exc


def _resolve_planner_strategy(
    *,
    llm_planner: Any,
    state: GraphState,
    context: PlannerRunContext,
    max_turns: int,
) -> tuple[PlannerDecision, list[str], list[LLMCallMetadata]]:
    planner_errors: list[str] = []
    llm_calls: list[LLMCallMetadata] = []

    if context.planner_mode != "force_llm":
        deterministic = build_deterministic_planner_decision(
            user_input=context.user_input,
            has_retriever=context.has_retriever,
        )
        if deterministic is not None:
            return deterministic, planner_errors, llm_calls

    try:
        planner_raw = llm_planner.invoke(build_planner_messages(state, max_turns=max_turns))
        planner_output, raw_message, parse_error = _coerce_structured_planner_result(planner_raw)
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
                    ),
                    status="llm",
                ),
                planner_errors,
                llm_calls,
            )
        planner_errors.append(f"planner: output validation failed ({parse_error})")
    except Exception as exc:
        planner_errors.append(f"planner: structured output invocation failed ({exc})")

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


def make_planner_node(llm_planner: Any, verbose: bool, max_turns: int = 6):
    def planner(state: GraphState) -> GraphState:
        runtime = get_runtime_state(state)
        retrieval = get_retrieval_state(state)
        debug = get_debug_state(state)
        existing_retry_context = get_retry_state(state)
        context = PlannerRunContext(
            user_input=runtime.user_input,
            has_retriever=bool(runtime.retriever),
            planner_attempt=int(existing_retry_context.attempt) + 1,
            planner_mode=runtime.planner_mode,
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
