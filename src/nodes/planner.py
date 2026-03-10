from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..logging_utils import log_event
from ..planner_schema import PlannerOutput, RetrievalTask
from ..prompts import needs_rag, needs_search
from .actions import is_action_only_request
from .retry import build_missing_upload_followup, format_retry_context_for_planner
from .session import keep_recent_messages
from .state import (
    DEFAULT_MAX_RETRIES,
    LLMCallMetadata,
    ROUTE_ORDER,
    PlannerDiagnostic,
    PlannerOverrideReason,
    PlannerStatus,
    RetryContext,
    State,
    build_llm_call_metadata,
    coerce_retry_context,
    safe_list,
)

PLANNER_SYS = (
    "You are a retrieval planner. Return a structured plan only.\n"
    "Rules:\n"
    "- Choose retrieval routes from: docs, upload, local.\n"
    "- docs: official/latest docs on the web.\n"
    "- upload: currently uploaded-file retriever context.\n"
    "- local: local notebook/vector index examples.\n"
    "- If retrieval is unnecessary, set use_retrieval=false and tasks=[].\n"
    "- If retrieval is needed, set use_retrieval=true and include 1-3 tasks.\n"
    "- Each selected route must appear at most once.\n"
    "- Keep each task.query short and route-specific.\n"
    "- If the request is only asking to save/share/send the current answer, retrieval is unnecessary.\n"
    "- If retriever_available=true and the user is asking about the currently uploaded file, prefer upload over local.\n"
    "- Do not include actions for save/slack; only retrieval planning."
)

logger = logging.getLogger(__name__)


def has_upload_route_intent(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if not lowered.strip():
        return False

    keywords = (
        "upload",
        "uploaded",
        "current file",
        "current notebook",
        "this file",
        "this notebook",
        ".ipynb",
        ".py",
        "업로드",
        "업로드한 파일",
        "업로드된",
        "현재 파일",
        "이 파일",
        "이 노트북",
    )
    return any(keyword in lowered for keyword in keywords)


def needs_upload_followup(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    if not lowered.strip():
        return False

    keywords = (
        "upload",
        "uploaded",
        "current file",
        "this file",
        "업로드",
        "업로드한 파일",
        "업로드된",
        "현재 파일",
        "이 파일",
    )
    return any(keyword in lowered for keyword in keywords)


def normalize_planner_diagnostics(
    *,
    status: PlannerStatus,
    reason: str | None = None,
    fallback_routes: list[str] | None = None,
    intent_required: bool = False,
    required_routes: list[str] | None = None,
    override_applied: bool = False,
    override_reason: PlannerOverrideReason | None = None,
) -> PlannerDiagnostic:
    return {
        "status": status,
        "reason": reason,
        "fallback_routes": list(fallback_routes or []),
        "intent_required": bool(intent_required),
        "required_routes": [route for route in ROUTE_ORDER if route in set(required_routes or [])],
        "override_applied": bool(override_applied),
        "override_reason": override_reason,
    }


def build_heuristic_planner_output(
    *,
    user_input: str,
    has_retriever: bool,
) -> tuple[PlannerOutput, PlannerDiagnostic, str | None]:
    trimmed_query = str(user_input or "").strip()
    routes: list[str] = []
    upload_route_intent = has_upload_route_intent(user_input)
    guided_followup: str | None = None

    if needs_search(user_input):
        routes.append("docs")

    if has_retriever and upload_route_intent:
        routes.append("upload")
    elif upload_route_intent and needs_upload_followup(user_input):
        guided_followup = build_missing_upload_followup()

    if needs_rag(user_input) and not upload_route_intent:
        routes.append("local")

    unique_routes = [route for route in ROUTE_ORDER if route in routes]
    if unique_routes:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route=route, query=trimmed_query, k=4) for route in unique_routes],
        )
        return (
            planner_output,
            normalize_planner_diagnostics(
                status="heuristic_fallback",
                reason="planner_failed_or_invalid",
                fallback_routes=unique_routes,
            ),
            guided_followup,
        )

    return (
        PlannerOutput.fallback(),
        normalize_planner_diagnostics(
            status="fallback_no_routes",
            reason="planner_failed_or_invalid",
            fallback_routes=[],
        ),
        guided_followup,
    )


def detect_required_routes(user_input: str) -> list[str]:
    trimmed = str(user_input or "").strip()
    if not trimmed:
        return []

    lowered = trimmed.lower()
    upload_route_intent = has_upload_route_intent(trimmed)
    docs_route_intent = needs_search(trimmed)
    if upload_route_intent:
        explicit_docs_keywords = (
            "official",
            "docs",
            "documentation",
            "reference",
            "manual",
            "api",
            "공식",
            "문서",
            "레퍼런스",
            "참고자료",
            "최신",
        )
        docs_route_intent = any(keyword in lowered for keyword in explicit_docs_keywords)

    routes: list[str] = []
    if docs_route_intent:
        routes.append("docs")
    if upload_route_intent:
        routes.append("upload")
    elif needs_rag(trimmed):
        routes.append("local")
    return [route for route in ROUTE_ORDER if route in routes]


def apply_required_route_guardrail(
    *,
    planner_output: PlannerOutput,
    planner_status: PlannerStatus,
    planner_diagnostics: PlannerDiagnostic,
    user_input: str,
    has_retriever: bool,
) -> tuple[PlannerOutput, PlannerDiagnostic, str | None]:
    required_routes = detect_required_routes(user_input)
    diagnostics = normalize_planner_diagnostics(
        status=planner_status,
        reason=planner_diagnostics.get("reason"),
        fallback_routes=planner_diagnostics.get("fallback_routes", []),
        intent_required=bool(required_routes),
        required_routes=required_routes,
        override_applied=False,
        override_reason=None,
    )

    if not required_routes:
        return planner_output, diagnostics, None

    if "upload" in required_routes and not has_retriever:
        diagnostics["reason"] = "upload_retriever_missing"
        diagnostics["override_applied"] = True
        diagnostics["override_reason"] = "upload_retriever_missing"
        return PlannerOutput.fallback(), diagnostics, build_missing_upload_followup()

    existing_tasks = {task.route: task for task in planner_output.tasks}
    existing_routes = {task.route for task in planner_output.tasks} if planner_output.use_retrieval else set()
    missing_required_routes = [route for route in required_routes if route not in existing_routes]

    override_reason: PlannerOverrideReason | None = None
    if required_routes and not planner_output.use_retrieval:
        override_reason = "missing_required_retrieval"
    elif missing_required_routes:
        override_reason = "missing_required_routes"

    if override_reason is None:
        return planner_output, diagnostics, None

    diagnostics["override_applied"] = True
    diagnostics["override_reason"] = override_reason
    if diagnostics.get("reason") is None:
        diagnostics["reason"] = override_reason

    merged_tasks: list[RetrievalTask] = []
    for route in ROUTE_ORDER:
        existing_task = existing_tasks.get(route)
        if existing_task is not None:
            merged_tasks.append(existing_task)
            continue
        if route in required_routes:
            merged_tasks.append(RetrievalTask(route=route, query=str(user_input).strip(), k=4))

    return PlannerOutput(use_retrieval=True, tasks=merged_tasks), diagnostics, None


def sanitize_planner_output(
    planner_output: PlannerOutput,
    *,
    has_retriever: bool,
    errors: list[str],
) -> PlannerOutput:
    tasks: list[RetrievalTask] = list(planner_output.tasks)
    if not has_retriever and any(task.route == "upload" for task in tasks):
        tasks = [task for task in tasks if task.route != "upload"]
        errors.append("planner: dropped upload route because retriever is unavailable")

    try:
        return PlannerOutput(
            use_retrieval=bool(planner_output.use_retrieval and tasks),
            tasks=tasks,
        )
    except Exception as exc:
        errors.append(f"planner: sanitized output validation failed ({exc})")
        return PlannerOutput.fallback()


def build_planner_messages(state: State, max_turns: int = 6) -> list[BaseMessage]:
    model_messages: list[BaseMessage] = [SystemMessage(content=PLANNER_SYS)]
    model_messages.append(
        SystemMessage(content=f"[Planner Context]\nretriever_available={bool(state.get('retriever'))}")
    )

    retry_context = coerce_retry_context(state.get("retry_context"))
    retry_context_message = format_retry_context_for_planner(state, retry_context)
    if retry_context_message:
        model_messages.append(SystemMessage(content=retry_context_message))

    if state.get("memory_summary"):
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{state['memory_summary']}"))

    conversation = [message for message in state.get("messages", []) if not isinstance(message, ToolMessage)]
    conversation = keep_recent_messages(conversation, max_turns=max_turns)
    model_messages.extend(conversation)

    if not any(isinstance(message, HumanMessage) for message in model_messages):
        model_messages.append(HumanMessage(content=str(state.get("user_input", "")).strip()))
    return model_messages


def _coerce_structured_planner_result(
    result: Any,
) -> tuple[PlannerOutput | None, AIMessage | None, Exception | None]:
    if isinstance(result, PlannerOutput):
        return result, None, None

    if not isinstance(result, dict):
        try:
            return PlannerOutput.model_validate(result), None, None
        except Exception as exc:
            return None, None, exc

    raw_message = result.get("raw")
    parsed = result.get("parsed")
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
        return PlannerOutput.model_validate(parsed), raw_message, None
    except Exception as exc:
        return None, raw_message, exc


def make_planner_node(llm_planner: Any, verbose: bool, max_turns: int = 6):
    def planner(state: State) -> State:
        planner_errors: list[str] = []
        llm_calls: list[LLMCallMetadata] = []
        existing_retry_context = coerce_retry_context(state.get("retry_context"))
        user_input = str(state.get("user_input", "") or "")
        has_retriever = bool(state.get("retriever"))
        planner_status: PlannerStatus = "llm"
        planner_diagnostics = normalize_planner_diagnostics(status="llm", reason=None, fallback_routes=[])
        guided_followup: str | None = None
        planner_attempt = int(existing_retry_context.get("attempt", 0)) + 1

        if is_action_only_request(user_input):
            planner_output = PlannerOutput.fallback()
        else:
            try:
                planner_raw = llm_planner.invoke(build_planner_messages(state, max_turns=max_turns))
                planner_output, raw_message, parse_error = _coerce_structured_planner_result(planner_raw)
                if raw_message is not None:
                    llm_calls.append(
                        build_llm_call_metadata(
                            stage="planner",
                            attempt=planner_attempt,
                            path="structured",
                            message=raw_message,
                        )
                    )
                if planner_output is None:
                    planner_errors.append(f"planner: output validation failed ({parse_error})")
                    planner_output, planner_diagnostics, guided_followup = build_heuristic_planner_output(
                        user_input=user_input,
                        has_retriever=has_retriever,
                    )
                    planner_status = planner_diagnostics["status"]
            except Exception as exc:
                planner_errors.append(f"planner: structured output invocation failed ({exc})")
                planner_output, planner_diagnostics, guided_followup = build_heuristic_planner_output(
                    user_input=user_input,
                    has_retriever=has_retriever,
                )
                planner_status = planner_diagnostics["status"]

        planner_output = sanitize_planner_output(
            planner_output,
            has_retriever=has_retriever,
            errors=planner_errors,
        )
        planner_output, planner_diagnostics, guardrail_followup = apply_required_route_guardrail(
            planner_output=planner_output,
            planner_status=planner_status,
            planner_diagnostics=planner_diagnostics,
            user_input=user_input,
            has_retriever=has_retriever,
        )
        if guardrail_followup:
            guided_followup = guardrail_followup
        if verbose:
            log_event(
                logger,
                logging.INFO,
                "planner",
                status=planner_status,
                use_retrieval=planner_output.use_retrieval,
                task_count=len(planner_output.tasks),
                required_routes=planner_diagnostics.get("required_routes", []),
                override=planner_diagnostics.get("override_applied", False),
            )

        retry_context: RetryContext = dict(existing_retry_context)
        retry_context["max_retries"] = int(
            existing_retry_context.get("max_retries", DEFAULT_MAX_RETRIES)
        )
        retry_context["evidence_start_index"] = len(safe_list(state.get("retrieved_evidence")))
        retry_context["retrieval_error_start_index"] = len(safe_list(state.get("retrieval_errors")))
        retry_context["retrieval_diagnostic_start_index"] = len(
            safe_list(state.get("retrieval_diagnostics"))
        )
        if int(retry_context.get("attempt", 0)) <= 0:
            retry_context["retrieval_feedback"] = ""
            retry_context["score_avg"] = None
            retry_context.pop("retry_reason", None)

        updates: State = {
            "planner_output": planner_output,
            "planner_status": planner_status,
            "planner_diagnostics": planner_diagnostics,
            "guided_followup": guided_followup,
            "retrieval_errors": planner_errors,
            "synthesis_attempt": int(state.get("synthesis_attempt", 0)),
            "needs_retry": False,
            "retry_context": retry_context,
        }
        if llm_calls:
            updates["llm_calls"] = llm_calls
        return updates

    return planner
