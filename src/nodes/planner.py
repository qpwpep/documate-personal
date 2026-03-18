from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..logging_utils import log_event
from ..planner_schema import PlannerOutput, RetrievalTask
from ..prompts import (
    has_explicit_docs_intent,
    has_explicit_local_intent,
    needs_rag,
    needs_save,
    needs_search,
    needs_slack,
)
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

_UPLOAD_KEYWORDS = (
    "upload",
    "uploaded",
    "current file",
    "current notebook",
    "this file",
    "this notebook",
    ".ipynb",
    ".py",
    "\uc5c5\ub85c\ub4dc",
    "\uc5c5\ub85c\ub4dc\ud55c",
    "\ud604\uc7ac \ud30c\uc77c",
    "\uc774 \ud30c\uc77c",
    "\uc774 \ub178\ud2b8\ubd81",
)
_AUXILIARY_MARKERS = (
    "\ucd94\uac00 \uc870\uac74",
    "\ucd94\uac00 \uc9c0\uc2dc",
    "\ubd80\uac00 \uc870\uac74",
    "additional condition",
    "additional instruction",
    "extra condition",
    "extra instruction",
)
_ACTION_CLAUSE_PATTERN = re.compile(
    r"(?i)\b(save|export|write|download|slack|dm|channel|share|send)\b|\uc800\uc7a5|\uc2ac\ub799|\uacf5\uc720|\uc804\uc1a1|\ud14d\uc2a4\ud2b8\s*\ud30c\uc77c"
)
_DOCS_CLAUSE_PATTERN = re.compile(
    r"(?i)\b(official|docs?|documentation|reference|manual|api|latest)\b|\uacf5\uc2dd|\ubb38\uc11c|\ub808\ud37c\ub7f0\uc2a4|\ucc38\uace0\s*\uc790\ub8cc|\ucd5c\uc2e0"
)
_COMPARE_CLAUSE_PATTERN = re.compile(
    r"(?i)\b(compare|comparison|versus|vs\.?|along with|together)\b|\ube44\uad50|\ud568\uaed8|\uac19\uc774"
)
_TRAILING_DOCS_STOP_PHRASES = (
    "\uacf5\uc2dd \ubb38\uc11c \uae30\uc900\uc73c\ub85c",
    "\uacf5\uc2dd \ub808\ud37c\ub7f0\uc2a4 \uae30\uc900\uc73c\ub85c",
    "\uacf5\uc2dd \ubb38\uc11c\ub85c",
    "\uacf5\uc2dd \ub808\ud37c\ub7f0\uc2a4\ub85c",
    "based on official docs",
    "according to official docs",
    "\uacf5\uc2dd \ubb38\uc11c",
    "\uacf5\uc2dd \ub808\ud37c\ub7f0\uc2a4",
    "\uc124\uba85\ud558\uace0",
    "\uc124\uba85\ud574\uc918",
    "\uc54c\ub824\uc918",
    "\ubcf4\uc5ec\uc918",
    "\ucc3e\uc544\uc918",
    "\uc694\uc57d\ud574\uc918",
    "\uc815\ub9ac\ud574\uc918",
)
_DOCS_IDENTIFIER_PATTERN = re.compile(r"\b(?:[A-Za-z][A-Za-z0-9._-]*|v\d+)\b")
_DOCS_IDENTIFIER_STOPWORDS = {
    "from",
    "official",
    "docs",
    "doc",
    "documentation",
    "reference",
    "with",
    "the",
    "it",
    "and",
}


def _normalize_query_text(text: str) -> str:
    collapsed = " ".join(str(text or "").replace("\r", "\n").split())
    return collapsed.strip(" ,.;:-")


def has_upload_route_intent(user_input: str) -> bool:
    lowered = str(user_input or "").lower()
    return bool(lowered.strip()) and any(keyword in lowered for keyword in _UPLOAD_KEYWORDS)


def needs_upload_followup(user_input: str) -> bool:
    return has_upload_route_intent(user_input)


def is_upload_only_request(user_input: str) -> bool:
    return has_upload_route_intent(user_input) and not has_explicit_docs_intent(user_input)


def _strip_auxiliary_clauses(text: str) -> str:
    normalized = str(text or "")
    lowered = normalized.lower()
    cut_index = len(normalized)
    for marker in _AUXILIARY_MARKERS:
        index = lowered.find(marker.lower())
        if index >= 0:
            cut_index = min(cut_index, index)
    normalized = normalized[:cut_index]
    if "\n" in normalized:
        normalized = normalized.split("\n", 1)[0]
    return _normalize_query_text(normalized)


def _strip_action_clauses(text: str) -> str:
    parts = re.split(r"(?<=[?.!,])|\band\b|\uadf8\ub9ac\uace0|\ub610\ub294", str(text or ""), flags=re.I)
    kept = [part.strip() for part in parts if part.strip() and not _ACTION_CLAUSE_PATTERN.search(part)]
    return _normalize_query_text(" ".join(kept or [str(text or "")]))


def _compact_docs_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    compact = re.sub(_COMPARE_CLAUSE_PATTERN, " ", compact)
    upload_match = re.search(
        r"(?i)(upload(?:ed)?|current file|current notebook|this file|this notebook|\.ipynb|\.py|\uc5c5\ub85c\ub4dc|\ud604\uc7ac \ud30c\uc77c|\uc774 \ud30c\uc77c|\uc774 \ub178\ud2b8\ubd81)",
        compact,
    )
    if upload_match:
        compact = compact[: upload_match.start()]
    for phrase in _TRAILING_DOCS_STOP_PHRASES:
        compact = re.sub(re.escape(phrase), " ", compact, flags=re.I)
    compact = re.sub(r"(?i)\b(official docs?|official documentation)\b", " ", compact)
    compact = re.sub(
        r"(?i)\b(explain|describe|summarize|show|find|tell)\b|\uc124\uba85|\uc694\uc57d|\uc815\ub9ac|\ucc3e\uc544|\ubcf4\uc5ec",
        " ",
        compact,
    )
    compact = re.sub(r"(?i)\b(and|it with the|with the)\b", " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    compact = _normalize_query_text(compact)

    identifier_tokens = _DOCS_IDENTIFIER_PATTERN.findall(compact)
    if len(identifier_tokens) >= 2:
        deduped: list[str] = []
        for token in identifier_tokens:
            if token.lower() in _DOCS_IDENTIFIER_STOPWORDS:
                continue
            if token not in deduped:
                deduped.append(token)
        return " ".join(deduped[:4])

    return compact


def _compact_upload_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    upload_match = re.search(
        r"(?i)(upload(?:ed)?|current file|current notebook|this file|this notebook|\.ipynb|\.py|\uc5c5\ub85c\ub4dc|\ud604\uc7ac \ud30c\uc77c|\uc774 \ud30c\uc77c|\uc774 \ub178\ud2b8\ubd81)",
        compact,
    )
    if upload_match:
        compact = compact[upload_match.start() :]
    compact = re.sub(_DOCS_CLAUSE_PATTERN, " ", compact)
    compact = re.sub(_COMPARE_CLAUSE_PATTERN, " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    return _normalize_query_text(compact)


def _compact_local_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    compact = re.sub(_DOCS_CLAUSE_PATTERN, " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    return _normalize_query_text(compact)


def sanitize_retrieval_query(
    *,
    route: str,
    query: str,
    retry_context: RetryContext | None = None,
) -> str:
    base_query = _normalize_query_text(query)
    if route == "docs":
        sanitized = _compact_docs_query(base_query)
        retry_reason = str((retry_context or {}).get("retry_reason") or "")
        if retry_reason == "no_evidence":
            sanitized = re.sub(
                r"(?i)\b(why|how|explain|describe|summarize)\b|\uc124\uba85|\uc694\uc57d|\uc774\uc720|\uc8fc\uc758\uc810",
                " ",
                sanitized,
            )
            sanitized = _normalize_query_text(re.sub(r"\s+", " ", sanitized))
    elif route == "upload":
        sanitized = _compact_upload_query(base_query)
    else:
        sanitized = _compact_local_query(base_query)
    return sanitized or base_query


def sanitize_planner_output_queries(
    planner_output: PlannerOutput,
    *,
    user_input: str,
    retry_context: RetryContext | None = None,
) -> PlannerOutput:
    if not planner_output.use_retrieval or not planner_output.tasks:
        return planner_output
    sanitized_tasks = [
        RetrievalTask(
            route=task.route,
            query=sanitize_retrieval_query(
                route=task.route,
                query=task.query or user_input,
                retry_context=retry_context,
            ),
            k=task.k,
        )
        for task in planner_output.tasks
    ]
    return PlannerOutput(use_retrieval=True, tasks=sanitized_tasks)


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


def _build_deterministic_routes(
    *,
    user_input: str,
    has_retriever: bool,
) -> tuple[PlannerOutput, PlannerDiagnostic, str | None] | None:
    docs_intent = has_explicit_docs_intent(user_input)
    upload_intent = has_upload_route_intent(user_input)
    local_intent = has_explicit_local_intent(user_input) and not docs_intent and not upload_intent
    action_only = is_action_only_request(user_input) or (
        (needs_save(user_input) or needs_slack(user_input))
        and not docs_intent
        and not upload_intent
        and not local_intent
    )

    if action_only:
        return (
            PlannerOutput.fallback(),
            normalize_planner_diagnostics(
                status="deterministic",
                reason="action_only",
                fallback_routes=[],
            ),
            None,
        )

    routes: list[str] = []
    if docs_intent and upload_intent:
        routes = ["docs", "upload"]
    elif upload_intent:
        routes = ["upload"]
    elif docs_intent:
        routes = ["docs"]
    elif local_intent:
        routes = ["local"]
    else:
        return None

    if "upload" in routes and not has_retriever:
        return (
            PlannerOutput.fallback(),
            normalize_planner_diagnostics(
                status="deterministic",
                reason="upload_retriever_missing",
                fallback_routes=[],
                intent_required=True,
                required_routes=routes,
                override_applied=True,
                override_reason="upload_retriever_missing",
            ),
            build_missing_upload_followup(),
        )

    planner_output = PlannerOutput(
        use_retrieval=True,
        tasks=[RetrievalTask(route=route, query=user_input.strip(), k=4) for route in routes],
    )
    planner_output = sanitize_planner_output_queries(planner_output, user_input=user_input)
    return (
        planner_output,
        normalize_planner_diagnostics(
            status="deterministic",
            reason=None,
            fallback_routes=routes,
            intent_required=True,
            required_routes=routes,
        ),
        None,
    )


def build_heuristic_planner_output(
    *,
    user_input: str,
    has_retriever: bool,
) -> tuple[PlannerOutput, PlannerDiagnostic, str | None]:
    trimmed_query = str(user_input or "").strip()
    routes: list[str] = []
    upload_route_intent = has_upload_route_intent(user_input)
    explicit_docs_intent = has_explicit_docs_intent(user_input)
    explicit_local_intent = has_explicit_local_intent(user_input)
    guided_followup: str | None = None

    if (upload_route_intent and explicit_docs_intent) or (
        not upload_route_intent and needs_search(user_input)
    ):
        routes.append("docs")

    if has_retriever and upload_route_intent:
        routes.append("upload")
    elif upload_route_intent and needs_upload_followup(user_input):
        guided_followup = build_missing_upload_followup()

    if explicit_local_intent and not explicit_docs_intent and not upload_route_intent:
        routes.append("local")

    unique_routes = [route for route in ROUTE_ORDER if route in routes]
    if unique_routes:
        planner_output = PlannerOutput(
            use_retrieval=True,
            tasks=[RetrievalTask(route=route, query=trimmed_query, k=4) for route in unique_routes],
        )
        planner_output = sanitize_planner_output_queries(planner_output, user_input=trimmed_query)
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

    upload_route_intent = has_upload_route_intent(trimmed)
    docs_route_intent = has_explicit_docs_intent(trimmed) if upload_route_intent else needs_search(trimmed)
    local_route_intent = has_explicit_local_intent(trimmed) and not docs_route_intent and not upload_route_intent

    routes: list[str] = []
    if docs_route_intent:
        routes.append("docs")
    if upload_route_intent:
        routes.append("upload")
    elif local_route_intent:
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
        override_applied=bool(planner_diagnostics.get("override_applied", False)),
        override_reason=planner_diagnostics.get("override_reason"),
    )

    if not required_routes:
        return planner_output, diagnostics, None

    if "upload" in required_routes and not has_retriever:
        diagnostics["reason"] = "upload_retriever_missing"
        diagnostics["override_applied"] = True
        diagnostics["override_reason"] = "upload_retriever_missing"
        return PlannerOutput.fallback(), diagnostics, build_missing_upload_followup()

    upload_only = is_upload_only_request(user_input)
    existing_tasks = {task.route: task for task in planner_output.tasks}
    if upload_only:
        required_route_set = set(required_routes)
        existing_tasks = {
            route: task for route, task in existing_tasks.items() if route in required_route_set
        }
    existing_routes = {task.route for task in planner_output.tasks} if planner_output.use_retrieval else set()
    if upload_only:
        existing_routes = {route for route in existing_routes if route in set(required_routes)}
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
        if upload_only and route not in required_routes:
            continue
        existing_task = existing_tasks.get(route)
        if existing_task is not None:
            merged_tasks.append(existing_task)
            continue
        if route in required_routes:
            merged_tasks.append(RetrievalTask(route=route, query=str(user_input).strip(), k=4))

    sanitized_output = sanitize_planner_output_queries(
        PlannerOutput(use_retrieval=True, tasks=merged_tasks),
        user_input=user_input,
    )
    return sanitized_output, diagnostics, None


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
    latest_conversation: list[BaseMessage] = []
    latest_human_index = -1
    for index in range(len(conversation) - 1, -1, -1):
        if isinstance(conversation[index], HumanMessage):
            latest_human_index = index
            break
    if latest_human_index >= 0:
        latest_conversation.append(conversation[latest_human_index])
    else:
        latest_conversation = conversation[-1:]
    model_messages.extend(latest_conversation)

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

        deterministic = _build_deterministic_routes(
            user_input=user_input,
            has_retriever=has_retriever,
        )
        if deterministic is not None:
            planner_output, planner_diagnostics, guided_followup = deterministic
            planner_status = "deterministic"
        elif is_action_only_request(user_input):
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
        planner_output = sanitize_planner_output_queries(
            planner_output,
            user_input=user_input,
            retry_context=existing_retry_context,
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
            retry_context["failed_routes"] = []
            retry_context["preserved_evidence"] = []
            retry_context["preserved_retrieval_diagnostics"] = []

        updates: State = {
            "planner_output": planner_output,
            "planner_status": planner_status,
            "planner_diagnostics": planner_diagnostics,
            "guided_followup": guided_followup,
            "synthesis_attempt": int(state.get("synthesis_attempt", 0)),
            "needs_retry": False,
            "retry_context": retry_context,
        }
        if planner_errors:
            updates["planner_errors"] = planner_errors
        if llm_calls:
            updates["llm_calls"] = llm_calls
        return updates

    return planner
