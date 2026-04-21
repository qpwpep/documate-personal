import time
from typing import Any

from .contracts import GraphState
from .contracts.boundary.debug import get_debug_state, parse_debug_state
from .contracts.boundary.graph import get_retry_state, normalize_graph_update
from .contracts.boundary.planner import get_planner_state
from .contracts.boundary.response import get_response_state
from .contracts.boundary.runtime import get_runtime_state
from .latency import elapsed_ms, make_stage_latency_event
from .llm import build_llm_registry
from .make_graph import build_graph
from .nodes.actions import make_action_postprocess_node
from .nodes.planner import make_planner_node
from .nodes.retrieval import make_retrieve_dispatch_node
from .nodes.session import add_user_message, make_summarize_node
from .nodes.synthesis import make_synthesize_node
from .nodes.validation import make_validate_evidence_node
from .settings import AppSettings, get_settings
from .tools import build_tool_registry


class StageExecutionError(RuntimeError):
    def __init__(self, *, stage: str, latency_ms: int, cause: Exception):
        super().__init__(str(cause))
        self.stage = stage
        self.latency_ms = latency_ms
        self.cause = cause


def _resolve_stage_attempt_for_start(stage: str, state: GraphState) -> int:
    if stage in {"planner", "retrieval"}:
        retry_context = get_retry_state(state)
        return int(retry_context.attempt) + 1
    if stage == "synthesis":
        return max(1, int(get_response_state(state).synthesis_attempt or 0) + 1)
    if stage == "validation":
        return max(1, int(get_response_state(state).synthesis_attempt or 0))
    if stage == "action_postprocess":
        return max(1, int(get_response_state(state).synthesis_attempt or 1))
    return 1


def _resolve_stage_status(stage: str, updates: GraphState) -> str | None:
    if stage == "planner":
        status = get_planner_state(updates).status
        return str(status) if status else None
    if stage == "validation":
        return "retry" if get_retry_state(updates).needs_retry else "pass"
    return None


def _extract_stage_status_from_debug(stage: str, debug: Any, attempt: int) -> str | None:
    latency_trace = getattr(debug, "latency_trace", None)
    if not isinstance(latency_trace, list):
        return None

    for item in reversed(latency_trace):
        if not isinstance(item, dict):
            continue
        if str(item.get("kind") or "").strip() != "stage":
            continue
        if str(item.get("stage") or "").strip() != stage:
            continue
        item_attempt = item.get("attempt")
        try:
            if item_attempt is not None and int(item_attempt) != int(attempt):
                continue
        except (TypeError, ValueError):
            pass
        status = item.get("status")
        return str(status) if status else None
    return None


def _get_progress_emitter(state: GraphState) -> Any | None:
    return get_runtime_state(state).progress_emitter


def _normalize_node(node: Any):
    def wrapped(state: GraphState) -> GraphState:
        updates = node(state)
        if not isinstance(updates, dict):
            return updates
        return normalize_graph_update(updates)

    return wrapped


def _merge_debug_patch(state: GraphState, raw_debug_patch: Any):
    base_debug = get_debug_state(state)
    if raw_debug_patch is None:
        return base_debug
    if hasattr(raw_debug_patch, "model_dump"):
        return parse_debug_state(raw_debug_patch)
    if isinstance(raw_debug_patch, dict):
        return base_debug.model_copy(update=dict(raw_debug_patch))
    return base_debug


def _instrument_stage_node(stage: str, node: Any, *, record_latency_trace: bool = True):
    def wrapped(state: GraphState) -> GraphState:
        started = time.perf_counter()
        attempt = _resolve_stage_attempt_for_start(stage, state)
        progress_emitter = _get_progress_emitter(state)
        if progress_emitter is not None:
            progress_emitter.emit_stage_started(
                stage=stage,  # type: ignore[arg-type]
                attempt=attempt,
            )
        try:
            updates = node(state)
        except Exception as exc:
            if progress_emitter is not None:
                progress_emitter.emit_error(
                    message=str(exc),
                    stage=stage,  # type: ignore[arg-type]
                )
            raise StageExecutionError(
                stage=stage,
                latency_ms=elapsed_ms(started, time.perf_counter()),
                cause=exc,
            ) from exc
        if not isinstance(updates, dict):
            return updates

        raw_debug_patch = updates.get("debug") if "debug" in updates else None
        updates = normalize_graph_update(updates)
        debug = _merge_debug_patch(state, raw_debug_patch)
        stage_status = _resolve_stage_status(stage, updates) or _extract_stage_status_from_debug(
            stage,
            debug,
            attempt,
        )
        stage_latency_ms = elapsed_ms(started, time.perf_counter())
        if record_latency_trace:
            latency_event = make_stage_latency_event(
                stage=stage,  # type: ignore[arg-type]
                attempt=attempt,
                latency_ms=stage_latency_ms,
                status=stage_status,
            )
            updates["debug"] = debug.model_copy(
                update={"latency_trace": [*debug.latency_trace, latency_event]}
            )
        if progress_emitter is not None:
            progress_emitter.emit_stage_completed(
                stage=stage,  # type: ignore[arg-type]
                attempt=attempt,
                latency_ms=stage_latency_ms,
                status=stage_status,
            )
        return updates

    return wrapped


def build_agent_graph(settings: AppSettings | None = None):
    app_settings = settings or get_settings()
    has_default_slack_destination = bool(
        app_settings.slack_default_user_id or app_settings.slack_default_dm_email
    )

    tool_registry = build_tool_registry(app_settings)
    llm_registry = build_llm_registry(app_settings)

    summarize_node = make_summarize_node(
        llm_summarizer=llm_registry.llm_summarizer,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    summarize_node = _instrument_stage_node("summarize", summarize_node)
    planner_node = make_planner_node(
        llm_planner=llm_registry.llm_planner,
        verbose=llm_registry.verbose,
        max_turns=6,
    )
    planner_node = _instrument_stage_node("planner", planner_node)
    retrieve_dispatch_node = make_retrieve_dispatch_node(
        tavily_search_tool=tool_registry.tavily_search_tool,
        upload_search_tool=tool_registry.upload_search_tool,
        rag_search_tool=tool_registry.rag_search_tool,
        verbose=llm_registry.verbose,
    )
    retrieve_dispatch_node = _instrument_stage_node(
        "retrieval",
        _normalize_node(retrieve_dispatch_node),
        record_latency_trace=False,
    )
    synthesize_node = make_synthesize_node(
        llm_synthesizer=llm_registry.llm_synthesizer,
        llm_synthesizer_compact=llm_registry.llm_synthesizer_compact,
        verbose=llm_registry.verbose,
        max_turns=6,
        synthesis_max_tokens=app_settings.synthesis_max_tokens,
        prompt_snippet_char_limit=app_settings.synthesis_prompt_snippet_chars,
        has_default_slack_destination=has_default_slack_destination,
    )
    synthesize_node = _instrument_stage_node(
        "synthesis",
        _normalize_node(synthesize_node),
        record_latency_trace=False,
    )
    validate_evidence_node = make_validate_evidence_node(verbose=llm_registry.verbose)
    validate_evidence_node = _instrument_stage_node("validation", validate_evidence_node)
    action_postprocess_node = make_action_postprocess_node(
        save_text_tool=tool_registry.save_text_tool,
        slack_notify_tool=tool_registry.slack_notify_tool,
        verbose=llm_registry.verbose,
        has_default_slack_destination=has_default_slack_destination,
    )
    action_postprocess_node = _instrument_stage_node(
        "action_postprocess",
        action_postprocess_node,
    )
    add_user_node = _normalize_node(add_user_message)

    graph_object = build_graph(
        state_type=GraphState,
        add_user_node=add_user_node,
        summarize_node=summarize_node,
        planner_node=planner_node,
        retrieve_dispatch_node=retrieve_dispatch_node,
        synthesize_node=synthesize_node,
        validate_evidence_node=validate_evidence_node,
        action_postprocess_node=action_postprocess_node,
        summary_max_turns=6,
    )
    return graph_object
