from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import AnyMessage
from src.core.answer_schema import AnswerResponse

from src.core.contracts.debug import RetryState
from src.core.contracts.graph_state import DebugState, GraphState, PendingAction, PlannerState, ResponseState, RetrievalState, RuntimeState, SessionMetadata
from src.core.request_contracts import RequestContract, UserTurnSnapshot
from src.core.uploads import UploadFileInfo
from src.core.contracts.boundary.debug import parse_debug_state, parse_retry_state
from src.core.contracts.boundary.planner import parse_planner_state
from src.core.contracts.boundary.response import parse_response_state
from src.core.contracts.boundary.retrieval import parse_retrieval_state
from src.core.contracts.boundary.runtime import parse_request_contract, parse_runtime_state, parse_session_metadata


def build_graph_state_input(
    *,
    user_input: str,
    current_turn_id: str | None = None,
    user_turns: tuple[UserTurnSnapshot, ...] = (),
    messages: list[AnyMessage] | None = None,
    retriever: Any | None = None,
    upload_files: tuple[UploadFileInfo, ...] = (),
    progress_emitter: Any | None = None,
    session_metadata: SessionMetadata | dict[str, Any] | None = None,
    memory_summary: str | None = None,
    previous_response: AnswerResponse | None = None,
    request_contract: RequestContract | dict[str, Any] | None = None,
    pending_action: PendingAction | None = None,
    planner: PlannerState | dict[str, Any] | None = None,
    retrieval: RetrievalState | dict[str, Any] | None = None,
    retry: RetryState | dict[str, Any] | None = None,
    response: ResponseState | dict[str, Any] | None = None,
    debug: DebugState | dict[str, Any] | None = None,
) -> GraphState:
    state: GraphState = {
        "messages": list(messages or []),
        "runtime": RuntimeState(
            user_input=str(user_input or ""),
            current_turn_id=current_turn_id or f"user:{uuid4().hex}",
            user_turns=user_turns,
            retriever=retriever,
            upload_files=upload_files,
            session_metadata=parse_session_metadata(session_metadata),
            memory_summary=memory_summary,
            progress_emitter=progress_emitter,
            previous_response=previous_response,
            request_contract=parse_request_contract(request_contract),
            pending_action=pending_action,
        ),
    }
    if planner is not None:
        state["planner"] = parse_planner_state(planner)
    if retrieval is not None:
        state["retrieval"] = parse_retrieval_state(retrieval)
    if retry is not None:
        state["retry"] = parse_retry_state(retry)
    if response is not None:
        state["response"] = parse_response_state(response)
    if debug is not None:
        state["debug"] = parse_debug_state(debug)
    return normalize_graph_update(state)


def get_retry_state(state: dict[str, Any]) -> RetryState:
    return parse_retry_state(state.get("retry"))


def normalize_graph_update(updates: Any) -> Any:
    if not isinstance(updates, dict):
        return updates

    normalized = dict(updates)
    if "runtime" in normalized:
        normalized["runtime"] = parse_runtime_state(normalized.get("runtime"))
    if "planner" in normalized:
        normalized["planner"] = parse_planner_state(normalized.get("planner"))
    if "retrieval" in normalized:
        normalized["retrieval"] = parse_retrieval_state(normalized.get("retrieval"))
    if "retry" in normalized:
        normalized["retry"] = parse_retry_state(normalized.get("retry"))
    if "response" in normalized:
        normalized["response"] = parse_response_state(normalized.get("response"))
    if "debug" in normalized:
        normalized["debug"] = parse_debug_state(normalized.get("debug"))
    if "messages" in normalized and not isinstance(normalized.get("messages"), list):
        normalized["messages"] = []
    return normalized
