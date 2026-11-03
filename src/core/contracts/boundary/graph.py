from __future__ import annotations

from typing import Any
from uuid import uuid4

from langchain_core.messages import AnyMessage
from src.core.answer_schema import AnswerResponse
from src.core.save_contract import SaveOperation

from src.core.contracts.debug import RetryState
from src.core.contracts.retrieval import UploadRetriever
from src.core.contracts.graph_state import DebugState, GraphState, PendingAction, PlannerState, ResponseState, RetrievalState, RuntimeState, SessionMetadata
from src.core.request_contracts import RequestContract, UserTurnSnapshot
from src.core.uploads import UploadFileInfo


def build_graph_state_input(
    *,
    user_input: str,
    session_id: str = "",
    current_turn_id: str | None = None,
    user_turns: tuple[UserTurnSnapshot, ...] = (),
    messages: list[AnyMessage] | None = None,
    retriever: UploadRetriever | None = None,
    upload_files: tuple[UploadFileInfo, ...] = (),
    progress_emitter: Any | None = None,
    session_metadata: SessionMetadata | None = None,
    memory_summary: str | None = None,
    previous_response: AnswerResponse | None = None,
    request_contract: RequestContract | None = None,
    pending_action: PendingAction | None = None,
    answer_export_operations: dict[str, SaveOperation] | None = None,
    planner: PlannerState | None = None,
    retrieval: RetrievalState | None = None,
    retry: RetryState | None = None,
    response: ResponseState | None = None,
    debug: DebugState | None = None,
) -> GraphState:
    state: GraphState = {
        "messages": list(messages or []),
        "route_decisions": [],
        "runtime": RuntimeState(
            session_id=session_id,
            user_input=user_input,
            current_turn_id=current_turn_id or f"user:{uuid4().hex}",
            user_turns=user_turns,
            retriever=retriever,
            upload_files=upload_files,
            session_metadata=session_metadata if session_metadata is not None else SessionMetadata(),
            memory_summary=memory_summary,
            progress_emitter=progress_emitter,
            previous_response=previous_response,
            request_contract=request_contract,
            pending_action=pending_action,
            answer_export_operations=answer_export_operations or {},
        ),
    }
    if planner is not None:
        state["planner"] = planner
    if retrieval is not None:
        state["retrieval"] = retrieval
    if retry is not None:
        state["retry"] = retry
    if response is not None:
        state["response"] = response
    if debug is not None:
        state["debug"] = debug
    return state


def get_retry_state(state: GraphState) -> RetryState:
    return state["retry"] if "retry" in state else RetryState()
