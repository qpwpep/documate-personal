from __future__ import annotations

import json
from typing import Annotated, Any

from langchain_core.messages import AnyMessage, ToolMessage
from langgraph.graph import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from ..answer_schema import AgentResponsePayloadModel, SynthesisOutput, build_empty_response_payload
from ..planner_schema import PlannerOutput
from .debug import (
    DebugPayload,
    PlannerDiagnostic,
    PlannerStatus,
    RetryState,
    LLMCallMetadata,
    RetrievalDiagnostic,
    coerce_debug_payload,
    empty_planner_diagnostic,
)


class _ModelAccessMixin:
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        setattr(self, key, value)

    def values(self):
        return self.model_dump(mode="json").values()

    def items(self):
        return self.model_dump(mode="json").items()

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, dict):
            return self.model_dump(mode="json") == other
        return super().__eq__(other)


class SlackDestination(_ModelAccessMixin, BaseModel):
    channel_id: str | None = None
    user_id: str | None = None
    email: str | None = None


class SessionMetadata(_ModelAccessMixin, BaseModel):
    slack_destination: SlackDestination | None = None


class RuntimeState(_ModelAccessMixin, BaseModel):
    user_input: str = ""
    retriever: Any | None = None
    session_metadata: SessionMetadata = Field(default_factory=SessionMetadata)
    memory_summary: str | None = None


class PlannerState(_ModelAccessMixin, BaseModel):
    output: PlannerOutput = Field(default_factory=PlannerOutput.fallback)
    status: PlannerStatus = "llm"
    diagnostics: PlannerDiagnostic = Field(default_factory=empty_planner_diagnostic)
    guided_followup: str | None = None


class RetrievalState(_ModelAccessMixin, BaseModel):
    evidence_log: list[dict[str, Any]] = Field(default_factory=list)


class ResponseState(_ModelAccessMixin, BaseModel):
    final_answer: str = ""
    payload: AgentResponsePayloadModel = Field(default_factory=build_empty_response_payload)
    synthesis_output: SynthesisOutput = Field(default_factory=SynthesisOutput)
    synthesis_attempt: int = 0


class DebugState(DebugPayload):
    planner_errors: list[str] = Field(default_factory=list)
    retrieval_errors: list[str] = Field(default_factory=list)
    synthesis_errors: list[str] = Field(default_factory=list)
    validation_errors: list[str] = Field(default_factory=list)
    action_errors: list[str] = Field(default_factory=list)
    retrieval_diagnostics: list[RetrievalDiagnostic] = Field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = Field(default_factory=list)
    latency_trace: list[dict[str, Any]] = Field(default_factory=list)


class GraphState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    runtime: RuntimeState
    planner: PlannerState
    retrieval: RetrievalState
    retry: RetryState
    response: ResponseState
    debug: DebugState


def safe_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []


def slice_from_index(items: list[Any], start_index: int) -> list[Any]:
    if start_index < 0:
        start_index = 0
    if start_index >= len(items):
        return []
    return items[start_index:]


def build_tool_message(tool_name: str, payload: Any, index: int) -> ToolMessage:
    return ToolMessage(
        content=json.dumps(payload, ensure_ascii=False),
        name=tool_name,
        tool_call_id=f"{tool_name}-{index}",
    )


def coerce_planner_output(raw: Any, errors: list[str]) -> PlannerOutput:
    if isinstance(raw, PlannerOutput):
        return raw
    try:
        return PlannerOutput.model_validate(raw)
    except Exception as exc:
        errors.append(f"planner: output validation failed ({exc})")
        return PlannerOutput.fallback()


def empty_slack_destination() -> SlackDestination:
    return SlackDestination()


def coerce_slack_destination(value: Any) -> SlackDestination:
    if isinstance(value, SlackDestination):
        return value

    destination = empty_slack_destination()
    if not isinstance(value, dict):
        return destination

    for key in ("channel_id", "user_id", "email"):
        raw_item = value.get(key)
        if raw_item is None:
            setattr(destination, key, None)
            continue
        text = str(raw_item).strip()
        setattr(destination, key, text or None)
    return destination


def coerce_session_metadata(value: Any) -> SessionMetadata:
    if isinstance(value, SessionMetadata):
        return value
    if not isinstance(value, dict):
        return SessionMetadata()
    destination = coerce_slack_destination(value.get("slack_destination"))
    if any((destination.channel_id, destination.user_id, destination.email)):
        return SessionMetadata(slack_destination=destination)
    return SessionMetadata()


def coerce_runtime_state(value: Any) -> RuntimeState:
    if isinstance(value, RuntimeState):
        return value
    if not isinstance(value, dict):
        return RuntimeState()
    return RuntimeState(
        user_input=str(value.get("user_input", "") or ""),
        retriever=value.get("retriever"),
        session_metadata=coerce_session_metadata(value.get("session_metadata")),
        memory_summary=(
            str(value.get("memory_summary")).strip()
            if value.get("memory_summary") is not None
            else None
        ),
    )


def coerce_planner_state(value: Any) -> PlannerState:
    if isinstance(value, PlannerState):
        return value
    if not isinstance(value, dict):
        return PlannerState()
    planner_errors: list[str] = []
    diagnostics = value.get("diagnostics")
    status = value.get("status")
    if status not in {"llm", "deterministic", "heuristic_fallback", "fallback_no_routes"}:
        status = "llm"
    return PlannerState(
        output=coerce_planner_output(value.get("output"), planner_errors),
        status=status,
        diagnostics=PlannerDiagnostic.model_validate(diagnostics)
        if isinstance(diagnostics, dict)
        else empty_planner_diagnostic(status=status),
        guided_followup=(
            str(value.get("guided_followup")).strip()
            if value.get("guided_followup") is not None
            else None
        ),
    )


def coerce_retrieval_state(value: Any) -> RetrievalState:
    if isinstance(value, RetrievalState):
        return value
    if not isinstance(value, dict):
        return RetrievalState()
    evidence_log = value.get("evidence_log")
    return RetrievalState(
        evidence_log=[
            item
            for item in safe_list(evidence_log)
            if isinstance(item, dict)
        ]
    )


def coerce_response_state(value: Any) -> ResponseState:
    if isinstance(value, ResponseState):
        return value
    if not isinstance(value, dict):
        return ResponseState()

    raw_payload = value.get("payload")
    if isinstance(raw_payload, AgentResponsePayloadModel):
        payload = raw_payload
    else:
        try:
            payload = AgentResponsePayloadModel.model_validate(raw_payload)
        except Exception:
            payload = build_empty_response_payload(answer="")

    raw_synthesis_output = value.get("synthesis_output")
    if isinstance(raw_synthesis_output, SynthesisOutput):
        synthesis_output = raw_synthesis_output
    else:
        try:
            synthesis_output = SynthesisOutput.model_validate(raw_synthesis_output)
        except Exception:
            synthesis_output = SynthesisOutput()

    synthesis_attempt = value.get("synthesis_attempt", 0)
    try:
        synthesis_attempt_int = int(synthesis_attempt or 0)
    except (TypeError, ValueError):
        synthesis_attempt_int = 0

    return ResponseState(
        final_answer=str(value.get("final_answer", "") or ""),
        payload=payload,
        synthesis_output=synthesis_output,
        synthesis_attempt=max(0, synthesis_attempt_int),
    )


def coerce_debug_state(value: Any) -> DebugState:
    if isinstance(value, DebugState):
        return value
    if not isinstance(value, dict):
        return DebugState()

    payload = coerce_debug_payload(value)
    return DebugState.model_validate(payload.model_dump(mode="json"))


def build_graph_state_input(
    *,
    user_input: str,
    messages: list[AnyMessage] | None = None,
    retriever: Any | None = None,
    session_metadata: SessionMetadata | dict[str, Any] | None = None,
    memory_summary: str | None = None,
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
            retriever=retriever,
            session_metadata=coerce_session_metadata(session_metadata),
            memory_summary=memory_summary,
        ),
    }
    if planner is not None:
        state["planner"] = coerce_planner_state(planner)
    if retrieval is not None:
        state["retrieval"] = coerce_retrieval_state(retrieval)
    if retry is not None:
        state["retry"] = RetryState.model_validate(retry)
    if response is not None:
        state["response"] = coerce_response_state(response)
    if debug is not None:
        state["debug"] = coerce_debug_state(debug)
    return normalize_state_updates(state)


def runtime_state(state: dict[str, Any]) -> RuntimeState:
    return coerce_runtime_state(state.get("runtime"))


def planner_state(state: dict[str, Any]) -> PlannerState:
    return coerce_planner_state(state.get("planner"))


def retrieval_state(state: dict[str, Any]) -> RetrievalState:
    return coerce_retrieval_state(state.get("retrieval"))


def retry_state(state: dict[str, Any]) -> RetryState:
    return RetryState.model_validate(state.get("retry") or {})


def response_state(state: dict[str, Any]) -> ResponseState:
    return coerce_response_state(state.get("response"))


def debug_state(state: dict[str, Any]) -> DebugState:
    return coerce_debug_state(state.get("debug"))


def normalize_state_updates(updates: Any) -> Any:
    if not isinstance(updates, dict):
        return updates

    normalized = dict(updates)
    if "runtime" in normalized:
        normalized["runtime"] = coerce_runtime_state(normalized.get("runtime"))
    if "planner" in normalized:
        normalized["planner"] = coerce_planner_state(normalized.get("planner"))
    if "retrieval" in normalized:
        normalized["retrieval"] = coerce_retrieval_state(normalized.get("retrieval"))
    if "retry" in normalized:
        normalized["retry"] = RetryState.model_validate(normalized.get("retry") or {})
    if "response" in normalized:
        normalized["response"] = coerce_response_state(normalized.get("response"))
    if "debug" in normalized:
        normalized["debug"] = coerce_debug_state(normalized.get("debug"))
    if "messages" in normalized and not isinstance(normalized.get("messages"), list):
        normalized["messages"] = []
    return normalized
