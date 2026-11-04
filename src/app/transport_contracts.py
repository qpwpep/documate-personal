"""HTTP response and SSE contracts shared by the server and Python client."""
from __future__ import annotations

from functools import cache
from typing import Any, Literal, get_args

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field
from pydantic.json_schema import GenerateJsonSchema, models_json_schema
from pydantic_core import core_schema

from src.core.contracts.debug import DebugPayload
from src.core.answer_schema.wire import SerializedAnswer
from src.core.contracts.outcome import AgentCallResult, TurnResult
from src.core.latency import StageName


class HTTPResponseSchema(GenerateJsonSchema):
    """HTTP model dumps include defaults; model construction may still omit them."""

    def field_is_required(
        self,
        field: core_schema.ModelField | core_schema.DataclassField | core_schema.TypedDictField,
        total: bool,
    ) -> bool:
        if self.mode == "serialization" and field["type"] != "typed-dict-field":
            return field.get("serialization_exclude_if") is None
        return super().field_is_required(field, total)


class AgentResponse(AgentCallResult):
    model_config = ConfigDict(extra="forbid")

    response: SerializedAnswer | None = None
    trace: str
    debug: DebugPayload | None = None

    def to_turn_result(self) -> TurnResult:
        """Keep the decoded answer without the HTTP envelope or its serializer."""
        return TurnResult(
            status=self.status, response=self.response, answer_id=self.answer_id,
            message=self.message, problem=self.problem, execution_id=self.execution_id,
            missing_slots=self.missing_slots,
        )


AgentStreamEventName = Literal[
    "request_started",
    "stage_started",
    "stage_completed",
    "heartbeat",
    "progress_snapshot",
    "final_response",
    "error",
    "done",
]
class AgentStreamEvent(BaseModel):
    event: AgentStreamEventName
    data: dict[str, Any] = Field(default_factory=dict)


def _event_data_schema(required: list[str], **properties: Any) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_STREAM_STAGE_SCHEMA = {"type": "string", "enum": list(get_args(StageName))}
_STRING_SCHEMA = {"type": "string"}
_INTEGER_SCHEMA = {"type": "integer"}

# Each entry describes the JSON value in the named SSE frame's data field.
# final_response refers to the same runtime model used by response assembly.
AGENT_STREAM_EVENT_SCHEMAS = {
    "request_started": _event_data_schema(
        ["execution_id", "session_id"], execution_id=_STRING_SCHEMA, session_id=_STRING_SCHEMA,
    ),
    "stage_started": _event_data_schema(
        ["stage", "attempt"], stage=_STREAM_STAGE_SCHEMA, attempt=_INTEGER_SCHEMA,
    ),
    "stage_completed": _event_data_schema(
        ["stage", "attempt"], stage=_STREAM_STAGE_SCHEMA, attempt=_INTEGER_SCHEMA,
        latency_ms=_INTEGER_SCHEMA, status=_STRING_SCHEMA,
    ),
    "heartbeat": _event_data_schema(
        ["execution_id", "elapsed_ms"], execution_id=_STRING_SCHEMA, elapsed_ms=_INTEGER_SCHEMA,
    ),
    "progress_snapshot": _event_data_schema(
        ["stage", "summary"], stage=_STREAM_STAGE_SCHEMA, summary=_STRING_SCHEMA,
    ),
    "final_response": {"$ref": "#/components/schemas/AgentResponse"},
    "error": _event_data_schema(["message"], message=_STRING_SCHEMA, stage=_STREAM_STAGE_SCHEMA),
    "done": {"type": "object", "maxProperties": 0},
}


@cache
def agent_response_json_schema() -> dict[str, object]:
    """Build the same serialization contract as OpenAPI without creating an app."""
    _, schemas = models_json_schema(
        [(AgentResponse, "serialization")],
        ref_template="#/components/schemas/{model}",
        schema_generator=HTTPResponseSchema,
    )
    return {
        "$ref": "#/components/schemas/AgentResponse",
        "components": {"schemas": schemas["$defs"]},
    }


def parse_agent_event(event: str, data: dict[str, object]) -> AgentResponse | None:
    """Reject malformed wire data before domain models can supply defaults."""
    validator = _agent_event_validators().get(event)
    if validator is None:
        raise ValueError(f"Unknown agent SSE event: {event!r}")
    error = next(validator.iter_errors(data), None)
    if error is not None:
        raise ValueError(f"Invalid agent SSE event {event!r}: {error.message}") from error
    return AgentResponse.model_validate(data) if event == "final_response" else None


@cache
def _agent_event_validators() -> dict[str, Draft202012Validator]:
    response_schema = agent_response_json_schema()
    return {
        name: Draft202012Validator({**schema, "components": response_schema["components"]})
        for name, schema in AGENT_STREAM_EVENT_SCHEMAS.items()
    }
