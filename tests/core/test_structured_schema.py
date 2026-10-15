from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from itertools import product
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError, create_model

from src.core.answer_schema import AnswerDocument
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.structured_schema import (
    SchemaCompilationError, compile_output_schema, schema_fingerprint,
)


def _schema_model(schema: dict[str, Any]) -> type[BaseModel]:
    class SchemaModel(BaseModel):
        @classmethod
        def model_json_schema(cls, **_kwargs: Any) -> dict[str, Any]:
            return deepcopy(schema)

    return SchemaModel


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "additionalProperties": False}


def _validate_wire(model: type[BaseModel], value: dict[str, Any]) -> None:
    schema = compile_output_schema(model)["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(value)




@pytest.mark.parametrize("body", [
    {"kind": "compose", "instruction": "Explain lists"},
    {"kind": "acknowledge"},
    {"kind": "copy_input", "source": {"turn_id": "u1", "quote": "hello", "occurrence": None}},
    {"kind": "transform_input", "source": {"turn_id": "u1", "quote": "hello", "occurrence": None}, "instruction": "Translate"},
    {"kind": "copy_answer", "source": {"ref": "previous"}},
    {"kind": "transform_answer", "source": {"ref": "pending"}, "instruction": "Shorten"},
    {"kind": "extract", "instruction": "Show the source", "requirement_ids": ["req1"]},
    {"kind": "unresolved", "question": "Which function?"},
])
def test_planner_body_variants_keep_their_canonical_meaning(body):
    plan = PlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract(
        slack_recipient={"state": "omitted"}, body=body,
    ))
    value = plan.model_dump(mode="json")
    _validate_wire(PlannerOutput, value)
    assert PlannerOutput.model_validate(value) == plan




def test_wire_requires_canonical_fields_without_making_nonnullable_defaults_nullable():
    class Defaults(BaseModel):
        model_config = ConfigDict(extra="forbid")
        enabled: bool = False
        label: str | None = None

    # Strict wire input deliberately differs from the domain's shorthand input.
    domain = Defaults.model_validate({})
    schema = compile_output_schema(Defaults)["schema"]
    assert schema["required"] == ["enabled", "label"]
    assert "default" not in schema["properties"]["enabled"]
    _validate_wire(Defaults, domain.model_dump(mode="json"))
    for value in ({}, {"enabled": None, "label": None}, {"enabled": False}):
        with pytest.raises(jsonschema.ValidationError):
            _validate_wire(Defaults, value)
    assert Defaults.model_validate(domain.model_dump(mode="json")) == domain












@pytest.mark.parametrize("branches", [
    [_object({"kind": {"const": "same", "type": "string"}})] * 2,
    [_object({"kind": {"enum": ["a", "b"], "type": "string"}}), _object({"kind": {"const": "b", "type": "string"}})],
    [{"type": "string"}, {"type": "integer"}],
    [_object({"left": {"const": "a", "type": "string"}}), _object({"right": {"const": "b", "type": "string"}})],
])
def test_unproved_oneof_fails_with_its_schema_location(branches):
    with pytest.raises(SchemaCompilationError) as caught:
        compile_output_schema(_schema_model(_object({"item": {"oneOf": branches}})))
    assert caught.value.path == "#/properties/item/oneOf"
    assert caught.value.reason
    assert caught.value.path in str(caught.value)








































