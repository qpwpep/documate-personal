from __future__ import annotations

from copy import deepcopy
from typing import Any

import jsonschema
import pytest
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.structured_schema import SchemaCompilationError, compile_output_schema


def _schema_model(schema: dict[str, Any]) -> type[BaseModel]:
    class SchemaModel(BaseModel):
        @classmethod
        def model_json_schema(cls, **_kwargs: Any) -> dict[str, Any]:
            return deepcopy(schema)

    return SchemaModel


def _object(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object", "properties": properties,
        "required": list(properties), "additionalProperties": False,
    }


def _wire_validator(model: type[BaseModel]) -> jsonschema.Draft202012Validator:
    schema = compile_output_schema(model)["schema"]
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@pytest.mark.parametrize("body", [
    {"kind": "copy_input", "source": {"turn_id": "u1", "quote": "hello", "occurrence": None}},
    {"kind": "transform_answer", "source": {"ref": "pending"}, "instruction": "Shorten"},
    {"kind": "extract", "instruction": "Show the source", "requirement_ids": ["req1"]},
])
def test_planner_body_variants_keep_their_canonical_meaning(body):
    plan = PlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract(
        slack_recipient={"state": "omitted"}, body=body,
    ))
    value = plan.model_dump(mode="json")
    validator = _wire_validator(PlannerOutput)
    validator.validate(value)
    assert PlannerOutput.model_validate(value) == plan

    malformed = deepcopy(value)
    malformed["request_contract"]["body"]["kind"] = "unknown"
    with pytest.raises(jsonschema.ValidationError):
        validator.validate(malformed)
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(malformed)


def test_referenced_tagged_union_preserves_oneof_acceptance_when_lowered():
    source = _object({"item": {"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}})
    source["$defs"] = {
        "A": _object({"kind": {"const": "a", "type": "string"}, "value": {"type": "string"}}),
        "B": _object({"kind": {"enum": ["b", "c"], "type": "string"}, "value": {"type": "integer"}}),
    }
    original = jsonschema.Draft202012Validator(source)
    lowered = _wire_validator(_schema_model(source))
    assert "anyOf" in lowered.schema["properties"]["item"]

    for item in ({"kind": "a", "value": "text"}, {"kind": "b", "value": 1}, {"kind": "c", "value": 2}):
        candidate = {"item": item}
        assert original.is_valid(candidate)
        assert lowered.is_valid(candidate)
    for item in (
        {"kind": "a", "value": 1}, {"kind": "b", "value": "text"},
        {"kind": "b", "value": True}, {"kind": "unknown", "value": 1},
        {"value": "text"}, {"kind": "a", "value": "text", "extra": 1},
    ):
        candidate = {"item": item}
        assert not original.is_valid(candidate)
        assert not lowered.is_valid(candidate)


@pytest.mark.parametrize("branches", [
    [_object({"kind": {"enum": ["a", "b"], "type": "string"}}),
     _object({"kind": {"const": "b", "type": "string"}})],
    [_object({"left": {"const": "a", "type": "string"}}),
     _object({"right": {"const": "b", "type": "string"}})],
    [{"type": "string"}, {"type": "integer"}],
], ids=["overlapping-tags", "no-common-tag", "untagged-primitives"])
def test_unproved_oneof_fails_with_its_schema_location(branches):
    with pytest.raises(SchemaCompilationError) as caught:
        compile_output_schema(_schema_model(_object({"item": {"oneOf": branches}})))
    assert caught.value.path == "#/properties/item/oneOf"


def test_wire_requires_canonical_fields_without_making_nonnullable_defaults_nullable():
    class Defaults(BaseModel):
        model_config = ConfigDict(extra="forbid")
        enabled: bool = False
        label: str | None = None

    # Domain shorthand may omit defaults; canonical wire input must spell them out.
    domain = Defaults.model_validate({})
    validator = _wire_validator(Defaults)
    canonical = domain.model_dump(mode="json")
    validator.validate(canonical)
    assert Defaults.model_validate(canonical) == domain
    for value in ({}, {"enabled": None, "label": None}, {"enabled": False}):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(value)


def test_scalar_and_array_constraints_are_not_silently_relaxed():
    class Constrained(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int = Field(ge=1, le=3)
        name: str = Field(min_length=1, max_length=3, pattern="^[a-z]+$")
        values: list[int] = Field(min_length=1, max_length=2)

    validator = _wire_validator(Constrained)
    good = {"count": 2, "name": "abc", "values": [1, 2]}
    validator.validate(good)
    assert Constrained.model_validate(good).model_dump(mode="json") == good
    for key, wrong in (
        ("count", 0), ("count", 4), ("name", ""), ("name", "abcd"),
        ("name", "ABC"), ("values", []), ("values", [1, 2, 3]),
    ):
        with pytest.raises(jsonschema.ValidationError):
            validator.validate(good | {key: wrong})
        with pytest.raises(ValidationError):
            Constrained.model_validate(good | {key: wrong})
