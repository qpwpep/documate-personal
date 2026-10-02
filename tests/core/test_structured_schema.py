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


@pytest.mark.parametrize("recipient", [
    {"state": "omitted"},
    {"state": "explicit", "selector": {"kind": "channel", "value": "C123"}, "evidence_ids": ["e1"]},
    {"state": "unresolved", "raw_input": "the team", "reason": "ambiguous", "evidence_ids": ["e1"]},
])
def test_planner_recipient_variants_survive_the_wire_and_domain_boundaries(recipient):
    contract = WireRequestContract.model_validate({
        "slack_recipient": recipient,
        "evidence": [{"id": "e1", "turn_id": "u1", "quote": "the team", "scope": "slack_recipient", "interpretation": "instruction"}],
    })
    plan = PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract)
    value = plan.model_dump(mode="json")
    _validate_wire(PlannerOutput, value)
    assert PlannerOutput.model_validate(value) == plan
    malformed = deepcopy(value)
    malformed["request_contract"]["slack_recipient"]["state"] = "unknown"
    with pytest.raises(jsonschema.ValidationError):
        _validate_wire(PlannerOutput, malformed)
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(malformed)


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


@pytest.mark.parametrize("block", [
    {"type": "paragraph", "content": [{"text": "Paragraph"}]},
    {"type": "list", "items": [{"text": "Item"}]},
    {"type": "code", "content": {"text": "print(1)"}},
    {"type": "table", "columns": [{"text": "Name"}], "rows": [[{"text": "Value"}]]},
    {"type": "heading", "content": {"text": "Heading"}},
])
def test_every_answer_block_retains_its_domain_type(block):
    document = AnswerDocument.model_validate({"blocks": [block]})
    value = document.model_dump(mode="json")
    _validate_wire(AnswerDocument, value)
    assert AnswerDocument.model_validate(value) == document
    value["blocks"][0]["unexpected"] = "not allowed"
    with pytest.raises(jsonschema.ValidationError):
        _validate_wire(AnswerDocument, value)


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


def test_schema_keywords_used_as_property_names_are_not_rewritten_or_removed():
    model = create_model("KeywordFields", __config__=ConfigDict(extra="forbid"), **{
        name: (str, Field(default="value")) for name in ("default", "oneOf", "const", "discriminator")
    })
    value = model().model_dump(mode="json")
    schema = compile_output_schema(model)["schema"]
    assert set(schema["properties"]) == set(value)
    _validate_wire(model, value)


def test_scalar_and_array_constraints_are_preserved():
    class Constrained(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int = Field(ge=1, le=3)
        name: str = Field(min_length=1, max_length=3, pattern="^[a-z]+$")
        values: list[int] = Field(min_length=1, max_length=2)

    good = {"count": 2, "name": "abc", "values": [1, 2]}
    _validate_wire(Constrained, good)
    for key, wrong in (("count", 0), ("count", 4), ("name", ""), ("name", "abcd"),
                       ("name", "ABC"), ("values", []), ("values", [1, 2, 3])):
        with pytest.raises(jsonschema.ValidationError):
            _validate_wire(Constrained, good | {key: wrong})
        with pytest.raises(ValidationError):
            Constrained.model_validate(good | {key: wrong})


def test_referenced_oneof_is_lowered_only_when_required_tags_are_disjoint():
    source = _object({"item": {"oneOf": [{"$ref": "#/$defs/A"}, {"$ref": "#/$defs/B"}]}})
    source["$defs"] = {
        "A": _object({"kind": {"const": "a", "type": "string"}, "value": {"type": "string"}}),
        "B": _object({"kind": {"enum": ["b", "c"], "type": "string"}, "value": {"type": "integer"}}),
    }
    model = _schema_model(source)
    schema = compile_output_schema(model)["schema"]
    assert "anyOf" in schema["properties"]["item"]
    assert "oneOf" not in schema["properties"]["item"]
    for value in ({"kind": "a", "value": "text"}, {"kind": "b", "value": 1}, {"kind": "c", "value": 2}):
        _validate_wire(model, {"item": value})
    for value in ({"kind": "a", "value": 1}, {"kind": "b", "value": "text"}, {"value": "text"}):
        with pytest.raises(jsonschema.ValidationError):
            _validate_wire(model, {"item": value})
    assert source == model.model_json_schema(), "Compiling must not mutate a model-owned schema"


def test_tagged_union_acceptance_matches_oneof_for_canonical_wire_candidates():
    branches = [
        _object({"kind": {"type": "string", "const": "a"}, "value": {"type": "string"}}),
        _object({"kind": {"type": "string", "enum": ["b", "c"]}, "value": {"type": "integer"}}),
    ]
    for branch in branches:
        branch["required"] = ["kind", "value"]
    original = _object({"item": {"oneOf": branches}})
    original["required"] = ["item"]
    lowered = compile_output_schema(_schema_model(original))["schema"]
    candidates = [{"item": {"kind": tag, "value": value}}
                  for tag, value in product(("a", "b", "c", "unknown", None), ("text", 1, None, True))]
    candidates.extend(({"item": {"value": 1}}, {"item": {"kind": "a", "value": "text", "extra": 1}}, {}))
    for candidate in candidates:
        assert jsonschema.Draft202012Validator(original).is_valid(candidate) == jsonschema.Draft202012Validator(lowered).is_valid(candidate)


@pytest.mark.parametrize("constant,enumeration", [(1, [True]), (True, [1]), ("a", ["b"])])
def test_conflicting_const_and_enum_are_rejected_without_python_boolean_numeric_coercion(constant, enumeration):
    model = _schema_model(_object({"value": {"type": "integer", "const": constant, "enum": enumeration}}))
    with pytest.raises(SchemaCompilationError) as caught:
        compile_output_schema(model)
    assert caught.value.path == "#/properties/value/const"


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


@pytest.mark.parametrize("schema,path", [
    (_object({"value": {"$ref": "#/$defs/Missing"}}), "#/properties/value/$ref"),
    (_object({"value": {"$ref": "#/properties"}}), "#/properties/value/$ref"),
    (_object({"type": {"type": "string"}, "value": {"$ref": "#/properties"}}), "#/properties/value/$ref"),
    (_object({"value": {"type": "string", "not": {"const": "bad"}}}), "#/properties/value/not"),
    (_object({"value": {"type": "object", "additionalProperties": {"type": "string"}}}), "#/properties/value/additionalProperties"),
    (_object({"value": {"type": "array", "prefixItems": [{"type": "integer"}]}}), "#/properties/value/prefixItems"),
    (_object({"value": {}}), "#/properties/value"),
    ({"type": "object", "properties": {}, "additionalProperties": True}, "#/additionalProperties"),
    ({"type": "string"}, "#"),
    ({"anyOf": [_object({}), _object({})]}, "#"),
])
def test_unsupported_schema_shapes_fail_before_provider_requests(schema, path):
    with pytest.raises(SchemaCompilationError) as caught:
        compile_output_schema(_schema_model(schema))
    assert caught.value.path == path


def test_pydantic_schema_generation_failure_has_the_same_compilation_error_contract():
    class Unrepresentable(BaseModel):
        callback: Callable[[], str]

    with pytest.raises(SchemaCompilationError) as caught:
        compile_output_schema(Unrepresentable)
    assert caught.value.path == "#"
    assert "Pydantic" in caught.value.reason


def test_recursive_model_uses_local_definitions_without_losing_strict_fields():
    class Node(BaseModel):
        model_config = ConfigDict(extra="forbid")
        label: str
        children: list[Node] = Field(default_factory=list)

    document = Node(label="root", children=[Node(label="child")])
    schema = compile_output_schema(Node)["schema"]
    assert schema["type"] == "object"
    _validate_wire(Node, document.model_dump(mode="json"))
    with pytest.raises(jsonschema.ValidationError):
        _validate_wire(Node, {"label": "root", "children": [{"label": "child"}]})


@pytest.mark.parametrize("document", [
    {"blocks": [{"type": "paragraph", "content": [{"text": "   ", "basis": "interaction", "refs": []}]}]},
    {"blocks": [{"type": "code", "language": "py\n", "content": {"text": "print(1)", "basis": "interaction", "refs": []}}]},
    {"blocks": [{"type": "table", "columns": [{"text": "Column", "basis": "interaction", "refs": []}], "rows": [[]]}]},
])
def test_provider_valid_json_still_requires_the_existing_domain_validators(document):
    # Schema compilation does not claim to encode custom Python validators.
    _validate_wire(AnswerDocument, document)
    with pytest.raises(ValidationError):
        AnswerDocument.model_validate(document)


def test_all_service_schema_nodes_are_strict_without_changing_domain_schemas():
    for model in (PlannerOutput, AnswerDocument):
        before = model.model_json_schema()
        compiled = compile_output_schema(model)
        assert compiled["name"] == model.__name__
        assert compiled["strict"] is True

        def check(node):
            assert not {"oneOf", "discriminator", "default", "const"} & node.keys()
            if node.get("type") == "object":
                assert node["additionalProperties"] is False
                assert node["required"] == list(node["properties"])
            for key in ("properties", "$defs"):
                for child in node.get(key, {}).values():
                    check(child)
            for child in node.get("anyOf", []):
                check(child)
            if "items" in node:
                check(node["items"])

        check(compiled["schema"])
        assert model.model_json_schema() == before


def test_fingerprint_is_stable_across_dictionary_order_and_changes_with_contract():
    compiled = compile_output_schema(AnswerDocument, name="SynthesisDocument")
    assert compiled["name"] == "SynthesisDocument"
    fingerprint = schema_fingerprint(compiled)
    assert len(fingerprint) == 64
    assert fingerprint == schema_fingerprint(dict(reversed(list(compiled.items()))))
    assert fingerprint != schema_fingerprint(compiled | {"name": "OtherDocument"})


def test_provider_property_count_limit_is_checked_before_requests():
    allowed = _object({f"f{i}": {"type": "string"} for i in range(5000)})
    compile_output_schema(_schema_model(allowed))
    allowed["properties"]["overflow"] = {"type": "string"}
    with pytest.raises(SchemaCompilationError, match="5000"):
        compile_output_schema(_schema_model(allowed))


def test_provider_string_budget_counts_names_definitions_and_enum_literals():
    source = _object({"x" * 119990: {"$ref": "#/$defs/abc"}})
    source["$defs"] = {"abc": {"type": "string", "enum": ["1234567"]}}
    compile_output_schema(_schema_model(source))
    source["$defs"]["abc"]["enum"] = ["12345678"]
    with pytest.raises(SchemaCompilationError, match="120000"):
        compile_output_schema(_schema_model(source))


def test_provider_total_enum_limit_counts_all_schema_nodes():
    source = _object({f"v{i}": {"type": "integer", "enum": list(range(250))} for i in range(4)})
    compile_output_schema(_schema_model(source))
    source["properties"]["overflow"] = {"type": "integer", "const": 250}
    with pytest.raises(SchemaCompilationError, match="1000"):
        compile_output_schema(_schema_model(source))


def test_large_single_enum_has_a_separate_string_budget():
    values = [f"{i:03d}" + "x" * 56 for i in range(251)]
    values[0] += "x" * 191
    assert sum(map(len, values)) == 15000
    source = _object({"value": {"type": "string", "enum": values}})
    compile_output_schema(_schema_model(source))
    values[0] += "x"
    with pytest.raises(SchemaCompilationError, match="15000"):
        compile_output_schema(_schema_model(source))
    source["properties"]["value"]["enum"] = values[:250]
    compile_output_schema(_schema_model(source))


def test_nesting_limit_follows_references_and_allows_recursive_contracts():
    source = _object({"child": {"$ref": "#/$defs/Level1"}})
    source["$defs"] = {
        f"Level{i}": _object({"child": {"$ref": f"#/$defs/Level{i + 1}"}}) for i in range(1, 9)
    }
    source["$defs"]["Level9"] = _object({"value": {"type": "string"}})
    compile_output_schema(_schema_model(source))
    source["$defs"]["Level9"] = _object({"child": {"$ref": "#/$defs/Level10"}})
    source["$defs"]["Level10"] = _object({"value": {"type": "string"}})
    with pytest.raises(SchemaCompilationError, match="10"):
        compile_output_schema(_schema_model(source))
    recursive = _object({"child": {"anyOf": [{"$ref": "#"}, {"type": "null"}]}})
    compile_output_schema(_schema_model(recursive))


def test_supported_duration_format_is_not_discarded():
    schema = compile_output_schema(_schema_model(_object({"value": {"type": "string", "format": "duration"}})))
    assert schema["schema"]["properties"]["value"]["format"] == "duration"


def test_reference_annotations_are_removed_without_changing_the_referenced_contract():
    class Child(BaseModel):
        model_config = ConfigDict(extra="forbid")
        value: str

    class Parent(BaseModel):
        model_config = ConfigDict(extra="forbid")
        child: Child = Field(description="A provider cannot accept this beside a ref", title="Child field")

    schema = compile_output_schema(Parent)["schema"]
    assert set(schema["properties"]["child"]) == {"$ref"}
    _validate_wire(Parent, {"child": {"value": "present"}})
    with pytest.raises(jsonschema.ValidationError):
        _validate_wire(Parent, {"child": {}})


def test_reference_assertion_siblings_are_not_silently_removed():
    source = _object({"value": {"$ref": "#/$defs/Value", "enum": ["restricted"]}})
    source["$defs"] = {"Value": {"type": "string"}}
    with pytest.raises(SchemaCompilationError) as caught:
        compile_output_schema(_schema_model(source))
    assert caught.value.path == "#/properties/value/enum"
