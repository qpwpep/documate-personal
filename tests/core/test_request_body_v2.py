from __future__ import annotations

import jsonschema
import pytest
from pydantic import ValidationError

from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    AnswerReference, BoundInputText, CopyInputBody, InputTextReference, RequestContract,
    TransformInputBody, WireRequestContract,
)
from src.infra.llm import _build_planner_response_schema


@pytest.mark.parametrize("body", [
    {"kind": "compose", "instruction": "Explain lists", "evidence_ids": []},
    {"kind": "acknowledge", "evidence_ids": []},
    {"kind": "copy_input", "source": {"turn_id": "u1", "quote": "hello", "occurrence": None}},
    {"kind": "transform_input", "source": {"turn_id": "u1", "quote": "hello", "occurrence": None}, "instruction": "Translate into Korean", "evidence_ids": []},
    {"kind": "copy_answer", "source": {"ref": "previous"}},
    {"kind": "transform_answer", "source": {"ref": "pending"}, "instruction": "Shorten to three lines", "evidence_ids": []},
    {"kind": "extract", "instruction": "Show the source", "requirement_ids": ["req1"], "evidence_ids": []},
    {"kind": "unresolved", "question": "Which function?", "instruction": "Explain the function"},
])
def test_each_body_kind_has_the_same_static_contract_in_provider_and_server(body):
    """Every supported body operation has one valid source shape in both boundaries."""
    contract = WireRequestContract(body=body)
    value = PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract).model_dump(mode="json")
    jsonschema.validate(value, _build_planner_response_schema()["schema"])
    assert PlannerOutput.model_validate(value).request_contract.body == contract.body


@pytest.mark.parametrize("body", [
    {"operation": "reuse", "source": "current"},
    {"kind": "copy_answer", "source": {"turn_id": "u1", "quote": "hello"}},
    {"kind": "transform_input", "source": {"ref": "previous"}, "instruction": "Translate"},
    {"kind": "transform_answer", "source": {"ref": "previous"}, "instruction": ""},
])
def test_invalid_operation_source_combinations_are_rejected_by_both_boundaries(body):
    """Malformed or legacy bodies are rejected rather than silently normalized."""
    value = PlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract()).model_dump(mode="json")
    value["request_contract"]["body"] = body
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(value, _build_planner_response_schema()["schema"])
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(value)


def test_wire_schema_does_not_ask_the_model_for_server_identity_or_bound_text_fields():
    """Request identity, revisions, offsets and hashes are allocated after interpretation."""
    schema = _build_planner_response_schema()["schema"]
    properties = schema["$defs"]["WireRequestContract"]["properties"]
    assert not {"request_id", "revision", "status", "failure", "ready", "body_request"} & properties.keys()
    assert set(schema["$defs"]["InputTextReference"]["properties"]) == {"turn_id", "quote", "occurrence"}
    assert set(schema["$defs"]["AnswerReference"]["properties"]) == {"ref"}


@pytest.mark.parametrize("kind,value", [("line_count", None), ("line_count", 0), ("table", 3),
                                       ("line_count", True), ("line_count", "3"), ("line_count", 1.5)])
def test_format_values_rejected_by_server_are_also_rejected_by_generation_schema(kind, value):
    """A line count requires an integer, while other layouts cannot acquire a count."""
    candidate = PlannerOutput(use_retrieval=False, tasks=[], request_contract=WireRequestContract()).model_dump(mode="json")
    contract = candidate["request_contract"]
    contract["evidence"] = [{"id": "e1", "turn_id": "u1", "quote": "requested format", "scope": "answer.format", "interpretation": "instruction"}]
    contract["answer"]["format"] = [{"kind": kind, "mode": "forbidden", "value": value, "evidence_ids": ["e1"]}]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(candidate, _build_planner_response_schema()["schema"])
    with pytest.raises(ValidationError):
        PlannerOutput.model_validate(candidate)


@pytest.mark.parametrize("kind,value", [("line_count", 3), ("line_count", 1.0), ("table", None), ("ordered_list", None)])
def test_valid_format_variants_roundtrip_through_generation_schema_and_server(kind, value):
    contract = WireRequestContract.model_validate({
        "evidence": [{"id": "e1", "turn_id": "u1", "quote": "requested format", "scope": "answer.format", "interpretation": "instruction"}],
        "answer": {"format": [{"kind": kind, "mode": "required", "value": value, "evidence_ids": ["e1"]}]},
    })
    candidate = PlannerOutput(use_retrieval=False, tasks=[], request_contract=contract).model_dump(mode="json")
    jsonschema.validate(candidate, _build_planner_response_schema()["schema"])
    assert PlannerOutput.model_validate(candidate).request_contract.answer == contract.answer
