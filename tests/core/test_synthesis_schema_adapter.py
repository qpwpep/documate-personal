import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from src.core.answer_schema import AnswerDocument, text_document
from src.runtime.nodes.synthesis.schema_adapter import (
    build_structured_synthesizer, coerce_answer_document, coerce_structured_synthesis_result,
)


class StructuredBoundary:
    def with_structured_output(self, schema, **options):
        self.schema = schema
        self.options = options
        return self


def test_model_schema_requests_only_the_visible_document():
    """The structured boundary requests one canonical body instead of duplicate answer fields."""
    boundary = StructuredBoundary()
    build_structured_synthesizer(boundary)
    assert boundary.schema["name"] == "AnswerDocument"
    assert set(boundary.schema["schema"]["properties"]) == {"blocks"}
    assert boundary.options == {"method": "json_schema", "include_raw": True, "strict": True}


@pytest.mark.parametrize("raw", [{"answer": "unvalidated prose"}, "unvalidated prose", "", "{broken"])
def test_malformed_output_is_not_promoted_to_visible_text(raw):
    """A malformed generation cannot bypass the document contract as a plain answer."""
    with pytest.raises(ValidationError):
        coerce_answer_document(raw)


def test_valid_json_document_preserves_displayed_text_and_references():
    """JSON parsing retains the exact content units that validation and UI consume."""
    expected = text_document("Default mode is safe.", basis="source", refs=["E1"])
    assert coerce_answer_document(AIMessage(content=expected.model_dump_json())) == expected


def test_structured_boundary_preserves_parse_failure_and_usage_message():
    """Provider parsing failures remain failures while raw usage metadata remains observable."""
    raw = AIMessage(content="", response_metadata={"model_name": "test-model"})
    failure = ValueError("invalid JSON")
    assert coerce_structured_synthesis_result({"raw": raw, "parsed": None, "parsing_error": failure}) == (
        None, raw, failure,
    )


def test_wire_schema_uses_supported_nested_union_without_changing_server_types():
    """The provider receives a supported anyOf union while server parsing retains tagged blocks."""
    boundary = StructuredBoundary()
    build_structured_synthesizer(boundary)
    schema = boundary.schema["schema"]
    union = schema["properties"]["blocks"]["items"]
    assert "anyOf" in union
    assert "oneOf" not in union
    assert "discriminator" not in union
    assert schema["$defs"]["ListBlock"]["properties"]["type"]["enum"] == ["list"]
    assert "default" not in schema["$defs"]["ListBlock"]["properties"]["ordered"]
    assert AnswerDocument.model_json_schema()["properties"]["blocks"]["items"]["discriminator"]["propertyName"] == "type"
