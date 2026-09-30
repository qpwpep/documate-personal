"""Provider observations preserve zero, absence and invalid data across one contract."""

import pytest
from pydantic import ValidationError

from src.core.contracts.usage import LLMCallRecord, TokenUsage, normalize_token_usage


def test_empty_primary_metadata_uses_the_secondary_observation():
    usage = normalize_token_usage({}, {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}})
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 20, 120)
    assert usage.observation == "complete"
    assert usage.issues == []


@pytest.mark.parametrize(("raw", "expected", "observation"), [
    ({}, (None, None), "none"),
    ({"input_tokens": None, "output_tokens": None}, (None, None), "none"),
    ({"input_tokens": 0, "output_tokens": 0}, (0, 0), "complete"),
    ({"input_tokens": 0}, (0, None), "partial"),
    ({"output_tokens": 0}, (None, 0), "partial"),
    ({"total_tokens": 120}, (None, None), "none"),
])
def test_observation_is_derived_from_selected_dimensions(raw, expected, observation):
    usage = normalize_token_usage(raw, None)
    assert (usage.input_tokens, usage.output_tokens) == expected
    assert usage.observation == observation
    assert usage.total_tokens == (sum(expected) if observation == "complete" else None)


@pytest.mark.parametrize("invalid", [-1, True, False, 3.0, 3.9, "3", [], {}, float("nan"), float("inf"), float("-inf")])
def test_invalid_counts_are_not_coerced_and_can_fall_back(invalid):
    missing = normalize_token_usage({"input_tokens": invalid}, {})
    assert missing.input_tokens is None
    assert missing.observation == "none"
    assert [(issue.code, issue.field, issue.source) for issue in missing.issues] == [
        ("invalid_value", "input_tokens", "usage_metadata.input_tokens"),
    ]
    recovered = normalize_token_usage(
        {"input_tokens": invalid, "output_tokens": 0},
        {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20}},
    )
    assert (recovered.input_tokens, recovered.output_tokens) == (100, 0)
    assert recovered.observation == "complete"
    assert {issue.code for issue in recovered.issues} == {"invalid_value", "conflict"}
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=invalid)


def test_alias_and_candidate_precedence_never_replaces_an_observed_zero():
    usage = normalize_token_usage(
        {"input_tokens": 0, "prompt_tokens": 10, "output_tokens": 0, "completion_tokens": 20},
        {"token_usage": {"input_tokens": 30, "prompt_tokens": 40, "output_tokens": 50, "completion_tokens": 60}},
    )
    assert (usage.input_tokens, usage.output_tokens) == (0, 0)
    assert len(usage.issues) == 6
    assert all(issue.code == "conflict" for issue in usage.issues)


def test_partial_candidates_fill_only_missing_dimensions_of_the_same_call():
    usage = normalize_token_usage(
        {"input_tokens": 100},
        {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 999}},
    )
    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.total_tokens == 120
    assert [issue.code for issue in usage.issues] == ["total_mismatch"]


@pytest.mark.parametrize("invalid", [[], 1, True, "metadata"])
def test_invalid_container_does_not_hide_a_valid_candidate(invalid):
    usage = normalize_token_usage(invalid, {"token_usage": {"prompt_tokens": 2, "completion_tokens": 3}})
    assert usage.total_tokens == 5
    assert usage.issues[0].source == "usage_metadata"


def test_call_record_requires_canonical_usage_and_rejects_old_raw_only_payload():
    with pytest.raises(ValidationError):
        LLMCallRecord.model_validate({"stage": "synthesis", "path": "structured", "usage_metadata": {"output_tokens": 20}})
