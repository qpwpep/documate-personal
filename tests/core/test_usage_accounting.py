"""Provider counters and failed attempts remain useful without invented usage."""

import pytest
from langchain_core.messages import AIMessage

from src.core.contracts.usage import normalize_token_usage
from src.runtime.agent_runtime.llm_usage import capture_llm_usage, record_llm_call


@pytest.mark.parametrize("metadata", [
    {
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_token_details": {"cache_read": 30, "cache_creation": 40},
        "output_token_details": {"reasoning": 7},
    },
    {
        "input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
        "input_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 40},
        "output_tokens_details": {"reasoning_tokens": 7},
    },
    {
        "prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120,
        "prompt_tokens_details": {"cached_tokens": 30, "cache_write_tokens": 40},
        "completion_tokens_details": {"reasoning_tokens": 7},
    },
])
def test_cache_and_reasoning_are_subsets_of_observed_totals(metadata):
    usage = normalize_token_usage(metadata, None)

    assert usage.input_tokens == 100
    assert usage.output_tokens == 20
    assert usage.cached_input_tokens == 30
    assert usage.cache_write_input_tokens == 40
    assert usage.uncached_input_tokens == 30
    assert usage.reasoning_output_tokens == 7
    assert usage.total_tokens == 120
    assert usage.issues == []


def test_missing_cache_details_do_not_mean_zero_cache_usage():
    usage = normalize_token_usage({
        "input_tokens": 100, "output_tokens": 20,
        "input_token_details": {"cache_read": 0},
    }, None)

    assert usage.cached_input_tokens == 0
    assert usage.cache_write_input_tokens is None
    assert usage.uncached_input_tokens is None
    assert usage.reasoning_output_tokens is None
    assert usage.total_tokens == 120


def test_sdk_zero_cache_observation_wins_over_conflicting_response_metadata():
    usage = normalize_token_usage(
        {"input_tokens": 100, "output_tokens": 20, "input_token_details": {"cache_read": 0}},
        {"token_usage": {"prompt_tokens": 100, "completion_tokens": 20,
                         "prompt_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 10}}},
    )

    assert usage.cached_input_tokens == 0
    assert usage.cache_write_input_tokens == 10
    assert usage.uncached_input_tokens == 90
    assert any(issue.code == "conflict" and issue.field == "cached_input_tokens" for issue in usage.issues)


@pytest.mark.parametrize("invalid", [True, -1, "12"])
def test_invalid_nested_counters_are_unknown_and_reported(invalid):
    usage = normalize_token_usage({
        "input_tokens": 100, "output_tokens": 20,
        "input_token_details": {"cache_read": invalid, "cache_creation": 0},
        "output_token_details": {"reasoning": invalid},
    }, None)

    assert usage.cached_input_tokens is None
    assert usage.cache_write_input_tokens == 0
    assert usage.reasoning_output_tokens is None
    assert usage.uncached_input_tokens is None
    assert {(issue.code, issue.field) for issue in usage.issues} >= {
        ("invalid_value", "cached_input_tokens"), ("invalid_value", "reasoning_output_tokens"),
    }


def test_invalid_detail_container_does_not_discard_valid_totals():
    usage = normalize_token_usage({
        "input_tokens": 100, "output_tokens": 20, "input_token_details": "unavailable",
    }, None)

    assert usage.total_tokens == 120
    assert usage.cached_input_tokens is None
    assert usage.cache_write_input_tokens is None
    assert any(issue.code == "invalid_value" for issue in usage.issues)


def test_cache_subsets_larger_than_input_do_not_produce_negative_uncached_usage():
    usage = normalize_token_usage({
        "input_tokens": 100, "output_tokens": 20,
        "input_token_details": {"cache_read": 80, "cache_creation": 30},
    }, None)

    assert usage.uncached_input_tokens is None
    assert usage.issues


def test_failed_attempt_retains_identity_and_elapsed_time_without_inventing_usage():
    with capture_llm_usage() as recorder:
        with pytest.raises(TimeoutError, match="provider did not respond"):
            with record_llm_call(stage="planner", attempt=1, path="structured",
                                 model_name="test-model", endpoint="chat/completions"):
                started = recorder.snapshot()[0]
                assert started.status == "started"
                assert started.model_name == "test-model"
                assert started.endpoint == "chat/completions"
                raise TimeoutError("provider did not respond")

        failed = recorder.snapshot()[0]

    assert failed.status == "failed"
    assert failed.model_name == "test-model"
    assert failed.endpoint == "chat/completions"
    assert failed.latency_ms is not None and failed.latency_ms >= 0
    assert failed.failure_type == "TimeoutError"
    assert failed.usage.observation == "none"


def test_completed_attempt_retains_initial_identity_when_response_omits_model():
    with capture_llm_usage() as recorder:
        with record_llm_call(stage="summarize", attempt=1, path="direct",
                             model_name="test-model", endpoint="chat/completions") as call:
            call.complete(AIMessage(content="summary", usage_metadata={
                "input_tokens": 10, "output_tokens": 4, "total_tokens": 14,
            }))
        completed = recorder.snapshot()[0]

    assert completed.status == "completed"
    assert completed.model_name == "test-model"
    assert completed.endpoint == "chat/completions"
    assert completed.latency_ms is not None and completed.latency_ms >= 0
    assert completed.failure_type is None
    assert completed.usage.total_tokens == 14
