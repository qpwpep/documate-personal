"""The HTTP boundary validates canonical usage without reading provider metadata."""

from types import SimpleNamespace

import pytest

from src.app.web.agent_request_support import normalize_debug_info
from src.app.web.schemas import AgentDebugInfo
from src.core.contracts.boundary.debug import normalize_llm_call_observation
from src.core.contracts.debug import DebugPayload, build_llm_call_metadata


@pytest.mark.parametrize(("inputs", "outputs"), [(0, 0), (10, None), (None, 20), (None, None)])
def test_http_serialization_preserves_each_observed_dimension(inputs, outputs):
    call = build_llm_call_metadata(
        stage="synthesis", attempt=1, path="structured",
        message=SimpleNamespace(usage_metadata={"input_tokens": inputs, "output_tokens": outputs}, response_metadata={}),
    )
    payload = DebugPayload(llm_calls=[call]).model_dump(mode="json")
    normalized = normalize_debug_info(payload, latency_ms_server=1)
    received = AgentDebugInfo.model_validate_json(normalized.model_dump_json())
    assert received.llm_calls[0].usage.input_tokens == inputs
    assert received.llm_calls[0].usage.output_tokens == outputs
    assert "token_usage" not in received.model_dump()


def test_wire_usage_is_authoritative_over_retained_provider_diagnostics():
    call = build_llm_call_metadata(
        stage="synthesis", attempt=1, path="structured",
        message=SimpleNamespace(usage_metadata={"input_tokens": 0, "output_tokens": 0}, response_metadata={
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 20},
        }),
    )
    normalized = normalize_debug_info(DebugPayload(llm_calls=[call]).model_dump(mode="json"), 1)
    assert normalized.llm_calls[0].usage.total_tokens == 0
    assert len(normalized.llm_calls[0].usage.issues) == 2
    assert "llm_calls" not in normalized.missing_required_debug_fields


def test_invalid_wire_entry_cannot_disappear_and_make_a_scope_look_complete():
    valid = build_llm_call_metadata(stage="planner", attempt=1, path="structured").model_dump(mode="json")
    malformed = {**valid, "usage": {"input_tokens": True, "output_tokens": 3}}
    raw = DebugPayload(llm_calls=[]).model_dump(mode="json")
    raw["llm_calls"] = [valid, malformed]
    normalized = normalize_debug_info(raw, 1)
    assert normalized.llm_calls is None
    assert "llm_calls" in normalized.missing_required_debug_fields
    assert any("llm_calls[1]" in error for error in normalized.errors)


def test_unknown_call_scope_and_confirmed_no_call_roundtrip_distinctly():
    for calls in (None, []):
        normalized = normalize_debug_info(DebugPayload(llm_calls=calls).model_dump(mode="json"), 1)
        assert AgentDebugInfo.model_validate_json(normalized.model_dump_json()).llm_calls == calls


def test_live_raw_only_calls_are_not_treated_as_a_legacy_adapter():
    calls, errors = normalize_llm_call_observation([{
        "stage": "synthesis", "path": "structured",
        "usage_metadata": {"input_tokens": 100, "output_tokens": 20},
    }])
    assert calls is None
    assert errors and "usage" in errors[0]
