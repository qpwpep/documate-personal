"""Terminal outcomes retain their meaning through offline evaluation parsing."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from src.core.contracts.debug import DebugPayload
from src.core.llm_errors import make_problem
from src.eval.online_runner.response_parser import parse_agent_response
from tests.eval.response_fixtures import answer_provenance, plain_response, sse_http_response
from tests.eval.test_runner_sse import run_case


def terminal_payload(status, *, include_debug=True):
    response = plain_response("검증된 일부 답변") if status == "partial" else None
    code = "model_refusal" if status == "refused" else "provider_schema_invalid"
    problem = None if status == "needs_input" else make_problem(code, "planner").model_dump(mode="json")
    debug = DebugPayload().model_dump(mode="json") if include_debug else None
    if debug is not None:
        debug["answer_provenance"] = answer_provenance(response) if response is not None else None
        debug["error_codes"] = [code] if problem is not None else []
    return {
        "status": status, "response": response, "problem": problem,
        "message": "어떤 문서를 사용할까요?" if status == "needs_input" else "",
        "missing_slots": ["document"] if status == "needs_input" else [],
        "request_id": "terminal-request", "debug": debug,
        "upload_manifest": {"epoch": "fixture-epoch", "revision": 0, "files": []},
    }


@pytest.mark.parametrize("status", ["failed", "refused", "needs_input", "partial"])
def test_parser_preserves_the_terminal_contract_without_answer_schema_errors(status):
    payload = terminal_payload(status)
    parsed = parse_agent_response(payload)

    assert parsed.response_errors == []
    assert parsed.turn_result.status == status
    assert parsed.turn_result.message == payload["message"]
    assert parsed.turn_result.missing_slots == payload["missing_slots"]
    assert parsed.request_id == "terminal-request"
    assert parsed.debug_observability_status == "ok"
    if status == "needs_input":
        assert parsed.runtime_errors == parsed.error_codes == []
        assert parsed.response is None
        assert parsed.response_text == payload["message"]
    else:
        assert parsed.turn_result.problem.code == payload["problem"]["code"]
        assert parsed.error_codes == [payload["problem"]["code"]]
        assert any(payload["problem"]["code"] in error for error in parsed.runtime_errors)
        assert (parsed.response is not None) == (status == "partial")


@pytest.mark.usefixtures("empty_upload_manifest_http")
@pytest.mark.parametrize("status", ["failed", "refused", "needs_input"])
@pytest.mark.parametrize("include_debug", [False, True])
def test_sse_terminal_status_is_not_misclassified_as_a_response_contract_failure(status, include_debug):
    payload = terminal_payload(status, include_debug=include_debug)
    with patch("src.app.client.requests.post", return_value=sse_http_response(200, payload)) as post:
        result = run_case()

    assert post.call_count == 1
    assert result.response is None
    assert result.response_errors == []
    assert result.scenario_turns[0].response_errors == []
    if status == "needs_input":
        assert result.runtime_errors == []
    else:
        assert payload["problem"]["code"] in result.error_codes
        assert any(payload["problem"]["code"] in error for error in result.runtime_errors)


def test_an_invalid_terminal_contract_remains_a_response_error():
    payload = terminal_payload("failed")
    payload["problem"] = None
    parsed = parse_agent_response(payload)

    assert parsed.turn_result is None
    assert any("turn result invalid" in error for error in parsed.response_errors)
    assert parsed.runtime_errors == []


@pytest.mark.parametrize("status", ["completed", "partial"])
def test_answer_result_still_requires_provenance(status):
    payload = terminal_payload("partial" if status == "partial" else "needs_input")
    payload.update(status=status, response=plain_response("완성된 답변"), message="", missing_slots=[])
    payload["debug"]["answer_provenance"] = None
    parsed = parse_agent_response(payload)

    assert "debug.answer_provenance is missing" in parsed.response_errors


def test_no_answer_result_allows_absent_provenance_without_hiding_other_missing_diagnostics():
    payload = terminal_payload("failed")
    payload["debug"].pop("answer_provenance")
    payload["debug"].pop("tool_calls")
    parsed = parse_agent_response(payload)

    assert "answer_provenance" not in parsed.missing_required_debug_fields
    assert "tool_calls" in parsed.missing_required_debug_fields
    assert parsed.debug_observability_status == "failed"
    assert any("tool_calls" in error for error in parsed.response_errors)
    assert any("provider_schema_invalid" in error for error in parsed.runtime_errors)
