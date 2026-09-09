from pathlib import Path
from unittest.mock import patch

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from src.core.answer_schema import ActionReceipt, finalize_answer, text_document
from src.core.contracts.debug import DebugPayload, TokenUsage
from src.eval.config_models import BenchmarkCase, BenchmarkConfig
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner import _run_single_case
from tests.eval.response_fixtures import plain_response, source_hit, sse_frame, sse_http_response


def run_case():
    return _run_single_case(
        run_id="run-sse",
        endpoint="http://127.0.0.1:8000/",
        fixtures_path=Path("data/benchmarks/fixtures/cases.generated.jsonl"),
        case=BenchmarkCase(case_id="stream-case", category="tool_action", query="예제 저장"),
        timeout_seconds=5,
        judge=LLMJudge(model_name="test-model", enabled=False),
        config=BenchmarkConfig(judge_enabled=False),
    )


def final_payload():
    response = finalize_answer(
        text_document("예제 본문"),
        [],
        actions=[ActionReceipt(kind="save_text", status="success", file_path="output/result.txt")],
    )
    debug = DebugPayload(
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
        observed_hits=[source_hit().model_dump(mode="json")],
        tool_calls=["save_text"],
        tool_call_count=1,
        models_used=["test-model"],
        model_usage_status="llm_used",
        retrieval_diagnostics=[{"tool": "tavily_search", "route": "docs", "status": "success"}],
    ).model_dump(mode="json")
    debug["additional_diagnostic"] = {"measurements": [1, 2, 3]}
    return {"response": response.model_dump(mode="json"), "trace": "Request ID: trace-request", "debug": debug}


def test_stream_preserves_full_response_trace_debug_and_evaluation_contracts():
    """The final frame retains the entire result and all existing evaluation inputs."""
    payload = final_payload()
    response = sse_http_response(200, payload, headers={"x-request-id": "header-request"})
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=response) as post:
        result = run_case()

    assert result.response.model_dump(mode="json") == payload["response"]
    assert result.trace == payload["trace"]
    assert result.debug == payload["debug"]
    assert result.model_dump(mode="json")["debug"] == payload["debug"]
    assert result.request_id == "header-request"
    assert result.token_usage.model_dump() == payload["debug"]["token_usage"]
    assert result.output_tokens == 20
    assert result.cost_usd == pytest.approx(0.000027)
    assert [hit.model_dump(mode="json") for hit in result.observed_hits] == payload["debug"]["observed_hits"]
    assert result.retrieval_diagnostics[0].status == "success"
    assert [action.model_dump(mode="json") for action in result.actions] == payload["response"]["actions"]
    assert result.runtime_errors == []
    assert result.response_errors == []
    assert result.endpoint == "http://127.0.0.1:8000/agent/stream"
    assert post.call_args.kwargs["stream"] is True
    assert post.call_count == 1
    assert response.raw.closed


def test_latency_includes_final_delivery_without_waiting_for_done():
    """End-to-end latency ends at final-response receipt, before result parsing or done."""
    clock = {"now": 10.0}

    def chunks():
        clock["now"] = 11.0
        yield sse_frame("request_started", {})
        clock["now"] = 12.5
        yield sse_frame("final_response", final_payload())
        clock["now"] = 99.0
        raise AssertionError("The client must finish after final_response without awaiting done")

    with patch("src.eval.online_runner.case_runner.requests.post", return_value=sse_http_response(200, chunks=chunks())):
        with patch("src.eval.online_runner.case_runner.time.monotonic", side_effect=lambda: clock["now"]):
            result = run_case()

    assert result.latency_ms_e2e == 2500
    assert result.runtime_errors == []
    assert result.response_errors == []


def test_sse_error_keeps_a_later_final_response_and_fails_the_run():
    """A server error is recorded without discarding a final response that follows it."""
    payload = final_payload()
    response = sse_http_response(200, chunks=[
        sse_frame("error", {"message": "retrieval failed", "stage": "retrieval"}),
        sse_frame("final_response", payload),
    ])
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=response):
        result = run_case()

    assert result.runtime_errors == ["SSE error: retrieval failed"]
    assert result.response.model_dump(mode="json") == payload["response"]
    assert result.debug == payload["debug"]
    assert result.trace == payload["trace"]
    assert result.release_pass is False


@pytest.mark.parametrize(("chunks", "reason"), [
    ([], "empty stream"),
    ([sse_frame("done", {})], "done received"),
    ([sse_frame("request_started", {})], "stream ended"),
])
def test_stream_without_final_response_fails_even_when_http_status_is_200(chunks, reason):
    """Empty, done-only and unfinished streams cannot be mistaken for a completed answer."""
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=sse_http_response(200, chunks=chunks)):
        result = run_case()

    assert result.http_status == 200
    assert result.response is None
    assert result.response_errors == [f"SSE final_response missing ({reason})"]
    assert result.release_pass is False


def test_http_error_remains_distinct_from_a_stream_error():
    """HTTP rejection is reported without trying to interpret the body as an SSE stream."""
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=sse_http_response(503, {"detail": "unavailable"})):
        result = run_case()

    assert result.http_status == 503
    assert result.runtime_errors == ['HTTP 503: {"detail": "unavailable"}']
    assert result.response_errors == []


@pytest.mark.parametrize("before_event", [True, False])
def test_disconnection_is_not_retried_before_or_after_progress(before_event):
    """Transport loss before the final frame is classified as interruption without resubmission."""
    def chunks():
        if not before_event:
            yield sse_frame("request_started", {})
        raise requests.exceptions.ChunkedEncodingError("connection closed")

    response = sse_http_response(200, chunks=chunks())
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=response) as post:
        result = run_case()

    assert result.runtime_errors == ["SSE stream disconnected before final_response: connection closed"]
    assert result.response_errors == []
    assert result.release_pass is False
    assert post.call_count == 1
    assert response.raw.closed


def test_connection_failure_is_distinct_from_stream_interruption_and_not_retried():
    """A request that cannot establish HTTP is recorded separately and submitted only once."""
    with patch("src.eval.online_runner.case_runner.requests.post", side_effect=requests.ConnectionError("connection refused")) as post:
        result = run_case()

    assert result.http_status == 0
    assert result.runtime_errors == ["request failed: connection refused"]
    assert result.response_errors == []
    assert result.release_pass is False
    assert post.call_count == 1


@pytest.mark.parametrize("timeout", [requests.Timeout("read timeout"), ReadTimeoutError(None, "/agent/stream", "read timeout")])
def test_stream_timeout_retains_http_status_and_does_not_retry(timeout):
    """Timeout after HTTP starts is reported as a stream timeout without replaying the request."""
    def chunks():
        yield sse_frame("request_started", {})
        raise timeout

    with patch("src.eval.online_runner.case_runner.requests.post", return_value=sse_http_response(200, chunks=chunks())) as post:
        result = run_case()

    assert result.http_status == 200
    assert result.runtime_errors == ["SSE stream timeout before final_response"]
    assert result.response_errors == []
    assert result.release_pass is False
    assert post.call_count == 1


def test_invalid_stream_is_a_response_contract_error():
    """Malformed SSE JSON fails as a protocol violation instead of a successful empty answer."""
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=sse_http_response(200, chunks=[b"event: final_response\ndata: not-json\n\n"])):
        result = run_case()

    assert result.runtime_errors == []
    assert len(result.response_errors) == 1
    assert result.response_errors[0].startswith("SSE protocol error:")
    assert result.release_pass is False


def test_success_status_with_json_content_type_is_rejected():
    """A proxy or stale server returning JSON cannot satisfy the SSE contract."""
    response = sse_http_response(200, {"response": plain_response("ok")}, headers={"content-type": "application/json"})
    with patch("src.eval.online_runner.case_runner.requests.post", return_value=response):
        result = run_case()

    assert result.response_errors == ["SSE protocol error: expected text/event-stream, received application/json"]
    assert result.release_pass is False
