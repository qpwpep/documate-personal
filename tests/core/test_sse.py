import json

import pytest
from hypothesis import given, strategies as st

from src.infra.sse import SSEEvent, iter_sse_events


@given(text=st.text(), cuts=st.lists(st.integers(min_value=0, max_value=500), max_size=20))
def test_sse_preserves_json_across_arbitrary_utf8_chunks(text, cuts):
    """Splitting an SSE response at byte boundaries preserves its complete JSON data."""
    data = {"response": {"text": text}, "trace": "요청", "debug": {"count": 2}}
    frame = ("event: final_response\ndata: " + json.dumps(data, ensure_ascii=False) + "\n\n").encode("utf-8")
    boundaries = sorted({0, len(frame), *(cut % (len(frame) + 1) for cut in cuts)})
    chunks = [frame[start:end] for start, end in zip(boundaries, boundaries[1:])]
    assert list(iter_sse_events(chunks)) == [SSEEvent(event="final_response", data=data)]


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
def test_sse_ignores_comments_and_combines_multiline_data(newline):
    """SSE comments and metadata cannot masquerade as progress or final responses."""
    wire = newline.join([
        "\ufeff: keepalive", "", "id: ignored", "", "event: progress_snapshot",
        'data: {"stage": "retrieval",', 'data: "summary": "근거"}', "", "",
    ])
    assert list(iter_sse_events(wire)) == [
        SSEEvent(event="progress_snapshot", data={"stage": "retrieval", "summary": "근거"}),
    ]


@pytest.mark.parametrize("data", ["not json", "[]", "null", '"text"'])
def test_sse_rejects_invalid_event_data(data):
    """Agent events must contain JSON objects rather than accepting corrupt payloads."""
    with pytest.raises(ValueError):
        list(iter_sse_events([f"event: final_response\ndata: {data}\n\n"]))


def test_sse_rejects_a_truncated_final_frame():
    """EOF before the blank frame terminator cannot turn a partial event into success."""
    with pytest.raises(ValueError, match="Incomplete SSE event"):
        list(iter_sse_events(['event: final_response\ndata: {"response": {}}\n']))


def test_sse_rejects_a_truncated_utf8_character():
    """An incomplete UTF-8 sequence is reported instead of silently replacing content."""
    with pytest.raises(ValueError):
        list(iter_sse_events([b'event: final_response\ndata: {"text": "\xec']))
