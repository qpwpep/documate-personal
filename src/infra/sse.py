from __future__ import annotations

import codecs
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SSEEvent:
    event: str
    data: dict[str, Any]


def _iter_lines(chunks: Iterable[str | bytes]) -> Iterator[str]:
    decoder = codecs.getincrementaldecoder("utf-8")()
    line: list[str] = []
    skip_lf = False
    at_start = True

    def decoded_chunks() -> Iterator[str]:
        for chunk in chunks:
            if isinstance(chunk, bytes):
                yield decoder.decode(chunk)
            else:
                if decoder.getstate()[0]:
                    raise ValueError("Incomplete UTF-8 sequence before text SSE chunk")
                yield chunk
        yield decoder.decode(b"", final=True)

    for text in decoded_chunks():
        for character in text:
            if at_start:
                at_start = False
                if character == "\ufeff":
                    continue
            if skip_lf:
                skip_lf = False
                if character == "\n":
                    continue
            if character in "\r\n":
                yield "".join(line)
                line.clear()
                skip_lf = character == "\r"
            else:
                line.append(character)
    if line:
        yield "".join(line)


def iter_sse_events(chunks: Iterable[str | bytes]) -> Iterator[SSEEvent]:
    """Decode complete agent SSE frames without losing split UTF-8 characters.

    Only data-bearing frames are events. Completion and transport failures are
    handled by the caller; malformed JSON or an unfinished data frame is an error.
    """
    event_name = "message"
    data_lines: list[str] = []
    for line in _iter_lines(chunks):
        if not line:
            if data_lines:
                try:
                    data = json.loads("\n".join(data_lines))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in SSE event {event_name!r}") from exc
                if not isinstance(data, dict):
                    raise ValueError(f"SSE event {event_name!r} data must be an object")
                yield SSEEvent(event=event_name, data=data)
            event_name = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]
        if field == "event":
            event_name = value or "message"
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        raise ValueError(f"Incomplete SSE event {event_name!r}")
