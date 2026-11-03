from __future__ import annotations

from typing import Any

from src.core.contracts.debug import ErrorCode, RetryState
from src.core.contracts.graph_state import DebugState, GraphState
from src.core.contracts.usage import LLMCallRecord


def parse_retry_state(value: Any) -> RetryState:
    if isinstance(value, RetryState):
        return value
    if value is None:
        return RetryState()
    return RetryState.model_validate(value)


def normalize_llm_call_observation(value: Any) -> tuple[list[LLMCallRecord] | None, list[str]]:
    """Validate canonical wire observations without interpreting provider metadata.

    One malformed entry makes the scope unknown: silently dropping it would make
    the remaining calls look like a fully observed turn. Raw diagnostics remain
    available in the caller's response envelope.
    """
    if value is None:
        return None, []
    if not isinstance(value, list):
        return None, ["debug.llm_calls must be a list or null"]
    calls: list[LLMCallRecord] = []
    errors: list[str] = []
    for index, item in enumerate(value):
        try:
            payload = item.model_dump(mode="python") if isinstance(item, LLMCallRecord) else item
            calls.append(LLMCallRecord.model_validate(payload))
        except (TypeError, ValueError) as exc:
            errors.append(f"debug.llm_calls[{index}] invalid: {exc}")
    return (None if errors else calls), errors


def parse_error_codes(value: Any) -> list[ErrorCode]:
    allowed = set(ErrorCode.__args__)  # type: ignore[attr-defined]
    if not isinstance(value, list):
        return []
    parsed: list[ErrorCode] = []
    for item in value:
        # Execution errors and diagnostic codes use different canonical casing.
        code = str(item or "").strip()
        if code in allowed and code not in parsed:
            parsed.append(code)  # type: ignore[arg-type]
    return parsed


def get_debug_state(state: GraphState) -> DebugState:
    return state["debug"] if "debug" in state else DebugState()
