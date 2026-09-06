from langchain_core.messages import AIMessage

from src.core.contracts import DebugState, PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input, normalize_graph_update


def build_test_graph_state(*, user_input: str, messages: list | None = None, **kwargs):
    return build_graph_state_input(
        user_input=user_input,
        messages=messages or [],
        **kwargs,
    )


def build_test_state(payload: dict):
    raw = dict(payload)
    state = build_graph_state_input(
        user_input=str(raw.pop("user_input", "") or ""),
        messages=raw.pop("messages", []) or [],
        retriever=raw.pop("retriever", None),
        session_metadata=raw.pop("session_metadata", None),
        memory_summary=raw.pop("memory_summary", None),
        previous_response=raw.pop("previous_response", None),
    )
    if "planner" in raw:
        state["planner"] = PlannerState.model_validate(raw.pop("planner"))
    elif any(key in raw for key in ("planner_output", "planner_status", "planner_diagnostics", "guided_followup")):
        state["planner"] = PlannerState(
            output=raw.pop("planner_output", None) or PlannerState().output,
            status=raw.pop("planner_status", "llm") or "llm",
            diagnostics=raw.pop("planner_diagnostics", None) or PlannerState().diagnostics,
            guided_followup=raw.pop("guided_followup", None),
        )
    if "retrieval" in raw:
        state["retrieval"] = RetrievalState.model_validate(raw.pop("retrieval"))
    elif "retrieved_hits" in raw:
        state["retrieval"] = RetrievalState(hit_log=raw.pop("retrieved_hits"))
    if "retry" in raw:
        state["retry"] = RetryState.model_validate(raw.pop("retry"))
    elif "retry_context" in raw or "needs_retry" in raw:
        retry = dict(raw.pop("retry_context", {}) or {})
        if "needs_retry" in raw:
            retry["needs_retry"] = raw.pop("needs_retry")
        state["retry"] = RetryState.model_validate(retry)
    if "response" in raw:
        state["response"] = ResponseState.model_validate(raw.pop("response"))
    elif any(key in raw for key in ("result", "evidence_packet", "synthesis_attempt")):
        state["response"] = ResponseState(
            result=raw.pop("result", None) or ResponseState().result,
            evidence_packet=raw.pop("evidence_packet", []),
            synthesis_attempt=int(raw.pop("synthesis_attempt", 0) or 0),
        )
    debug = raw.pop("debug", None)
    if debug is None:
        debug = {key: raw.pop(key) for key in list(raw) if key in DebugState.model_fields}
    if debug:
        state["debug"] = DebugState.model_validate(debug)
    if raw:
        raise ValueError(f"Unknown test state fields: {sorted(raw)}")
    return normalize_graph_update(state)


class _FailingPlannerLLM:
    def invoke(self, _messages):
        raise RuntimeError("planner exploded")


class _InvalidPlannerLLM:
    def invoke(self, _messages):
        return {
            "use_retrieval": False,
            "tasks": [
                {"route": "docs", "query": "numpy", "k": 4},
            ],
        }


class _CaptureSynthesizeLLM:
    def __init__(
        self,
        *,
        content: str = "synth result",
        response_metadata: dict | None = None,
        usage_metadata: dict | None = None,
    ):
        self.last_messages = None
        self.content = content
        self.response_metadata = response_metadata
        self.usage_metadata = usage_metadata

    def invoke(self, messages):
        self.last_messages = messages
        kwargs = {}
        if self.response_metadata is not None:
            kwargs["response_metadata"] = self.response_metadata
        if self.usage_metadata is not None:
            kwargs["usage_metadata"] = self.usage_metadata
        return AIMessage(content=self.content, **kwargs)


class _CaptureStructuredSynthesizeLLM:
    def __init__(
        self,
        payload=None,
        *,
        include_raw: bool = False,
        raw_message: AIMessage | None = None,
        parsing_error: Exception | None = None,
    ):
        self.last_messages = None
        self.payload = payload if payload is not None else {
            "blocks": [{"type": "paragraph", "content": [{"text": "synth result", "basis": "interaction", "refs": []}]}],
        }
        self.include_raw = include_raw
        self.raw_message = raw_message or AIMessage(
            content="",
            response_metadata={
                "model_name": "gpt-5-mini",
                "token_usage": {
                    "prompt_tokens": 11,
                    "completion_tokens": 3,
                    "total_tokens": 14,
                },
            },
            usage_metadata={
                "input_tokens": 11,
                "output_tokens": 3,
                "total_tokens": 14,
            },
        )
        self.parsing_error = parsing_error

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.last_messages = messages
        if self.include_raw:
            return {
                "raw": self.raw_message,
                "parsed": self.payload,
                "parsing_error": self.parsing_error,
            }
        return self.payload


class _TimeoutStructuredSynthesizeLLM:
    def __init__(self):
        self.last_messages = None
        self.call_count = 0

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.last_messages = messages
        self.call_count += 1
        raise TimeoutError("structured timeout")


class _CapturePlannerLLM:
    def __init__(
        self,
        planner_output,
        *,
        include_raw: bool = False,
        raw_message: AIMessage | None = None,
        parsing_error: Exception | None = None,
    ):
        self.planner_output = planner_output
        self.last_messages = None
        self.call_count = 0
        self.include_raw = include_raw
        self.raw_message = raw_message or AIMessage(
            content="",
            response_metadata={
                "model_name": "gpt-5-nano",
                "token_usage": {
                    "prompt_tokens": 7,
                    "completion_tokens": 2,
                    "total_tokens": 9,
                },
            },
            usage_metadata={
                "input_tokens": 7,
                "output_tokens": 2,
                "total_tokens": 9,
            },
        )
        self.parsing_error = parsing_error

    def invoke(self, messages):
        self.last_messages = messages
        self.call_count += 1
        if self.include_raw:
            return {
                "raw": self.raw_message,
                "parsed": self.planner_output,
                "parsing_error": self.parsing_error,
            }
        return self.planner_output


class _CaptureSummaryLLM:
    def __init__(self, content: str = "summary line"):
        self.last_messages = None
        self.content = content

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(
            content=self.content,
            response_metadata={
                "model_name": "gpt-5-mini",
                "token_usage": {
                    "prompt_tokens": 13,
                    "completion_tokens": 4,
                    "total_tokens": 17,
                },
            },
            usage_metadata={
                "input_tokens": 13,
                "output_tokens": 4,
                "total_tokens": 17,
            },
        )


def _tool_payload(hits: list[dict] | None = None, **diagnostics):
    return {
        "hits": list(hits or []),
        "diagnostics": diagnostics,
    }
