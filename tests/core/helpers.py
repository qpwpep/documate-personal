from langchain_core.messages import AIMessage

from src.contracts.graph_state import (
    DebugState,
    PlannerState,
    ResponseState,
    RetrievalState,
    RetryState,
    build_graph_state_input,
    normalize_state_updates,
)


def build_test_graph_state(*, user_input: str, messages: list | None = None, **kwargs):
    return build_graph_state_input(
        user_input=user_input,
        messages=messages or [],
        **kwargs,
    )


def build_legacy_state(payload: dict):
    raw = dict(payload)
    state = build_graph_state_input(
        user_input=str(raw.pop("user_input", "") or ""),
        messages=raw.pop("messages", []) or [],
        retriever=raw.pop("retriever", None),
        session_metadata=raw.pop("session_metadata", None),
        memory_summary=raw.pop("memory_summary", None),
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
    elif "retrieved_evidence" in raw:
        state["retrieval"] = RetrievalState(evidence_log=raw.pop("retrieved_evidence"))

    if "retry" in raw:
        state["retry"] = RetryState.model_validate(raw.pop("retry"))
    elif any(key in raw for key in ("retry_context", "needs_retry")):
        retry_payload = raw.pop("retry_context", {}) or {}
        if not isinstance(retry_payload, dict):
            retry_payload = {}
        if "needs_retry" in raw:
            retry_payload = dict(retry_payload)
            retry_payload["needs_retry"] = bool(raw.pop("needs_retry"))
        state["retry"] = RetryState.model_validate(retry_payload)

    if "response" in raw:
        state["response"] = ResponseState.model_validate(raw.pop("response"))
    elif any(key in raw for key in ("final_answer", "response_payload", "synthesis_output", "synthesis_attempt")):
        state["response"] = ResponseState(
            final_answer=raw.pop("final_answer", "") or "",
            payload=raw.pop("response_payload", None) or ResponseState().payload,
            synthesis_output=raw.pop("synthesis_output", None) or ResponseState().synthesis_output,
            synthesis_attempt=int(raw.pop("synthesis_attempt", 0) or 0),
        )

    if "debug" in raw:
        state["debug"] = DebugState.model_validate(raw.pop("debug"))
    elif any(
        key in raw
        for key in (
            "tool_calls",
            "tool_call_count",
            "token_usage",
            "model_name",
            "models_used",
            "llm_calls",
            "errors",
            "planner_errors",
            "observed_evidence",
            "retry_context",
            "retrieval_diagnostics",
            "planner_diagnostics",
            "latency_breakdown",
            "retrieval_errors",
            "synthesis_errors",
            "validation_errors",
            "action_errors",
            "latency_trace",
        )
    ):
        debug_payload = {
            key: raw.pop(key)
            for key in list(raw.keys())
            if key
            in {
                "tool_calls",
                "tool_call_count",
                "token_usage",
                "model_name",
                "models_used",
                "llm_calls",
                "errors",
                "planner_errors",
                "observed_evidence",
                "retry_context",
                "retrieval_diagnostics",
                "planner_diagnostics",
                "latency_breakdown",
                "retrieval_errors",
                "synthesis_errors",
                "validation_errors",
                "action_errors",
                "latency_trace",
            }
        }
        state["debug"] = DebugState.model_validate(debug_payload)

    return normalize_state_updates(state)


class _ToolWrapper:
    def __init__(self, func):
        self.func = func


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
        self.payload = payload or {
            "answer": "synth result",
            "claims": [],
            "confidence": None,
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


class _StructuredThenPlainFallbackSynthesizeLLM:
    def __init__(self):
        self.structured_messages = None
        self.plain_messages = None

    def with_structured_output(self, *_args, **_kwargs):
        parent = self

        class _StructuredWrapper:
            def invoke(self, messages):
                parent.structured_messages = messages
                return {
                    "raw": AIMessage(
                        content="",
                        response_metadata={
                            "model_name": "gpt-5-mini",
                            "token_usage": {
                                "prompt_tokens": 9,
                                "completion_tokens": 2,
                                "total_tokens": 11,
                            },
                        },
                        usage_metadata={
                            "input_tokens": 9,
                            "output_tokens": 2,
                            "total_tokens": 11,
                        },
                    ),
                    "parsed": None,
                    "parsing_error": ValueError("schema mismatch"),
                }

        return _StructuredWrapper()

    def invoke(self, messages):
        self.plain_messages = messages
        return AIMessage(
            content="plain fallback answer",
            response_metadata={
                "model_name": "gpt-5-mini",
                "token_usage": {
                    "prompt_tokens": 15,
                    "completion_tokens": 6,
                    "total_tokens": 21,
                },
            },
            usage_metadata={
                "input_tokens": 15,
                "output_tokens": 6,
                "total_tokens": 21,
            },
        )


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


def _tool_payload(evidence: list[dict] | None = None, **diagnostics):
    return {
        "evidence": list(evidence or []),
        "diagnostics": diagnostics,
    }
