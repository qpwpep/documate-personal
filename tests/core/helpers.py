from langchain_core.messages import AIMessage


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
