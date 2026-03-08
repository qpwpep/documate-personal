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
    def __init__(self):
        self.last_messages = None

    def invoke(self, messages):
        self.last_messages = messages
        return AIMessage(content="synth result")


class _CaptureStructuredSynthesizeLLM:
    def __init__(self, payload=None):
        self.last_messages = None
        self.payload = payload or {
            "answer": "synth result",
            "claims": [],
            "confidence": None,
        }

    def with_structured_output(self, *_args, **_kwargs):
        return self

    def invoke(self, messages):
        self.last_messages = messages
        return self.payload


class _CapturePlannerLLM:
    def __init__(self, planner_output):
        self.planner_output = planner_output
        self.last_messages = None
        self.call_count = 0

    def invoke(self, messages):
        self.last_messages = messages
        self.call_count += 1
        return self.planner_output


def _tool_payload(evidence: list[dict] | None = None, **diagnostics):
    return {
        "evidence": list(evidence or []),
        "diagnostics": diagnostics,
    }
