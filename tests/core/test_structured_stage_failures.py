"""Provider failures remain system failures through the public runtime nodes."""
from __future__ import annotations

import pytest

from src.core.llm_errors import LLMCallError
from src.runtime.nodes.synthesis import make_synthesize_node
from tests.core.test_synthesis_validation import _state
from tests.synthesis_fixtures import prepare_synthesis_state


class ModelReplies:
    """Only the external model boundary is replaced; all node logic stays real."""

    def __init__(self, *replies):
        self.replies = replies
        self.calls = 0

    def invoke(self, messages):
        reply = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        if isinstance(reply, Exception):
            raise reply
        return reply


def test_synthesis_invalid_output_without_sources_remains_a_model_failure():
    model = ModelReplies({"answer": "unvalidated model prose"})

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(model)(prepare_synthesis_state(_state([])))

    assert caught.value.problem.code == "model_output_invalid"
    assert model.calls == 2
