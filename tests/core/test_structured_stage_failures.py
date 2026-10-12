"""Provider failures remain system failures through the public runtime nodes."""
from __future__ import annotations

import httpx
import pytest
from openai import BadRequestError, InternalServerError

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import ResponseState
from src.core.contracts.debug import RetryState
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import PlannerOutput, RetrievalRewriteOutput, RetrievalTask
from src.core.request_contracts import AnswerContract, ContractEvidence, FormatRequirement, RequestContract, UnresolvedBody, WireRequestContract
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_synthesis_validation import _hit, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


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
        return reply.model_dump(mode="json") if isinstance(reply, (PlannerOutput, RetrievalRewriteOutput)) else reply


def schema_error():
    return BadRequestError(
        "Invalid schema: oneOf is not permitted",
        response=httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")),
        body={"code": "invalid_json_schema", "param": "response_format", "message": "Invalid schema"},
    )






def test_synthesis_invalid_output_without_sources_remains_a_model_failure():
    model = ModelReplies({"answer": "unvalidated model prose"})

    with pytest.raises(LLMCallError) as caught:
        make_synthesize_node(model, excerpt_limits=synthesis_excerpt_limits())(_state([]))

    assert caught.value.problem.code == "model_output_invalid"
    assert model.calls == 2
