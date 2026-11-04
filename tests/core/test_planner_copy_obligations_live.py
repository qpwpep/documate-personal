"""Opt-in planner checks for copy fidelity and independent answer obligations.

From the repository root in Git Bash:
    RUN_LIVE_PLANNER_COPY_OBLIGATIONS=1 uv run --locked pytest -q -s tests/core/test_planner_copy_obligations_live.py

These tests send the fixed synthetic conversation below to the configured planner
provider and incur API usage. Model output can vary across runs. The normal suite
skips them. Only planning runs: no retrieval, synthesis, saving, or Slack calls.
With -s, each case prints its synthetic query, provider response text, resolved
contract, and tasks for a local diagnostic log; credentials are never printed.
"""

from __future__ import annotations

import json
import os

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.request_contracts import UserTurnSnapshot
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.llm_boundary import capture_structured_calls
from src.infra.settings import get_settings
from src.runtime.nodes.planner import make_planner_node


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_PLANNER_COPY_OBLIGATIONS") != "1",
    reason="Set RUN_LIVE_PLANNER_COPY_OBLIGATIONS=1 to call the configured planner provider.",
)

PREVIOUS_TEXT = "첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42"


@pytest.fixture(scope="module")
def live_planner_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("query, independent_conditions", [
    pytest.param(
        "방금 답변을 한 글자도 수정하지 말고 그대로 파일로 한 번 더 보관해 줘. 새 설명은 필요 없어.",
        False,
        id="copy-fidelity-does-not-create-extra-obligations",
    ),
    pytest.param(
        "방금 답변을 그대로 파일로 저장해줘. 본문에 계절 이름과 숫자 42가 모두 있어야 하고, 표 형식은 금지야.",
        True,
        id="independent-content-and-table-prohibition-remain",
    ),
])
def test_live_planner_separates_copy_fidelity_from_independent_answer_conditions(
    query: str, independent_conditions: bool, live_planner_registry: LLMRegistry,
) -> None:
    previous = finalize_answer(text_document(PREVIOUS_TEXT), [])
    state = build_graph_state_input(
        user_input=query, current_turn_id="copy-request",
        user_turns=(UserTurnSnapshot(turn_id="copy-request", text=query),),
        messages=[
            AIMessage(id="previous-answer", content=export_answer_text(previous)),
            HumanMessage(id="copy-request", content=query),
        ],
        previous_response=previous,
    )
    planner = make_planner_node(live_planner_registry.llm_planner, False)
    updates = None

    with capture_structured_calls() as calls:
        try:
            updates = planner(state)
        finally:
            provider_outputs = []
            for call in calls:
                if call.stage != "planner":
                    continue
                raw = call.output.get("raw") if isinstance(call.output, dict) else call.output
                content = raw.content if isinstance(raw, AIMessage) else None
                raw_text = "".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
                ) if isinstance(content, list) else content
                provider_outputs.append({"attempt": call.attempt, "raw_text": raw_text})
            print(json.dumps({
                "query": query,
                "provider_outputs": provider_outputs,
                "resolved_contract": updates["runtime"].request_contract.model_dump(mode="json")
                if updates is not None else None,
                "tasks": [task.model_dump(mode="json") for task in updates["planner"].output.tasks]
                if updates is not None else None,
            }, ensure_ascii=False))

    contract = updates["runtime"].request_contract
    # Explicit layout requirements are bound to a source-preserving transform.
    assert contract.body.kind == ("transform_answer" if independent_conditions else "copy_answer")
    assert contract.body.source.ref == "previous"
    assert contract.body.source.response_hash == previous.content_hash
    assert contract.actions.save_text.intent == "requested"
    assert not updates["planner"].output.use_retrieval
    assert updates["planner"].output.tasks == []
    if independent_conditions:
        required = [item for item in contract.answer.content if item.mode == "required"]
        assert required
        assert any("42" in item.description for item in required)
        assert any(item.kind == "table" and item.mode == "forbidden" for item in contract.answer.format)
    else:
        assert contract.answer.content == ()
        assert contract.answer.format == ()
