"""Opt-in real Decisions checks for exact copies with no-explanation conditions.

From the repository root in Git Bash:
    RUN_LIVE_EXACT_INPUT_VALIDATION=1 uv run --locked pytest -q tests/core/test_exact_input_validation_live.py

These tests send the synthetic request and copied answer below to the configured
Decisions provider and incur API usage. Model judgments can vary across runs.
The normal suite skips them. Preparation uses the existing HTTP fake, but final
semantic validation uses the real provider. No planner, retrieval, synthesis API,
or Slack call runs. A successful answer is saved only inside pytest's tmp_path.
"""

from __future__ import annotations

import os
from hashlib import sha256
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import export_answer_text
from src.core.contracts import PlannerState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, BoundInputText, ContentRequirement,
    ContractEvidence, CopyInputBody, RequestContract, UserTurnSnapshot,
)
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.settings import get_settings
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import mock_chat_model
from tests.core.test_request_slots_delivery import DocumentModel
from tests.synthesis_fixtures import prepare_synthesis_state


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_EXACT_INPUT_VALIDATION") != "1",
    reason="Set RUN_LIVE_EXACT_INPUT_VALIDATION=1 to call the configured Decisions provider.",
)

ORIGINAL = "첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42"
REQUEST = "아래 텍스트만 글자와 줄바꿈을 바꾸지 말고 TXT로 저장해줘. 설명은 붙이지 마.\n\n" + ORIGINAL


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("kind, description", [
    pytest.param(
        "custom", "Do not add an explanation; save only the selected text.",
        id="observed-custom-prohibition",
    ),
    pytest.param(
        "explanation", "Additional explanation beyond the selected input text.",
        id="canonical-explanation-prohibition",
    ),
])
def test_live_exact_copy_keeps_text_and_can_be_saved_with_no_explanation(
    kind: Literal["custom", "explanation"], description: str,
    live_registry: LLMRegistry, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output_dir)
    start = REQUEST.index(ORIGINAL)
    contract = RequestContract(
        body=CopyInputBody(source=BoundInputText(
            turn_id="current", text=ORIGINAL, start=start, end=start + len(ORIGINAL),
            content_hash=sha256(ORIGINAL.encode("utf-8")).hexdigest(),
        )),
        actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("e1",))),
        answer=AnswerContract(content=(ContentRequirement(
            kind=kind, mode="forbidden", description=description, evidence_ids=("e2",),
        ),)),
        evidence=(
            ContractEvidence(
                id="e1", turn_id="current",
                quote="TXT로 저장해줘",
                scope="actions.save_text", interpretation="instruction",
            ),
            ContractEvidence(
                id="e2", turn_id="current", quote="설명은 붙이지 마.",
                scope="answer.content", interpretation="negation",
            ),
        ),
    )
    state = build_graph_state_input(
        user_input=REQUEST, current_turn_id="current", session_id="exact-input-live",
        user_turns=(UserTurnSnapshot(turn_id="current", text=REQUEST),),
        messages=[HumanMessage(content=REQUEST, id="current")], request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        retry=RetryState(max_retries=0, max_content_repairs=0),
    )
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(mock_chat_model(DocumentModel().invoke))(state))
    assert export_answer_text(state["response"].result) == ORIGINAL

    state.update(make_post_synthesis_validation_node(
        False, decisions_client=live_registry.decisions_client,
    )(state))

    assert state["response"].kind == "answer", "\n".join(state["debug"].validation_events)
    assert export_answer_text(state["response"].result) == ORIGINAL
    state.update(make_action_postprocess_node(build_save_text_tool(), None, False)(state))
    answer = state["response"].result
    assert export_answer_text(answer) == ORIGINAL
    assert len(answer.actions) == 1
    receipt = answer.actions[0]
    assert receipt.kind == "save_text" and receipt.status == "success"
    assert Path(receipt.file_path).read_bytes() == ORIGINAL.encode("utf-8-sig")
