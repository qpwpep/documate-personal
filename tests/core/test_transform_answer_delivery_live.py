"""Opt-in checks separating answer edits from subsequent TXT delivery.

From the repository root in Git Bash:
    RUN_LIVE_TRANSFORM_ANSWER_DELIVERY=1 uv run --locked pytest -q -s tests/core/test_transform_answer_delivery_live.py

The fixed conversation reproduces the observed 42-to-43 correction. One separate
planner test checks the edit and save intent. Four answer candidates run against
both the original frozen contract and the later browser-failure contract,
independently of that planner output. Each makes two paid Decisions calls using
the public node's unchanged grounding and request_fulfilled inputs and questions.
The grounding call retains the original request, conversation and interpretation
evidence; the operation view checks the requested edit separately. Both returned
judgments are replayed through the public node without another API call. Model
judgments can vary. The default suite skips these nine tests and their seventeen
paid calls: one planner call and two Decisions calls for each of eight candidates.
No retrieval, synthesis, file save or notification runs. Diagnostic JSON stays
in pytest's temporary directories and is also printed for the caller's local log.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import re
from typing import Literal

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.message_utils import extract_text_content
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, BoundAnswerReference, ContractEvidence, RequestContract,
    TransformAnswerBody, UserTurnSnapshot,
)
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.llm_boundary import (
    StructuredModel, capture_llm_diagnostics, capture_structured_calls, request_call_budget,
)
from src.infra.settings import get_settings
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_TRANSFORM_ANSWER_DELIVERY") != "1",
    reason="Set RUN_LIVE_TRANSFORM_ANSWER_DELIVERY=1 to call the configured planner and Decisions providers.",
)

PREVIOUS_TEXT = "첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 42"
CORRECT_TEXT = "첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 43"
WRONG_TEXT = "첫 줄: 봄  여름\n둘째 줄: 가을 🧪\n마지막 줄: 44"
UNREQUESTED_FACT_TEXT = CORRECT_TEXT + "\n이 프로그램은 실제로 실행되어 성공했습니다."
FIRST_REQUEST = "아래 텍스트만 글자와 줄바꿈을 바꾸지 말고 TXT로 저장해줘. 설명은 붙이지 마.\n\n" + PREVIOUS_TEXT
COPY_REQUEST = "방금 답변을 한 글자도 수정하지 말고 그대로 파일로 한 번 더 보관해 줘. 새 설명은 필요 없어."
REQUEST = "아니, 마지막 줄의 42만 43으로 고쳐서 새 TXT 파일로 저장해줘. 나머지 글자와 줄바꿈은 그대로 둬."
FIRST_TURN = "user:02e76c9a70dc4b18b5676a7d11fa08c6"
COPY_TURN = "user:0d09836a3d0d4bfa88ac439418cec0e4"
CURRENT_TURN = "user:1b49a092edc14ea7a42f79b2a88be5b7"
BODY_INSTRUCTION = "직전 답변의 마지막 줄에서 42를 43으로 바꾸고, 그 외 글자와 줄바꿈은 그대로 유지한다."
EVIDENCE_IDS = (
    "ev:f6579247-f8bb-441d-82b5-49072dea34eb:1:user:1b49a092edc14ea7a42f79b2a88be5b7:1",
    "ev:f6579247-f8bb-441d-82b5-49072dea34eb:1:user:1b49a092edc14ea7a42f79b2a88be5b7:2",
    "ev:f6579247-f8bb-441d-82b5-49072dea34eb:1:user:1b49a092edc14ea7a42f79b2a88be5b7:3",
)
LATEST_CURRENT_TURN = "user:d583778a372640f9b123fea7a4fc70df"
LATEST_BODY_INSTRUCTION = "답변에서 마지막 줄의 42만 43으로 바꾸고, 나머지 모든 글자와 줄바꿈은 그대로 유지한다."
LATEST_EVIDENCE_IDS = (
    "ev:37df5485-cf55-43f3-9ad8-3d8729a6401e:1:user:d583778a372640f9b123fea7a4fc70df:1",
    "ev:37df5485-cf55-43f3-9ad8-3d8729a6401e:1:user:d583778a372640f9b123fea7a4fc70df:2",
)
ContractVariant = Literal["frozen", "latest-browser"]


def _conversation_state(current_turn_id: str = CURRENT_TURN) -> GraphState:
    previous = finalize_answer(text_document(PREVIOUS_TEXT), [])
    assert previous.content_hash == "03aac20b69aefc21a39d37d91a7296a3fa1a74c3ee339f2a4dc7e4c729a2ee4e"
    return build_graph_state_input(
        user_input=REQUEST, current_turn_id=current_turn_id,
        user_turns=tuple(UserTurnSnapshot(turn_id=turn_id, text=text) for turn_id, text in (
            (FIRST_TURN, FIRST_REQUEST), (COPY_TURN, COPY_REQUEST), (current_turn_id, REQUEST),
        )),
        messages=[
            HumanMessage(id=FIRST_TURN, content=FIRST_REQUEST),
            AIMessage(id="94760050-582b-4051-a7c3-a7f352e443b1", content=PREVIOUS_TEXT),
            HumanMessage(id=COPY_TURN, content=COPY_REQUEST),
            AIMessage(id="446ccafa-bc1a-4caa-b7d2-77db372c754b", content=PREVIOUS_TEXT),
            HumanMessage(id=current_turn_id, content=REQUEST),
        ],
        previous_response=previous,
        retry=RetryState(max_retries=0, max_content_repairs=1),
    )


def _frozen_state(contract_variant: ContractVariant) -> GraphState:
    state = _conversation_state(CURRENT_TURN if contract_variant == "frozen" else LATEST_CURRENT_TURN)
    if contract_variant == "frozen":
        contract = RequestContract(
            request_id="f6579247-f8bb-441d-82b5-49072dea34eb", revision=1, relation="correction",
            actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=[EVIDENCE_IDS[0]])),
            body=TransformAnswerBody(
                source=BoundAnswerReference(ref="previous", response_hash=state["runtime"].previous_response.content_hash),
                instruction=BODY_INSTRUCTION, evidence_ids=EVIDENCE_IDS[1:],
            ),
            evidence=[
                ContractEvidence(
                    id=EVIDENCE_IDS[0], turn_id=CURRENT_TURN,
                    quote="아니, 마지막 줄의 42만 43으로 고쳐서 새 TXT 파일로 저장해줘.",
                    scope="current_request", interpretation="correction",
                ),
                ContractEvidence(
                    id=EVIDENCE_IDS[1], turn_id=CURRENT_TURN, quote="마지막 줄의 42만 43으로 고쳐서",
                    scope="body.instruction", interpretation="instruction",
                ),
                ContractEvidence(
                    id=EVIDENCE_IDS[2], turn_id=CURRENT_TURN, quote="나머지 글자와 줄바꿈은 그대로 둬.",
                    scope="body.instruction", interpretation="instruction",
                ),
            ],
        )
    else:
        # Preserve the independently captured 634a718c contract, including its
        # two evidence records and the current turn's server-bound identifiers.
        contract = RequestContract(
            request_id="37df5485-cf55-43f3-9ad8-3d8729a6401e", revision=1, relation="correction",
            actions=ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=[LATEST_EVIDENCE_IDS[0]])),
            body=TransformAnswerBody(
                source=BoundAnswerReference(ref="previous", response_hash=state["runtime"].previous_response.content_hash),
                instruction=LATEST_BODY_INSTRUCTION, evidence_ids=LATEST_EVIDENCE_IDS,
            ),
            evidence=[
                ContractEvidence(
                    id=LATEST_EVIDENCE_IDS[0], turn_id=LATEST_CURRENT_TURN,
                    quote="아니, 마지막 줄의 42만 43으로 고쳐서 새 TXT 파일로 저장해줘.",
                    scope="current_request", interpretation="correction",
                ),
                ContractEvidence(
                    id=LATEST_EVIDENCE_IDS[1], turn_id=LATEST_CURRENT_TURN,
                    quote="나머지 글자와 줄바꿈은 그대로 둬.",
                    scope="body.instruction", interpretation="instruction",
                ),
            ],
        )
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[]))
    return state


def _plan_state(planner_model: StructuredModel, record: Path) -> GraphState:
    state = _conversation_state()
    updates = None
    with capture_llm_diagnostics(), capture_structured_calls() as calls:
        request_call_budget("planner").max_calls = 1
        try:
            updates = make_planner_node(planner_model, False)(state)
        finally:
            provider_outputs = []
            for call in calls:
                raw = call.output.get("raw") if isinstance(call.output, dict) else call.output
                provider_outputs.append({
                    "attempt": call.attempt,
                    "raw_text": extract_text_content(raw.content) if isinstance(raw, AIMessage) else None,
                    "error": str(call.exception) if call.exception is not None else None,
                })
            observation = {
                "query": REQUEST, "provider_outputs": provider_outputs,
                "resolved_contract": updates["runtime"].request_contract.model_dump(mode="json")
                if updates is not None else None,
                "tasks": [task.model_dump(mode="json") for task in updates["planner"].output.tasks]
                if updates is not None else None,
            }
            record.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({**observation, "record": str(record)}, ensure_ascii=True))
    assert len(calls) == 1, str(record)
    state.update(updates)
    contract = state["runtime"].request_contract
    assert contract.body.kind == "transform_answer", str(record)
    assert contract.body.source.ref == "previous"
    assert contract.body.source.response_hash == state["runtime"].previous_response.content_hash
    assert contract.actions.save_text.intent == "requested"
    assert contract.actions.slack_notify.intent == "not_requested"
    assert contract.answer.content == contract.answer.format == contract.answer.sentence_counts == ()
    assert not state["planner"].output.use_retrieval
    assert state["planner"].output.tasks == []
    assert "42" in contract.body.instruction and "43" in contract.body.instruction
    # This fixture requests one text edit. Delivery belongs only to its action,
    # regardless of whether the planner describes that edit in Korean or English.
    assert re.search(r"\b(?:save|saved|saving|export|txt)\b|저장|보관|내보내", contract.body.instruction,
                     flags=re.IGNORECASE) is None, str(record)
    assert export_answer_text(state["runtime"].previous_response) == PREVIOUS_TEXT
    return state


def _post_state(reference_state: GraphState, text: str) -> GraphState:
    state = deepcopy(reference_state)
    contract = state["runtime"].request_contract
    state["response"] = ResponseState(
        result=finalize_answer(text_document(text), []), synthesis_attempt=1,
        request_id=contract.request_id, contract_revision=contract.revision,
    )
    return state


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


def test_live_transform_planner_separates_edit_and_delivery(live_registry: LLMRegistry, tmp_path: Path) -> None:
    _plan_state(live_registry.llm_planner, tmp_path / "planner.json")


@pytest.mark.parametrize("contract_variant", ["frozen", "latest-browser"])
@pytest.mark.parametrize("text,expected_fulfilled,expected_grounded", [
    pytest.param(CORRECT_TEXT, True, True, id="correct-final-43"),
    pytest.param(PREVIOUS_TEXT, False, None, id="unchanged-final-42"),
    pytest.param(WRONG_TEXT, False, None, id="wrong-final-44"),
    pytest.param(UNREQUESTED_FACT_TEXT, False, False, id="unrequested-execution-fact"),
])
def test_live_transform_edits_the_bound_answer_before_txt_delivery(
    text: str, expected_fulfilled: bool, expected_grounded: bool | None, contract_variant: ContractVariant,
    live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    frozen_state = _frozen_state(contract_variant)
    state = _post_state(frozen_state, text)
    original = state["runtime"].previous_response.model_copy(deep=True)
    capture = DecisionsBoundary()
    make_post_synthesis_validation_node(False, decisions_client=capture, pass_threshold=0.8)(state)
    body_call, = [call for call in capture.requests
                  if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    assert [question["name"] for question in body_call["questions"]] == ["request_fulfilled"]
    payload = json.loads(body_call["input"])
    contract = frozen_state["runtime"].request_contract
    assert "latest_user_request" not in payload
    assert "conversation_context" not in payload
    assert "evidence" not in payload["confirmed_request"]
    assert payload["confirmed_request"]["body"] == contract.body.model_dump(mode="json", exclude={"evidence_ids"})
    assert payload["bound_source_response"]["content"] == original.content.model_dump(mode="json")
    assert payload["bound_source_response"]["content_hash"] == original.content_hash
    assert payload["rendered_response"]["body"] == state["response"].result.content.model_dump(mode="json")
    grounding_call, = [call for call in capture.requests
                       if any(question["name"] == "grounded_0" for question in call["questions"])]
    assert [question["name"] for question in grounding_call["questions"]] == ["grounded_0"]
    grounding_payload = json.loads(grounding_call["input"])
    assert grounding_payload["latest_user_request"] == REQUEST
    assert grounding_payload["confirmed_request"]["body"] == contract.body.model_dump(mode="json")
    assert grounding_payload["confirmed_request"]["evidence"] == [
        item.model_dump(mode="json") for item in contract.evidence
    ]
    assert grounding_payload["conversation_context"]["messages"] == [
        {"role": "user", "content": FIRST_REQUEST},
        {"role": "assistant", "content": PREVIOUS_TEXT},
        {"role": "user", "content": COPY_REQUEST},
        {"role": "assistant", "content": PREVIOUS_TEXT},
    ]

    with capture_llm_diagnostics():
        request_call_budget("post_synthesis_validation").max_calls = 2
        grounding_judgment = live_registry.decisions_client.evaluate(
            input=grounding_call["input"], questions=grounding_call["questions"],
            stage="post_synthesis_validation", reserve_calls=1,
        )
        operation_judgment = live_registry.decisions_client.evaluate(
            input=body_call["input"], questions=body_call["questions"], stage="post_synthesis_validation",
        )
    probabilities = {**grounding_judgment.probabilities, **operation_judgment.probabilities}
    refused_names = grounding_judgment.refused_names | operation_judgment.refused_names
    replay = DecisionsBoundary(probabilities=probabilities, refused_names=refused_names)
    updates = make_post_synthesis_validation_node(False, decisions_client=replay, pass_threshold=0.8)(state)
    for name, original_call in (("grounded_0", grounding_call), ("request_fulfilled", body_call)):
        replay_call, = [call for call in replay.requests
                        if any(question["name"] == name for question in call["questions"])]
        assert replay_call["input"] == original_call["input"]
        assert replay_call["questions"] == original_call["questions"]
    observation = {
        "contract_variant": contract_variant, "candidate": text, "expected_fulfilled": expected_fulfilled,
        "expected_grounded": expected_grounded,
        "input": payload, "questions": body_call["questions"],
        "probabilities": probabilities, "refused_names": sorted(refused_names),
        "calls": [{
            "boundary": boundary, "input": call["input"], "questions": call["questions"],
            "probabilities": judgment.probabilities, "refused_names": sorted(judgment.refused_names),
        } for boundary, call, judgment in (
            ("grounding", grounding_call, grounding_judgment),
            ("request_fulfillment", body_call, operation_judgment),
        )],
        "node_needs_retry": updates["retry"].needs_retry,
        "node_response_kind": updates.get("response", state["response"]).kind,
        "validation_events": updates["debug"].validation_events,
    }
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**observation, "record": str(record)}, ensure_ascii=True))

    assert refused_names == set(), str(record)
    assert (probabilities["request_fulfilled"] >= 0.8) is expected_fulfilled, str(record)
    if expected_grounded is not None:
        assert (probabilities["grounded_0"] >= 0.8) is expected_grounded, str(record)
    assert updates["retry"].needs_retry is not expected_fulfilled, str(record)
    assert state["runtime"].previous_response == original
    if expected_fulfilled:
        result = updates["response"].result
        assert updates["response"].kind == "answer", str(record)
        assert result.semantic_validation.passed, str(record)
        assert export_answer_text(result) == CORRECT_TEXT
        assert result.actions == []
    else:
        assert "semantic_request_fulfillment" in updates["retry"].retrieval_feedback
