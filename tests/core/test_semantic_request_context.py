"""Semantic judgments use request meaning without internal bookkeeping fields."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from src.core.request_contracts import (
    AnswerContract, ComposeBody, ContentRequirement, ContractEvidence, RequestContract,
)
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_semantic_request_preserves_meaning_without_internal_identity(stage):
    request = "앞선 요청을 정정합니다. 사용자에게 짧은 인사말을 작성해 주세요."
    payloads = []
    for revision in (1, 4):
        state = _state(retrieval=False, answer="안녕하세요. 반갑습니다.")
        contract = RequestContract(
            request_id=f"request-{revision}", revision=revision,
            relation="correction", target_request_id=f"pending-{revision}",
            body=ComposeBody(instruction="짧은 인사말을 작성해 주세요.", evidence_ids=("instruction",)),
            body_request=ComposeBody(instruction=f"planner-only-{revision}"),
            answer=AnswerContract(content=(ContentRequirement(
                kind="custom", mode="required", description="사용자에게 인사합니다.",
                evidence_ids=("instruction",),
            ),)),
            evidence=(ContractEvidence(
                id="instruction", turn_id="current", quote=request,
                scope="current_request", interpretation="correction",
            ),),
        )
        original_contract = contract.model_dump(mode="json")
        state["runtime"] = state["runtime"].model_copy(update={
            "user_input": request, "request_contract": contract,
        })
        state["messages"] = [HumanMessage(content=request, id="current")]
        state["response"] = state["response"].model_copy(update={
            "request_id": contract.request_id, "contract_revision": contract.revision,
        })
        decisions = DecisionsBoundary()
        node = (
            make_pre_synthesis_validation_node(
                False, decisions_client=decisions, excerpt_limits=synthesis_excerpt_limits(),
            ) if stage == "pre" else make_post_synthesis_validation_node(False, decisions_client=decisions)
        )

        node(state)

        payload, = [json.loads(item["input"]) for item in decisions.requests]
        confirmed = payload["confirmed_request"]
        assert not {"request_id", "revision", "target_request_id", "body_request"} & confirmed.keys()
        assert confirmed["relation"] == "correction"
        assert confirmed["body"] == {
            "kind": "compose", "instruction": "짧은 인사말을 작성해 주세요.", "evidence_ids": ["instruction"],
        }
        assert confirmed["answer"]["content"] == [{
            "kind": "custom", "mode": "required", "description": "사용자에게 인사합니다.",
            "evidence_ids": ["instruction"], "replaces": None,
        }]
        assert confirmed["evidence"] == [{
            "id": "instruction", "turn_id": "current", "quote": request,
            "scope": "current_request", "interpretation": "correction",
        }]
        assert state["runtime"].request_contract.model_dump(mode="json") == original_contract
        payloads.append(payload)

    assert payloads[0] == payloads[1]
