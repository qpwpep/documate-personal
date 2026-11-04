"""Semantic judgments use request meaning without internal bookkeeping fields."""
from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage

from src.core.request_contracts import (
    AnswerContract, ComposeBody, ContentRequirement, ContractEvidence, ExtractBody, RequestContract,
)
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


@pytest.mark.parametrize("stage", ["pre", "post"])
@pytest.mark.parametrize("bound_requirement", ["path_exists", "other_scope", None])
def test_source_tasks_receive_only_their_bound_extract_operation(stage, bound_requirement):
    state = _state()
    task = state["planner"].output.tasks[0].model_copy(update={
        "query": "Extract the Path.exists declaration and locate it in the selected source.",
    })
    other = task.model_copy(update={"requirement_id": "other_scope", "query": "Explain Path.exists behavior."})
    state["planner"] = state["planner"].model_copy(update={
        "output": state["planner"].output.model_copy(update={"tasks": [task, other]}),
    })
    hit = state["retrieval"].hit_log[0]
    state["retrieval"] = state["retrieval"].model_copy(update={
        "hit_log": [hit, hit.model_copy(update={"requirement_id": other.requirement_id})],
    })
    state["response"] = state["response"].model_copy(update={
        "evidence_requirement_map": {hit.evidence.id: [task.requirement_id, other.requirement_id]},
    })
    extraction = ExtractBody(instruction="PRIVATE_BODY_INSTRUCTION", requirement_ids=(
        bound_requirement or task.requirement_id,
    ))
    contract = RequestContract(
        request_id="decisions-validation",
        body=extraction if bound_requirement else ComposeBody(instruction="Explain the selected material."),
        body_request=extraction,
    )
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    boundary = DecisionsBoundary()
    node = (make_pre_synthesis_validation_node(False, decisions_client=boundary,
                                               excerpt_limits=synthesis_excerpt_limits()) if stage == "pre"
            else make_post_synthesis_validation_node(False, decisions_client=boundary))

    node(state)

    calls = [call for call in boundary.requests if "tasks" in json.loads(call["input"])]
    assert calls
    for call in calls:
        assert "PRIVATE_BODY_INSTRUCTION" not in call["input"]
        projected = json.loads(call["input"])["tasks"]
        for expected in (task, other):
            assert projected[expected.requirement_id] == {
                **expected.model_dump(mode="json", exclude={"k"}),
                **({"operation": "extract"} if expected.requirement_id == bound_requirement else {}),
            }
    if stage == "post":
        body_call, = [call for call in boundary.requests
                      if any(question["name"] == "request_fulfilled" for question in call["questions"])]
        body = json.loads(body_call["input"])
        assert body["confirmed_request"]["body"] == contract.body.model_dump(mode="json")
        assert [entry["task"] for entry in body["answer_scope"]["requirements"]] == [
            expected.model_dump(mode="json", exclude={"k"}) for expected in (task, other)
        ]
    assert state["runtime"].request_contract == contract


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

        updates = node(state)

        assert state["runtime"].request_contract.model_dump(mode="json") == original_contract
        if stage == "pre":
            assert decisions.requests == []
            assert updates["retry"].needs_retry is False
            prepared = updates["retrieval"].prepared_evidence
            assert prepared.request_id == contract.request_id
            assert prepared.contract_revision == contract.revision
            assert prepared.normal.failed_requirement_ids == []
            continue

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
        payloads.append(payload)

    if stage == "post":
        assert payloads[0] == payloads[1]
