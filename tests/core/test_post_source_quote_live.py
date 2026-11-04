"""Opt-in Decisions checks for source quotations versus unsupported source claims.

From the repository root in Git Bash:
    RUN_LIVE_POST_SOURCE_QUOTE=1 uv run --locked pytest -q -s tests/core/test_post_source_quote_live.py

The public post-validation node builds every isolated source-unit input and
predicate through the existing provider HTTP fake. Each unchanged source call
reaches the configured Decisions provider once: two calls for the exact quote
and location, two for unsupported execution, and one for the fabricated quote.
The returned probabilities are replayed through the same public node without
another provider call. These five paid calls are skipped by default. No planner, synthesis, retrieval,
saving or notification API runs. Source quotations deliberately keep basis=source.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.contracts import GraphState, PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import ContractEvidence, ExtractBody, RequestContract
from src.infra.chunking import chunk_python_text
from src.infra.llm import LLMRegistry, build_llm_registry
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.infra.settings import get_settings
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary


pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_POST_SOURCE_QUOTE") != "1",
    reason="Set RUN_LIVE_POST_SOURCE_QUOTE=1 to call the configured Decisions provider.",
)

SOURCE = "import numpy as np\n\nif use_service:\n    np = service\nelse:\n    result = np.array([1, 2])\n"
REQUEST = "첨부한 branch.py에서 실제 NumPy의 numpy.array를 호출하는 문장만 원문 그대로 인용하고 줄 번호를 알려줘. 다른 객체의 array 호출은 포함하지 마."
Case = Literal["exact-source-and-location", "unsupported-execution", "fabricated-source-quote"]


def _state(case: Case) -> GraphState:
    parsed = chunk_python_text(
        path="branch.py", text=SOURCE, source_content=SOURCE.encode("utf-8"),
        source_uri="upload:///post-source-quote/branch.py", chunk_size=2000, chunk_overlap=0,
    ).parsed
    evidence = build_evidence(snapshot=parsed.snapshot, element=parsed.elements[0])
    task = RetrievalTask(
        route="upload", requirement_id="numpy_array_calls", query=REQUEST, k=5,
        requirement={"symbols": ["numpy.array"], "match": "symbol"},
    )
    contract = RequestContract(
        body=ExtractBody(instruction=REQUEST, requirement_ids=[task.requirement_id], evidence_ids=["e1"]),
        evidence=[ContractEvidence(id="e1", turn_id="current", quote=REQUEST,
                                   scope="current_request", interpretation="instruction")],
    )
    if case == "unsupported-execution":
        claim = "이 np.array([1, 2]) 호출은 실제로 실행되어 결과를 반환했습니다."
        answer = AnswerDocument(blocks=[
            *text_document(claim, basis="source", refs=[evidence.id]).blocks,
            *text_document(claim, basis="inference", refs=[evidence.id]).blocks,
        ])
    else:
        answer = AnswerDocument.model_validate({"blocks": [{
            "type": "code", "language": "python", "content": {
                "text": "    result = np.array([1, 2])" if case == "exact-source-and-location"
                else "    result = np.array([9])",
                "basis": "source", "refs": [evidence.id],
            },
        }, *([{
            "type": "paragraph", "content": [{
                "text": SOURCE_LOCATION_PLACEHOLDER, "basis": "source", "refs": [evidence.id],
                "source_location": {"ref": evidence.id, "start": 59, "end": 89},
            }],
        }] if case == "exact-source-and-location" else [])]})
    response = finalize_answer(answer, [evidence], retrieval_required=True)
    if case == "exact-source-and-location":
        assert response.content.blocks[1].content[0].text == "branch.py:6"
    return build_graph_state_input(
        user_input=REQUEST, current_turn_id="current", request_contract=contract,
        messages=[HumanMessage(content=REQUEST, id="current")],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(
            evidence=evidence, requirement_id=task.requirement_id, rank=1,
            score=RetrievalScore(metric="source_symbol", direction="higher"),
        )]),
        retry=RetryState(max_retries=0, max_content_repairs=1),
        response=ResponseState(
            result=response, evidence_packet=[evidence],
            evidence_requirement_map={evidence.id: [task.requirement_id]}, synthesis_attempt=1,
            request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


@pytest.fixture(scope="module")
def live_registry() -> LLMRegistry:
    return build_llm_registry(get_settings().model_copy(update={"verbose": False}))


@pytest.mark.parametrize("case", [
    "exact-source-and-location", "unsupported-execution", "fabricated-source-quote",
])
def test_live_post_grounding_distinguishes_quotes_from_execution_and_fabrication(
    case: Case, live_registry: LLMRegistry, tmp_path: Path,
) -> None:
    state = _state(case)
    original = state["response"].result.model_copy(deep=True)
    capture = DecisionsBoundary()
    make_post_synthesis_validation_node(False, decisions_client=capture, pass_threshold=0.8)(state)
    source_calls = [call for call in capture.requests if set(json.loads(call["input"])) == {"units"}]
    assert len(source_calls) == (1 if case == "fabricated-source-quote" else 2)
    for call in source_calls:
        unit, = json.loads(call["input"])["units"]
        question, = call["questions"]
        assert question["name"] == unit["name"]
        assert question["name"].startswith("grounded_")
        assert all(source["excerpt"] == SOURCE and not source["is_partial"] for source in unit["cited_sources"])
        assert unit["basis"] != "excerpt"

    probabilities = {}
    refused_names = set()
    source_observations = []
    with capture_llm_diagnostics():
        request_call_budget("post_synthesis_validation").max_calls = len(source_calls)
        for call in source_calls:
            judgment = live_registry.decisions_client.evaluate(
                input=call["input"], questions=call["questions"], stage="post_synthesis_validation",
            )
            probabilities.update(judgment.probabilities)
            refused_names.update(judgment.refused_names)
            source_observations.append({
                "input": call["input"], "questions": call["questions"],
                "probabilities": judgment.probabilities, "refused_names": sorted(judgment.refused_names),
            })
    replay = DecisionsBoundary(probabilities=probabilities, refused_names=refused_names)
    updates = make_post_synthesis_validation_node(False, decisions_client=replay, pass_threshold=0.8)(state)
    replay_source_calls = [call for call in replay.requests if set(json.loads(call["input"])) == {"units"}]
    for actual, captured in zip(replay_source_calls, source_calls, strict=True):
        assert actual["input"] == captured["input"]
        assert actual["questions"] == captured["questions"]
    final_state = {**state, **updates}
    observation = {
        "case": case, "expected_supported": case == "exact-source-and-location",
        "source_calls": source_observations,
        "probabilities": probabilities, "refused_names": sorted(refused_names),
        "node_response_kind": final_state["response"].kind,
        "node_needs_retry": updates["retry"].needs_retry,
        "validation_events": updates["debug"].validation_events,
    }
    record = tmp_path / "decisions.json"
    record.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**observation, "record": str(record)}, ensure_ascii=False))

    assert refused_names == set(), str(record)
    expected_supported = case == "exact-source-and-location"
    assert all((probability >= 0.8) is expected_supported
               for probability in probabilities.values()), str(record)
    assert updates["retry"].needs_retry is not expected_supported, str(record)
    assert state["response"].result == original
    if expected_supported:
        assert updates["response"].kind == "answer", str(record)
        assert updates["response"].result.content == original.content
        assert updates["response"].result.content_hash == original.content_hash
        assert updates["response"].result.citations == original.citations
