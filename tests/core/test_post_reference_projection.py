"""Body views use aliases; requirement judgments retain exact source identities."""
from __future__ import annotations

from copy import deepcopy
import json

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, finalize_answer, iter_content_units
from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.contracts import GraphState, PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.evidence import EvidenceRef, RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import (
    BoundAnswerReference, ContractEvidence, ExtractBody, RequestContract, TransformAnswerBody,
)
from src.infra.chunking import chunk_python_text
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary


FIRST_TEXT = "value = 1\n"
REQUEST = "first.py의 value 대입문과 원본 줄 위치, notes.py의 REFERENCE_LITERAL 대입문을 원문 그대로 제시해줘."


def _source(filename: str, text: str) -> EvidenceRef:
    parsed = chunk_python_text(
        path=filename, text=text, source_content=text.encode("utf-8"),
        source_uri=f"upload:///reference-projection/{filename}", chunk_size=2000, chunk_overlap=0,
    ).parsed
    element = parsed.elements[0].model_copy(update={
        "metadata": {**parsed.elements[0].metadata, "file_id": filename},
    })
    return build_evidence(snapshot=parsed.snapshot, element=element)


def _state(*, bound_previous: bool, terminal_partial: bool) -> GraphState:
    first = _source("first.py", FIRST_TEXT)
    # An actual source ID inside source content is literal text, not a reference field.
    literal_line = f'REFERENCE_LITERAL = "{first.id}"\n'
    notes = _source("notes.py", literal_line + "# Retained source context.\n")
    second = build_evidence(snapshot=notes.snapshot, element=notes.element, start=0, end=len(literal_line))
    uncited = build_evidence(snapshot=notes.snapshot, element=notes.element,
                             start=len(literal_line), end=len(notes.element.text))
    document = AnswerDocument.model_validate({"blocks": [
        {"type": "code", "language": "python", "content": {
            "text": "value = 1", "basis": "source", "refs": [first.id],
        }},
        {"type": "paragraph", "content": [{
            "text": SOURCE_LOCATION_PLACEHOLDER, "basis": "source", "refs": [first.id],
            "source_location": {"ref": first.id, "start": 0, "end": len(FIRST_TEXT)},
        }]},
        {"type": "code", "language": "python", "content": {
            "text": f'REFERENCE_LITERAL = "{first.id}"', "basis": "source", "refs": [second.id],
        }},
    ]})
    result = finalize_answer(document, [first, second], retrieval_required=True)
    prior_only = _source("prior-only.py", "previous_value = 3\n") if bound_previous else None
    previous = finalize_answer(AnswerDocument.model_validate({"blocks": [
        result.content.blocks[2].model_dump(mode="json"),
        result.content.blocks[0].model_dump(mode="json"),
        result.content.blocks[1].model_dump(mode="json"),
        {"type": "code", "language": "python", "content": {
            "text": "previous_value = 3", "basis": "source", "refs": [prior_only.id],
        }},
    ]}), [first, second, prior_only], retrieval_required=True) if prior_only is not None else None
    requirement_ids = ["first_value", "reference_literal"]
    body = TransformAnswerBody(change_requirement="if_needed",
        source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash),
        instruction="원문과 위치를 유지하고 first.py 다음 notes.py 순서로 재배치해줘.",
        evidence_ids=["instruction"],
    ) if previous is not None else ExtractBody(
        instruction=REQUEST, requirement_ids=requirement_ids, evidence_ids=["instruction"],
    )
    contract = RequestContract(
        body=body,
        evidence=[ContractEvidence(id="instruction", turn_id="current", quote=REQUEST,
                                   scope="current_request", interpretation="instruction")],
    )
    tasks = [
        RetrievalTask(route="upload", requirement_id="first_value", query="first.py의 value 대입문과 줄 위치", k=2,
                      requirement={"file_ids": ["first.py"], "symbols": ["value"], "match": "symbol"}),
        RetrievalTask(route="upload", requirement_id="reference_literal", query="notes.py의 REFERENCE_LITERAL 대입문", k=2,
                      requirement={"file_ids": ["notes.py"], "symbols": ["REFERENCE_LITERAL"], "match": "symbol"}),
    ]
    return build_graph_state_input(
        user_input=REQUEST, current_turn_id="current", request_contract=contract, previous_response=previous,
        messages=[HumanMessage(id="current", content=REQUEST)],
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks)),
        retrieval=RetrievalState(hit_log=[
            SearchHit(evidence=evidence, requirement_id=requirement_id, rank=1,
                      score=RetrievalScore(metric="source_symbol", direction="higher"))
            for evidence, requirement_id in zip([first, second, uncited], [*requirement_ids, "reference_literal"])
        ]),
        retry=RetryState(max_retries=0, max_content_repairs=1,
                        terminal_failed_requirement_ids=["reference_literal"] if terminal_partial else [],
                        terminal_retrieval_reason="no_evidence" if terminal_partial else None),
        response=ResponseState(
            result=result, evidence_packet=[first, second, uncited],
            evidence_requirement_map={first.id: ["first_value"], second.id: ["reference_literal"],
                                      uncited.id: ["reference_literal"]},
            synthesis_attempt=1, request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


@pytest.mark.parametrize("bound_previous", [False, True], ids=["new-extract", "bound-answer"])
@pytest.mark.parametrize("terminal_partial", [False, True], ids=["complete", "partial"])
def test_post_body_aliases_and_requirement_source_ids_preserve_text_and_domain_ids(
    bound_previous: bool, terminal_partial: bool,
) -> None:
    state = _state(bound_previous=bound_previous, terminal_partial=terminal_partial)
    original = deepcopy(state)
    first, second, uncited = state["response"].evidence_packet
    boundary = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    source_calls = [call for call in boundary.requests if set(json.loads(call["input"])) == {"units"}]
    assert len(source_calls) == 2
    source_units = []
    for index, call in zip((0, 2), source_calls):
        unit, = json.loads(call["input"])["units"]
        assert unit["name"] == f"grounded_{index}"
        assert [question["name"] for question in call["questions"]] == [f"grounded_{index}"]
        source_units.append(unit)
        if index < 2:
            assert "REFERENCE_LITERAL" not in call["input"]
        else:
            assert FIRST_TEXT.strip() not in call["input"]
    body_call, = [call for call in boundary.requests
                  if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    requirement_call, = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    body_input = json.loads(body_call["input"])
    requirement_input = json.loads(requirement_call["input"])
    assert [unit["refs"] for unit in source_units] == [["e1"], ["e2"]]
    assert [unit["refs"] for unit in body_input["units"]] == [["e1"], ["e1"], ["e2"]]
    displayed = AnswerDocument.model_validate(body_input["rendered_response"]["body"])
    assert [unit.refs for _, unit in iter_content_units(displayed)] == [["e1"], ["e1"], ["e2"]]
    assert displayed.blocks[1].content[0].source_location.model_dump() == {
        "ref": "e1", "start": 0, "end": len(FIRST_TEXT),
    }
    assert displayed.blocks[1].content[0].text == "first.py:1"
    assert first.id in displayed.blocks[2].content.text
    for source in [source for unit in source_units for source in unit["cited_sources"]]:
        evidence = first if source["id"] == "e1" else second
        assert source["id"] in {"e1", "e2"}
        assert source["excerpt"] == evidence.excerpt
        assert source["snapshot_id"] == evidence.snapshot.snapshot_id
        assert source["uri"] == evidence.snapshot.source_uri
        assert source["selection"] == evidence.selection.model_dump(mode="json")
        assert source["source_lines"][0]["start"] == 0
        assert source["source_lines"][0]["end"] == len(evidence.excerpt)
    scope = {entry["task"]["requirement_id"]: entry for entry in body_input["answer_scope"]["requirements"]}
    assert scope["first_value"]["available_source_ids"] == scope["first_value"]["cited_source_ids"] == ["e1"]
    assert scope["reference_literal"]["available_source_ids"] == ["e2", "e3"]
    assert scope["reference_literal"]["cited_source_ids"] == ["e2"]
    uncited_source, = scope["reference_literal"]["uncited_sources"]
    assert uncited_source["id"] == "e3"
    assert uncited_source["excerpt"] == uncited.excerpt
    assert uncited_source["snapshot_id"] == uncited.snapshot.snapshot_id
    assert uncited_source["uri"] == uncited.snapshot.source_uri
    assert uncited_source["selection"] == uncited.selection.model_dump(mode="json")
    assert scope["reference_literal"]["coverage"] == ("partial" if terminal_partial else "required")
    assert set(requirement_input["sources"]) == ({first.id} if terminal_partial else {first.id, second.id})
    assert [entry["source_ids"] for entry in requirement_input["requirements"]] == (
        [[first.id]] if terminal_partial else [[first.id], [second.id]]
    )
    if not terminal_partial:
        assert first.id in requirement_input["sources"][second.id]["excerpt"]
    if bound_previous:
        bound = body_input["bound_source_response"]
        assert bound["content_hash"] == original["runtime"].previous_response.content_hash
        assert [source["id"] for source in bound["sources"]] == ["e2", "e1", "e4"]
        assert bound["sources"][2]["excerpt"] == "previous_value = 3\n"
        assert bound["sources"][2]["uri"] == "upload:///reference-projection/prior-only.py"
        bound_document = AnswerDocument.model_validate(bound["content"])
        assert [unit.refs for _, unit in iter_content_units(bound_document)] == [["e2"], ["e1"], ["e1"], ["e4"]]
        assert first.id in bound_document.blocks[0].content.text
    assert state == original
    assert updates["retry"].needs_retry is False
    assert updates["response"].result.content == original["response"].result.content
    assert updates["response"].result.citations == original["response"].result.citations
