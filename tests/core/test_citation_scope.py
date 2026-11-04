"""Explicit citation kinds constrain actual references independently of model scores."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import ResponseState
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.request_contracts import AnswerContract, RequestContract, check_answer_contract
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state


def _sources():
    sources = []
    for name, kind, text in (
        ("settings.py", "code", "WORKER_COUNT = 6\n"),
        ("notes.md", "paragraph", "WORKER_COUNT is 6.\n"),
    ):
        snapshot = build_snapshot(
            source_uri=f"upload:///{name}", title=name, media_type="text/plain",
            source_type="upload", content=text, parser="test", parser_version="1",
        )
        sources.append(build_evidence(
            snapshot=snapshot, element=DocumentElement(element_id="content", kind=kind, text=text),
        ))
    return sources


def _contract():
    return RequestContract.model_validate({
        "answer": {"citation_scope": {"allowed_kinds": ["code"], "evidence_ids": ["code-only"]}},
        "evidence": [{"id": "code-only", "turn_id": "current-user", "quote": "첨부 코드만 인용해줘",
                      "scope": "answer.citation_scope", "interpretation": "instruction"}],
    })


@pytest.mark.parametrize("cited_indices, valid", [([0], True), ([1], False), ([0, 1], False)])
def test_citation_scope_checks_actual_references_not_every_available_source(cited_indices, valid):
    packet = _sources()
    document = text_document("WORKER_COUNT는 6입니다.", basis="source",
                             refs=[packet[index].id for index in cited_indices])

    check = check_answer_contract(_contract().answer, document, evidence=iter(packet))

    assert check.valid is valid
    assert check.forbidden_present == ([] if valid else ["citation_scope"])


@pytest.mark.parametrize("cited_index, status", [(0, "completed"), (1, "failed")])
def test_post_validation_enforces_citation_scope_despite_high_semantic_scores(cited_index, status):
    packet = _sources()
    contract = _contract()
    document = text_document("WORKER_COUNT는 6입니다.", basis="source", refs=[packet[cited_index].id])
    state = _state(retrieval=False, repairs=0)
    state["runtime"].request_contract = contract
    state["runtime"].user_input = "WORKER_COUNT를 알려주고 첨부 코드만 인용해줘"
    state["response"] = ResponseState(
        result=finalize_answer(document, packet), evidence_packet=packet,
        synthesis_attempt=1, request_id=contract.request_id, contract_revision=contract.revision,
    )
    provider = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=provider)(state)
    outcome = ResponseAssembler().assemble(response={**state, **updates})

    assert outcome.status == status
    if status == "failed":
        assert outcome.response is None
        assert outcome.problem.code == "answer_validation_failed"
    else:
        assert outcome.response is not None
        assert outcome.response.semantic_validation.passed


@pytest.mark.parametrize("allowed_kinds", [[], ["markdown"]])
def test_citation_scope_requires_at_least_one_document_element_kind(allowed_kinds):
    with pytest.raises(ValidationError) as error:
        AnswerContract.model_validate({"citation_scope": {"allowed_kinds": allowed_kinds, "evidence_ids": []}})
    assert error.value.errors()[0]["loc"][:2] == ("citation_scope", "allowed_kinds")


@pytest.mark.parametrize("evidence_ids, scope, interpretation", [
    ([], "answer.citation_scope", "instruction"),
    (["code-only"], "answer.format", "instruction"),
    (["code-only"], "answer.citation_scope", "quotation"),
])
def test_citation_scope_requires_user_authority(evidence_ids, scope, interpretation):
    with pytest.raises(ValidationError, match="answer.citation_scope requires user evidence"):
        RequestContract.model_validate({
            "answer": {"citation_scope": {"allowed_kinds": ["code"], "evidence_ids": evidence_ids}},
            "evidence": [{"id": "code-only", "turn_id": "current-user", "quote": "첨부 코드만 인용해줘",
                          "scope": scope, "interpretation": interpretation}],
        })
