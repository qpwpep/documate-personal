"""Every displayed citation must contribute to the claim it accompanies."""
import json

import pytest

from src.core.answer_schema import finalize_answer, text_document
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_cited_ranges import PassingDecisions, _state


def _answer_with_two_sources(additional_text="The sort option orders the keys lexicographically."):
    state = _state()
    primary = state["response"].evidence_packet[0]
    additional = _state(source_text=additional_text)["response"].evidence_packet[0]
    packet = [primary, additional]
    document = text_document("maxsize에 None을 지정하면 저장 항목 수를 제한하지 않습니다.",
                             basis="source", refs=[item.id for item in packet])
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(document, packet, retrieval_required=True),
        "evidence_packet": packet,
        "evidence_requirement_map": {item.id: ["cache_limit"] for item in packet},
    })
    return state


@pytest.mark.parametrize("refused", [False, True])
def test_unrelated_extra_citation_requires_repair_even_when_the_claim_is_supported(refused):
    state = _answer_with_two_sources()
    boundary = PassingDecisions({} if refused else {"citations_0": 0.1},
                                refused_names={"citations_0"} if refused else ())

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert "semantic_citation_relevance:b0.content.0" in updates["retry"].retrieval_feedback
    body_input = json.loads(boundary.requests[0]["input"])
    assert set(body_input) == {"units"}
    assert len(body_input["units"]) == 1
    assert [question["name"] for question in boundary.requests[0]["questions"]] == ["grounded_0", "citations_0"]
    assert {source["id"] for source in body_input["units"][0]["cited_sources"]} == {"e1", "e2"}


def test_semantically_approved_complementary_citations_keep_the_exact_source_ranges():
    state = _answer_with_two_sources("The cache limit is configured by the maxsize argument.")
    boundary = PassingDecisions()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is False
    assert updates["response"].result.citations == state["response"].result.citations
    check = next(check for check in updates["response"].result.semantic_validation.checks
                 if check.name == "citation_relevance")
    assert check.probability == 0.99
