"""Opt-in checks of model judgments, separate from deterministic boundary tests.

Git Bash: RUN_LIVE_CITATION_SUPPORT=1 uv run --locked pytest -q tests/core/test_citation_support_live.py
These tests call the configured Decisions provider with short documentation fixtures.
"""
import os

import pytest

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import PlannerState, ResponseState, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.llm import build_llm_registry
from src.infra.settings import get_settings
from src.runtime.nodes.validation import make_post_synthesis_validation_node


pytestmark = pytest.mark.skipif(os.environ.get("RUN_LIVE_CITATION_SUPPORT") != "1",
                                reason="Set RUN_LIVE_CITATION_SUPPORT=1 for paid model checks.")


@pytest.fixture(scope="module")
def live_decisions():
    return build_llm_registry(get_settings().model_copy(update={"verbose": False})).decisions_client


@pytest.mark.parametrize("with_error_source, with_unrelated_source, accepted", [
    pytest.param(False, False, False, id="exception-name-needs-evidence"),
    pytest.param(True, True, False, id="unrelated-extra-citation"),
    pytest.param(True, False, True, id="complementary-supported-citations"),
])
def test_live_inference_and_each_citation_need_selected_support(
    with_error_source, with_unrelated_source, accepted, live_decisions,
):
    query = "What does pandas.merge(validate='one_to_one') check, and what happens with duplicate keys?"
    task = RetrievalTask(requirement_id="merge", route="docs", query=query, k=2,
        requirement={"library": "pandas", "symbols": ["pandas.merge"], "aspects": ["validate", "one_to_one"],
                     "evidence_needs": ["explanation"]})
    passages = [("pandas.merge", "validate: str, optional\nIf specified, checks if merge is of specified type.\n"
                 "one_to_one or 1:1: check if merge keys are unique in both left and right datasets.")]
    if with_error_source:
        passages.append(("pandas.errors.MergeError", "exception pandas.errors.MergeError\n"
            "Exception raised when merging data.\nExamples\n"
            ">>> pandas.merge(left, right, on='a', validate='one_to_one')\n"
            "Traceback (most recent call last):\n"
            "pandas.errors.MergeError: Merge keys are not unique in left dataset; not a one-to-one merge"))
    if with_unrelated_source:
        passages.append(("pandas.merge", "sort: bool, default False\n"
            "Sort the join keys lexicographically in the result DataFrame."))
    packet = []
    for index, (symbol, text) in enumerate(passages):
        snapshot = build_snapshot(source_uri=f"https://pandas.pydata.org/docs/reference/api/{symbol}.html",
            title=f"{symbol} — pandas 3.1.0 documentation", media_type="text/html", source_type="official",
            content=text, parser="official_html", parser_version="2")
        packet.append(build_evidence(snapshot=snapshot,
            element=DocumentElement(element_id=f"passage-{index}", kind="paragraph", text=text)))
    contract = RequestContract(request_id="citation-support")
    document = text_document("validate='one_to_one' checks key uniqueness in both datasets; "
        "duplicate keys in the left dataset raise pandas.errors.MergeError.",
        basis="inference", refs=[item.id for item in packet])
    state = build_graph_state_input(user_input=query, request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[SearchHit(evidence=item, requirement_id="merge", rank=index + 1,
            score={"metric": "fixture", "direction": "higher"}) for index, item in enumerate(packet)]),
        retry=RetryState(max_retries=1, max_content_repairs=1),
        response=ResponseState(result=finalize_answer(document, packet, retrieval_required=True),
            evidence_packet=packet, evidence_requirement_map={item.id: ["merge"] for item in packet},
            synthesis_attempt=1, request_id=contract.request_id, contract_revision=contract.revision))

    updates = make_post_synthesis_validation_node(False, decisions_client=live_decisions)(state)

    assert (not updates["retry"].needs_retry) is accepted, updates["debug"].validation_events
