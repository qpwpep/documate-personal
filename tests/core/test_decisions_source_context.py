"""Semantic validation receives the meaning and limits of selected evidence."""
from __future__ import annotations

import json

import pytest

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.request_contracts import RequestContract
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state


def _snapshot():
    return build_snapshot(
        source_uri="https://manual.test/manual.pdf", title="manual.pdf", media_type="application/pdf",
        source_type="official", content="Captured timeout documentation", parser="test", parser_version="1",
        capture_scope="provider_excerpt", quality_issues=["Some diagrams were not extracted."],
    )


def _validation_input(evidence_packet, location):
    state = _state()
    response = state["response"]
    document = AnswerDocument(blocks=[
        block for item in evidence_packet
        for block in text_document("기본 대기 시간은 30초입니다.", basis="source", refs=[item.id]).blocks
    ])
    result = finalize_answer(document, evidence_packet, retrieval_required=True)
    state["retrieval"] = state["retrieval"].model_copy(update={"hit_log": [
        SearchHit(
            evidence=item, requirement_id="path_exists", rank=index + 1,
            score=RetrievalScore(metric="test", raw=0.99, normalized=0.99, direction="higher"),
        ) for index, item in enumerate(evidence_packet)
    ]})
    state["response"] = response.model_copy(update={
        "result": result, "evidence_packet": evidence_packet,
        "evidence_requirement_map": {item.id: ["path_exists"] for item in evidence_packet},
    })
    if location.startswith("bound_"):
        state = _state(retrieval=False)
        contract = RequestContract(request_id="decisions-validation", body={
            "kind": "transform_answer", "instruction": "이전 답변을 요약해 주세요.",
            "source": {"ref": "previous", "response_hash": result.content_hash},
        })
        state["runtime"] = state["runtime"].model_copy(update={
            "request_contract": contract, "previous_response": result,
        })
    boundary = DecisionsBoundary()
    factory = make_pre_synthesis_validation_node if location.endswith("pre") else make_post_synthesis_validation_node
    factory(False, decisions_client=boundary)(state)
    # Source judgment and accepted-source coverage must receive the same
    # selected ranges and source context. Bound answers have no new sources.
    payload = json.loads(boundary.requests[0 if location == "source_pre" else -1]["input"])
    if location.startswith("bound_"):
        sources = payload["bound_source_response"]["sources"]
    elif location == "source_pre":
        sources = [item["source"] for item in payload["evidence"]]
    elif location == "pre":
        sources = payload["requirements"][0]["sources"]
    else:
        sources = [source for unit in payload["units"] for source in unit["cited_sources"]]
    return sources, payload


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "bound_pre", "bound_post"])
def test_validation_distinguishes_identical_excerpts_by_heading_and_preserves_selection_limits(location):
    """Identical timeout text stays attributable to its Windows or Linux section."""
    text = "Unselected introduction. 30 seconds Unselected conclusion."
    start = text.index("30 seconds")
    anchor = SourceAnchor(kind="page", page_no=2, precision="element")
    packet = [build_evidence(
        snapshot=_snapshot(),
        element=DocumentElement(
            element_id=platform, kind="paragraph", text=text,
            heading_path=[platform, "Default timeout"], anchors=[anchor],
        ),
        start=start, end=start + len("30 seconds"),
    ) for platform in ["Windows", "Linux"]]

    sources, payload = _validation_input(packet, location)

    assert [source["heading_path"] for source in sources] == [
        ["Windows", "Default timeout"], ["Linux", "Default timeout"],
    ]
    for source in sources:
        assert source["kind"] == "paragraph"
        assert source["excerpt"] == "30 seconds"
        assert source["selection"] == {"start": start, "end": start + len("30 seconds"), "cell_ids": []}
        assert source["is_partial"] is True
        assert source["capture_scope"] == "provider_excerpt"
        assert source["quality_issues"] == ["Some diagrams were not extracted."]
        assert source["source_locations"] == [anchor.model_dump(mode="json")]
    assert "Unselected introduction" not in json.dumps(payload)
    assert "Unselected conclusion" not in json.dumps(payload)


def test_validation_preserves_selected_table_cells_and_never_exposes_omitted_cells():
    """Merged headers and row/column positions survive without adding unselected data."""
    selected_anchor = SourceAnchor(kind="table", page_no=2, precision="element")
    omitted_anchor = SourceAnchor(kind="table", page_no=3, precision="element")
    selected_cells = [
        {"cell_id": "section", "row": 0, "col": 0, "row_span": 1, "col_span": 2, "text": "Windows", "is_header": True},
        {"cell_id": "timeout", "row": 1, "col": 0, "row_span": 2, "col_span": 1, "text": "Default timeout", "is_header": True},
        {"cell_id": "value", "row": 1, "col": 1, "row_span": 1, "col_span": 1, "text": "30 seconds", "is_header": False},
    ]
    item = build_evidence(
        snapshot=_snapshot(), element=DocumentElement(
            element_id="timeouts", kind="table", heading_path=["Configuration", "Timeouts"],
            anchors=[selected_anchor, omitted_anchor], table=TableData(cells=[
                *(TableCell(**cell, anchors=[selected_anchor]) for cell in selected_cells),
                TableCell(cell_id="omitted", row=2, col=1, text="UNSELECTED_TIMEOUT_VALUE", anchors=[omitted_anchor]),
            ]),
        ), cell_ids=["section", "timeout", "value"],
    )

    sources, payload = _validation_input([item], "pre")

    assert len(sources) == 1
    source = sources[0]
    assert source["heading_path"] == ["Configuration", "Timeouts"]
    assert source["kind"] == "table"
    assert source["table_cells"] == selected_cells
    assert source["selection"]["cell_ids"] == ["section", "timeout", "value"]
    assert source["is_partial"] is True
    assert source["source_locations"] == [selected_anchor.model_dump(mode="json")]
    assert "UNSELECTED_TIMEOUT_VALUE" not in json.dumps(payload)
