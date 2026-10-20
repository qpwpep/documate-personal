"""Semantic validation receives the meaning and limits of selected evidence."""
from __future__ import annotations

from functools import partial
import hashlib
import json

import pytest

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.chunking import chunk_python_text
from src.infra.tools.local_rag.requirements import resolve_source_requirement
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state
from tests.synthesis_fixtures import synthesis_excerpt_limits


make_pre_synthesis_validation_node = partial(make_pre_synthesis_validation_node, excerpt_limits=synthesis_excerpt_limits())

def _snapshot():
    return build_snapshot(
        source_uri="https://manual.test/manual.pdf", title="manual.pdf", media_type="application/pdf",
        source_type="official", content="Captured timeout documentation", parser="test", parser_version="1",
        capture_scope="provider_excerpt", quality_issues=["Some diagrams were not extracted."],
    )


def _validation_input(evidence_packet, location):
    state = _state()
    if evidence_packet and evidence_packet[0].route == "upload":
        state["planner"] = state["planner"].model_copy(update={"output": PlannerOutput(
            use_retrieval=True, tasks=[RetrievalTask(route="upload", query="selected source", k=2,
                                                  requirement_id="path_exists")],
        )})
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


@pytest.mark.parametrize("clipped", [False, True])
def test_validation_knows_whether_a_selected_python_definition_is_complete(clipped):
    """A partial file selection can contain a whole function, but a valid prefix is not the whole function."""
    definition = 'def discounted_total(prices, discount=0.20):\n    total = sum(prices)\n    return total * (1 - discount)'
    text = definition + '\n\nprivate_example = "UNSELECTED_SOURCE_CONTENT"\n'
    end = definition.index("\n    return") if clipped else len(definition)
    item = build_evidence(
        snapshot=build_snapshot(
            source_uri="upload:///test/price_calc.py", title="price_calc.py",
            media_type="text/x-python", source_type="upload", content=text,
            parser="python", parser_version="1",
        ),
        element=DocumentElement(element_id="code", kind="code", language="python", text=text),
        end=end,
    )

    sources, payload = _validation_input([item], "pre")

    assert sources[0]["is_partial"] is True
    assert sources[0]["selection_scope"] == {
        "kind": "python_definitions",
        "definitions": [{"name": "discounted_total", "kind": "function", "fully_selected": not clipped,
                         "line_start": 1, "line_end": 3}],
    }
    assert "UNSELECTED_SOURCE_CONTENT" not in json.dumps(payload)


@pytest.mark.parametrize("line_endings, clipped", [
    (("\n",), False),
    (("\r\n",), False),
    (("\r",), False),
    (("\r", "\n", "\r\n"), True),
], ids=["lf", "crlf", "cr", "mixed-clipped"])
def test_uploaded_python_definitions_use_python_line_endings_without_changing_source(line_endings, clipped):
    lines = ["@one", "def one():", "    return 1", "", "@(", "    decorate", ")",
             "def 계산():", "    return '한글'", "", "private = 'UNSELECTED_SOURCE_CONTENT'"]
    text = "".join(line + line_endings[index % len(line_endings)] for index, line in enumerate(lines))
    upload = chunk_python_text(path="source.py", text=text, source_content=text.encode("utf-8"),
                               chunk_size=2000, chunk_overlap=0)
    original = upload.hydrate(upload.chunks[0])
    assert original.excerpt == text
    assert original.snapshot.content_hash == "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert original.element.anchors[0].line_end == len(lines)
    start = text.index("def 계산") if clipped else text.index("@(")
    end = text.index("    return '한글'") + len("    return '한글'")
    item = build_evidence(snapshot=original.snapshot, element=original.element, start=start, end=end)

    sources, payload = _validation_input([item], "pre")

    assert sources[0]["excerpt"] == text[start:end]
    assert sources[0]["selection"] == {"start": start, "end": end, "cell_ids": []}
    assert sources[0]["selection_scope"] == {
        "kind": "python_definitions",
        "definitions": [{"name": "계산", "kind": "function", "fully_selected": not clipped,
                         "line_start": 5, "line_end": 9}],
    }
    assert "UNSELECTED_SOURCE_CONTENT" not in json.dumps(payload)


def test_python_source_lookup_and_validation_do_not_count_non_python_line_separators():
    separator = "\f\v\x85\u2028\u2029"
    line_ending = "\n"
    prefix = f"# Source comment contains {separator} inside one physical line.{line_ending}"
    definition = line_ending.join(["@(", "    decorate", ")", "def 계산():", f"    return '한글{separator}문자'"])
    text = prefix + definition + line_ending
    upload = chunk_python_text(path="source.py", text=text, source_content=text.encode("utf-8"),
                               chunk_size=2000, chunk_overlap=0)

    resolved = resolve_source_requirement(requirement=RetrievalRequirement(symbols=["계산"], match="definition"),
                                          source_documents=(upload.parsed,))

    assert resolved.answerability == "covered"
    item, = [hit.evidence for hit in resolved.hits]
    assert item.excerpt == definition
    assert item.element.text == text
    assert item.selection.start == len(prefix)
    assert item.selection.end == len(prefix) + len(definition)
    assert item.element.anchors[0].line_end == 6
    sources, _ = _validation_input([item], "pre")
    assert sources[0]["selection_scope"] == {
        "kind": "python_definitions",
        "definitions": [{"name": "계산", "kind": "function", "fully_selected": True,
                         "line_start": 2, "line_end": 6}],
    }


@pytest.mark.parametrize("text,language", [("def broken(\n", "python"), ("def plausible():\n    pass", "text")])
def test_validation_never_invents_definition_completeness_without_a_valid_python_source(text, language):
    item = build_evidence(
        snapshot=_snapshot(), element=DocumentElement(element_id="code", kind="code", language=language, text=text),
    )

    sources, _ = _validation_input([item], "pre")

    assert "selection_scope" not in sources[0]
