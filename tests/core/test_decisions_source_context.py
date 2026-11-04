"""Semantic validation receives the meaning and limits of selected evidence."""
from __future__ import annotations

from functools import partial
import hashlib
import json

import pytest
from nbformat.v4 import new_code_cell, new_notebook

from src.core.answer_schema import AnswerDocument, finalize_answer, iter_content_units, text_document
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.chunking import chunk_notebook, chunk_python_text
from src.infra.tools.local_rag.requirements import resolve_source_requirement
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core import test_post_source_quote_live
from tests.core.test_decisions_validation import DecisionsBoundary, _state, read_decision_input
from tests.synthesis_fixtures import synthesis_excerpt_limits


make_pre_synthesis_validation_node = partial(make_pre_synthesis_validation_node, excerpt_limits=synthesis_excerpt_limits())

def _snapshot():
    return build_snapshot(
        source_uri="https://manual.test/manual.pdf", title="manual.pdf", media_type="application/pdf",
        source_type="official", content="Captured timeout documentation", parser="test", parser_version="1",
        capture_scope="provider_excerpt", quality_issues=["Some diagrams were not extracted."],
    )


def _validation_input(evidence_packet, location, *, raw=False):
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
            "kind": "transform_answer", "change_requirement": "if_needed", "instruction": "이전 답변을 요약해 주세요.",
            "source": {"ref": "previous", "response_hash": result.content_hash},
        })
        state["runtime"] = state["runtime"].model_copy(update={
            "request_contract": contract, "previous_response": result,
        })
    if location == "synthesis":
        messages, _, _ = build_synthesis_messages(
            state=state, request_contract=state["runtime"].request_contract,
            action_rules=[], evidence_packet=evidence_packet, attempt=1, max_turns=4,
            requirement_ids_by_evidence={item.id: ["path_exists"] for item in evidence_packet},
            reference_aliases={}, retrieval_required=True,
        )
        packet_message = next(str(message.content) for message in messages
                              if str(message.content).startswith("[Evidence Packet]"))
        sources = json.loads(packet_message.split("\n", 2)[2])
        return sources, sources
    boundary = DecisionsBoundary()
    factory = make_pre_synthesis_validation_node if location.endswith("pre") else make_post_synthesis_validation_node
    factory(False, decisions_client=boundary)(state)
    # Source judgment and accepted-source coverage must receive the same
    # selected ranges and source context. Bound answers have no new sources.
    if location == "source_pre":
        request, = [call for call in boundary.requests if "evidence" in json.loads(call["input"])]
    elif location in {"pre", "post_requirements"}:
        request, = [call for call in boundary.requests if "requirements" in json.loads(call["input"])]
    elif location.startswith("bound_"):
        request, = [call for call in boundary.requests if "bound_source_response" in json.loads(call["input"])]
    else:
        requests = [call for call in boundary.requests if set(json.loads(call["input"])) == {"units"}]
        assert len(requests) == len(evidence_packet)
        payloads = [read_decision_input(call["input"]) for call in requests]
        sources = []
        for index, (request, payload) in enumerate(zip(requests, payloads, strict=True)):
            unit, = payload["units"]
            assert unit["name"] == f"grounded_{index}"
            assert [question["name"] for question in request["questions"]] == [f"grounded_{index}"]
            sources.extend(unit["cited_sources"])
        return sources, payloads
    wire = request["input"]
    if raw:
        payload = json.loads(wire)
        return list(payload["sources"].values()), payload
    payload = read_decision_input(wire)
    if location.startswith("bound_"):
        sources = payload["bound_source_response"]["sources"]
    elif location == "source_pre":
        sources = [item["source"] for item in payload["evidence"]]
    elif location in {"pre", "post_requirements"}:
        sources = payload["requirements"][0]["sources"]
    return sources, payload


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post", "synthesis"])
@pytest.mark.parametrize("has_tail", [False, True])
def test_source_projections_distinguish_python_file_end_from_an_omitted_tail(location, has_tail):
    prefix = "import numpy as np\n\n"
    selected = 'match service:\n    case {"client": np}:\n        result = np.array([3, 4])\n'
    tail = "# UNSELECTED_TAIL\nresult = another()\n" if has_tail else ""
    original = prefix + selected + tail
    upload = chunk_python_text(path="capture.py", text=original, chunk_size=1000, chunk_overlap=0)
    item = build_evidence(snapshot=upload.parsed.snapshot, element=upload.parsed.elements[0],
                          start=len(prefix), end=len(prefix + selected))
    saved = item.model_dump(mode="json")

    sources, payload = _validation_input([item], location)

    source, = sources
    assert source["excerpt"] == selected
    assert source["selection"] == {"start": 20, "end": 94, "cell_ids": []}
    assert source["is_partial"] is True
    assert source["element_extent"] == {
        "kind": "python_file", "text_length": len(original),
        "omitted_prefix": True, "omitted_suffix": has_tail,
    }
    assert "import numpy" not in json.dumps(payload)
    assert "UNSELECTED_TAIL" not in json.dumps(payload)
    assert item.model_dump(mode="json") == saved


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post", "synthesis"])
def test_source_projections_do_not_confuse_a_complete_notebook_cell_with_the_file_end(location):
    selected = "result = calculate_total(prices)\n"
    upload = chunk_notebook(
        path="capture.ipynb", notebook=new_notebook(cells=[
            new_code_cell(selected, id="selected"),
            new_code_cell("UNSELECTED_LATER_CELL = True\n", id="later"),
        ]), chunk_size=1000, chunk_overlap=0,
    )
    item = build_evidence(snapshot=upload.parsed.snapshot, element=upload.parsed.elements[0])

    sources, payload = _validation_input([item], location)

    source, = sources
    assert source["excerpt"] == selected
    assert source["is_partial"] is False
    assert source["element_extent"] == {
        "kind": "notebook_cell", "text_length": len(selected),
        "omitted_prefix": False, "omitted_suffix": False,
    }
    assert "UNSELECTED_LATER_CELL" not in json.dumps(payload)


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post", "synthesis"])
def test_source_projections_do_not_promote_a_generic_code_element_to_a_complete_python_file(location):
    selected = "result = calculate_total(prices)\n"
    item = build_evidence(snapshot=build_snapshot(
        source_uri="upload:///snippet.py", title="snippet.py", media_type="text/x-python",
        source_type="upload", content=selected, parser="generic", parser_version="1",
    ), element=DocumentElement(element_id="code", kind="code", text=selected, language="python"))

    sources, _ = _validation_input([item], location)

    source, = sources
    assert source["is_partial"] is False
    assert source["element_extent"] == {
        "kind": "document_element", "text_length": len(selected),
        "omitted_prefix": False, "omitted_suffix": False,
    }


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post", "synthesis"])
@pytest.mark.parametrize("omit_cell", [False, True])
def test_source_projections_keep_table_cell_selection_separate_from_text_extent(location, omit_cell):
    item = build_evidence(snapshot=_snapshot(), element=DocumentElement(
        element_id="values", kind="table", table=TableData(cells=[
            TableCell(cell_id="selected", row=0, col=0, text="selected value"),
            TableCell(cell_id="other", row=1, col=0, text="OTHER_TABLE_VALUE"),
        ]),
    ), cell_ids=["selected"] if omit_cell else ["selected", "other"])

    sources, payload = _validation_input([item], location)

    source, = sources
    assert source["is_partial"] is omit_cell
    assert "element_extent" not in source
    if omit_cell:
        assert "OTHER_TABLE_VALUE" not in json.dumps(payload)


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post", "synthesis"])
@pytest.mark.parametrize("file_id", ["captured-upload-id", None])
def test_source_projections_preserve_original_file_identity_without_inferring_it(location, file_id):
    source = "RETRY_LIMIT = 3\n"
    document = chunk_python_text(path="settings.py", text=source, source_uri="upload:///session/uri-is-not-file-id",
                                 chunk_size=1000, chunk_overlap=0)
    element = document.parsed.elements[0]
    if file_id is not None:
        element.metadata["file_id"] = file_id
    item = build_evidence(snapshot=document.parsed.snapshot, element=element)
    saved = item.model_dump(mode="json")

    sources, _ = _validation_input([item], location)

    selected, = sources
    assert selected["file_id"] == file_id
    assert item.model_dump(mode="json") == saved


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post"])
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
        assert source["element_extent"] == {
            "kind": "document_element", "text_length": len(text),
            "omitted_prefix": True, "omitted_suffix": True,
        }
        assert source["capture_scope"] == "provider_excerpt"
        assert source["quality_issues"] == ["Some diagrams were not extracted."]
        assert source["source_locations"] == [anchor.model_dump(mode="json", exclude={"cell_index"})]
    assert "Unselected introduction" not in json.dumps(payload)
    assert "Unselected conclusion" not in json.dumps(payload)


@pytest.mark.parametrize("location", ["source_pre", "pre", "post", "post_requirements", "bound_post"])
def test_validation_notebook_locations_use_human_cell_numbers_without_mutating_anchors(location):
    text = "result = np.array([1, 2, 3])\n"
    anchor = SourceAnchor(kind="notebook", start=0, end=len(text), line_start=1, line_end=1,
                          cell_id="result-cell", cell_index=2, precision="exact")
    item = build_evidence(snapshot=build_snapshot(
        source_uri="upload:///branch.ipynb", title="branch.ipynb", media_type="application/x-ipynb+json",
        source_type="upload", content=text, parser="notebook", parser_version="1",
    ), element=DocumentElement(element_id="result-cell", kind="code", text=text,
                               language="python", anchors=[anchor]))
    saved = item.model_dump(mode="json")

    sources, _ = _validation_input([item], location)

    source, = sources
    source_location, = source["source_locations"]
    assert source_location["cell_number"] == 3
    assert "cell_index" not in source_location
    assert source_location["cell_id"] == "result-cell"
    assert source_location["line_start"] == 1
    assert item.element.anchors[0].cell_index == 2
    assert item.model_dump(mode="json") == saved


def test_bound_answer_preparation_preserves_selected_sources_without_predicting_the_body():
    text = "Unselected introduction. 30 seconds Unselected conclusion."
    start = text.index("30 seconds")
    evidence = build_evidence(
        snapshot=_snapshot(),
        element=DocumentElement(element_id="timeout", kind="paragraph", text=text,
                                anchors=[SourceAnchor(kind="page", page_no=2, precision="element")]),
        start=start, end=start + len("30 seconds"),
    )
    previous = finalize_answer(text_document("기본 대기 시간은 30초입니다.", basis="source", refs=[evidence.id]),
                               [evidence], retrieval_required=True)
    saved = previous.model_dump(mode="json")
    contract = RequestContract(request_id="bound-answer", body={
        "kind": "transform_answer", "change_requirement": "if_needed", "instruction": "이전 답변을 요약해 주세요.",
        "source": {"ref": "previous", "response_hash": previous.content_hash},
    })
    state = _state(retrieval=False)
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": contract, "previous_response": previous,
    })
    decisions = DecisionsBoundary()

    state.update(make_pre_synthesis_validation_node(False, decisions_client=decisions)(state))

    assert decisions.requests == []
    assert state["retry"].needs_retry is False
    prepared = state["retrieval"].prepared_evidence
    assert prepared.source_response_hash == previous.content_hash
    assert prepared.normal.evidence == [evidence]
    assert prepared.compact.evidence == [evidence]
    assert state["runtime"].previous_response.model_dump(mode="json") == saved


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
    assert source["source_locations"] == [selected_anchor.model_dump(mode="json", exclude={"cell_index"})]
    assert "UNSELECTED_TIMEOUT_VALUE" not in json.dumps(payload)


def test_validation_table_wire_preserves_cell_text_through_offsets_without_duplicates():
    """Repeated, merged and multiline cells keep their original positions and text."""
    anchor = SourceAnchor(kind="table", page_no=2, precision="element")
    cells = [
        TableCell(cell_id="heading", row=0, col=0, col_span=2, text='한글 | 머리\n"글"', is_header=True),
        TableCell(cell_id="first", row=1, col=0, text="반복\n값"),
        TableCell(cell_id="second", row=1, col=1, text="반복\n값"),
        TableCell(cell_id="omitted", row=2, col=0, text="UNSELECTED_TABLE_SECRET"),
    ]
    item = build_evidence(
        snapshot=_snapshot(), element=DocumentElement(
            element_id="repeated-values", kind="table", anchors=[anchor], table=TableData(cells=cells),
        ), cell_ids=["heading", "first", "second"],
    )

    sources, payload = _validation_input([item], "pre", raw=True)

    source, = sources
    assert source["excerpt"] == '한글 | 머리\n"글"\n반복\n값 | 반복\n값'
    assert [cell["cell_id"] for cell in source["table_cells"]] == ["heading", "first", "second"]
    assert all("text" not in cell for cell in source["table_cells"])
    assert [source["excerpt"][cell["excerpt_range"][0]:cell["excerpt_range"][1]]
            for cell in source["table_cells"]] == ['한글 | 머리\n"글"', "반복\n값", "반복\n값"]
    assert source["table_cells"][0]["col_span"] == 2
    assert source["table_cells"][0]["is_header"] is True
    assert source["table_cells"][1].get("col_span", 1) == 1
    assert source["table_cells"][1].get("row_span", 1) == 1
    assert source["table_cells"][1].get("is_header", False) is False
    assert source["table_cells"][1]["excerpt_range"][1] < source["table_cells"][2]["excerpt_range"][0]
    assert "UNSELECTED_TABLE_SECRET" not in json.dumps(payload)


def test_validation_line_wire_uses_original_offsets_without_repeating_selected_text():
    """Partial CRLF lines retain physical positions in the single selected excerpt."""
    text = "private = 1\r\n\r\ndef 계산():\r\n    value = '한글'\r\n    return value\r\n"
    upload = chunk_python_text(path="offsets.py", text=text, source_content=text.encode("utf-8"),
                               chunk_size=2000, chunk_overlap=0)
    original = upload.hydrate(upload.chunks[0])
    start = text.index("계산")
    end = text.index("    return") + len("    return")
    item = build_evidence(snapshot=original.snapshot, element=original.element, start=start, end=end)

    sources, payload = _validation_input([item], "source_pre", raw=True)

    source, = sources
    assert source["excerpt"] == "계산():\r\n    value = '한글'\r\n    return"
    assert source["selection"]["start"] == start
    lines = source["source_lines"]
    assert [line["line_number"] for line in lines] == [3, 4, 5]
    assert all("text" not in line for line in lines)
    assert [source["excerpt"][line["start"] - start:line["end"] - start] for line in lines] == [
        "계산():\r\n", "    value = '한글'\r\n", "    return",
    ]
    assert lines[0]["partial_start"] is True
    assert lines[-1]["partial_end"] is True
    assert "private = 1" not in json.dumps(payload)


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


@pytest.mark.parametrize("location_first", [True, False])
def test_extract_fulfillment_distinguishes_position_labels_from_quoted_source(location_first):
    state = test_post_source_quote_live._state("exact-source-and-location")
    response = state["response"]
    blocks = list(response.result.content.blocks)
    if location_first:
        blocks.reverse()
    document = AnswerDocument(blocks=blocks)
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, response.evidence_packet, retrieval_required=True),
    })
    boundary = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    call = next(call for call in boundary.requests
                if any(question["name"] == "request_fulfilled" for question in call["questions"]))
    payload = read_decision_input(call["input"])
    displayed = AnswerDocument.model_validate(payload["rendered_response"]["body"])
    assert [block.model_dump(exclude={"content"}) for block in displayed.blocks] == [
        block.model_dump(exclude={"content"}) for block in document.blocks
    ]
    for (actual_path, actual), (original_path, original) in zip(
        iter_content_units(displayed), iter_content_units(document), strict=True,
    ):
        assert actual_path == original_path
        assert actual.model_dump(exclude={"refs", "source_location"}) == original.model_dump(
            exclude={"refs", "source_location"},
        )
        assert actual.refs == ["e1"]
        if original.source_location is not None:
            assert actual.source_location.model_dump() == {"ref": "e1", "start": 59, "end": 89}
        else:
            assert actual.source_location is None
    assert len(payload["units"]) == 2
    assert all(unit["cited_sources"][0]["excerpt"] == test_post_source_quote_live.SOURCE
               for unit in payload["units"])
    instructions = next(question["instructions"] for question in call["questions"]
                        if question["name"] == "request_fulfilled")
    assert "A unit with structured source_location is a displayed source-position label" in instructions
    assert "the body must include both the requested quote and its position" in instructions
    assert "follow any explicitly requested order" in instructions
    assert not updates["retry"].needs_retry
    assert updates["response"].result.content == document
