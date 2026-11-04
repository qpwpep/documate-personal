"""Provider inputs keep one copy of source text without changing owned evidence."""
import hashlib
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.answer_schema import (
    ResponseIssue, SemanticCheck, SemanticValidation, export_answer_text, finalize_answer,
    semantic_evidence_fingerprint, text_document,
)
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence, source_line_records
from src.core.request_contracts import (
    BoundAnswerReference, BoundInputText, RequestContract, TransformAnswerBody, TransformInputBody,
)
from src.runtime.nodes.synthesis import make_synthesize_node
from tests.core.helpers import mock_chat_model
from tests.core.test_synthesis_validation import _hit, _state
from tests.synthesis_fixtures import prepare_synthesis_state


def _record(messages, heading):
    message = next(str(message.content) for message in messages
                   if str(message.content).startswith(heading))
    return json.loads(message.split("\n", 2)[2])


def _source_state(element, *, cell_ids=None, end=None):
    snapshot = build_snapshot(
        source_uri="upload:///sample", title="sample", media_type="text/plain", source_type="upload",
        content="owned source", parser="test", parser_version="1",
    )
    evidence = build_evidence(snapshot=snapshot, element=element, cell_ids=cell_ids, end=end)
    hit = SearchHit(evidence=evidence, rank=1,
                    score=RetrievalScore(metric="test", direction="higher", normalized=1))
    return _state([hit]), evidence


def test_code_source_is_sent_once_with_exact_line_offsets_and_citations():
    original = "# 한글\r\nSOURCE_SENTINEL = 'value'\r\n"
    state, evidence = _source_state(DocumentElement(
        element_id="code", kind="code", text=original, language="python",
        anchors=[SourceAnchor(kind="code", start=100, end=100 + len(original),
                              line_start=20, line_end=21, precision="exact")],
    ))
    saved = evidence.model_dump(mode="json")
    sent = []
    instructions = []

    def respond(messages):
        source, = _record(messages, "[Evidence Packet]")
        sent.append(source)
        instructions.extend(str(message.content) for message in messages if message.type == "system")
        return text_document(source["excerpt"], basis="excerpt", refs=[source["id"]])

    state.update(make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state)))

    source, = sent
    assert json.dumps(source).count("SOURCE_SENTINEL") == 1
    assert source["excerpt"] == original
    line = next(record for record in source["source_lines"] if record["line_number"] == 21)
    base = source["selection"]["start"]
    assert source["excerpt"][line["start"] - base:line["end"] - base] == "SOURCE_SENTINEL = 'value'\r\n"
    assert state["response"].result.citations[0].evidence == evidence
    assert state["response"].result.checks[0].support_status == "exact_match"
    assert evidence.model_dump(mode="json") == saved
    assert "A conventional alias or matching attribute spelling does not establish that binding" in "\n".join(instructions)


def test_notebook_model_locations_use_human_cell_numbers_without_mutating_anchors():
    original = "result = np.array([1, 2, 3])\n"
    state, evidence = _source_state(DocumentElement(
        element_id="result-cell", kind="code", text=original, language="python",
        anchors=[SourceAnchor(kind="notebook", start=0, end=len(original), line_start=1, line_end=1,
                              cell_id="result-cell", cell_index=2, precision="exact")],
    ))
    saved = evidence.model_dump(mode="json")
    sent = []

    def respond(messages):
        source, = _record(messages, "[Evidence Packet]")
        sent.append(source)
        return text_document(source["excerpt"], basis="excerpt", refs=[source["id"]])

    state.update(make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state)))

    source, = sent
    source_location, = source["source_locations"]
    assert source_location["cell_number"] == 3
    assert "cell_index" not in source_location
    assert source_location["cell_id"] == "result-cell"
    assert source_location["line_start"] == 1
    assert state["response"].result.citations[0].evidence.element.anchors[0].cell_index == 2
    assert evidence.model_dump(mode="json") == saved


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
@pytest.mark.parametrize("cut_content", [False, True])
def test_model_line_completeness_ignores_only_an_omitted_line_ending(newline, cut_content):
    statement = "result = calculate_total(prices)"
    original = statement + newline
    state, evidence = _source_state(DocumentElement(
        element_id="code", kind="code", text=original, language="python",
        anchors=[SourceAnchor(kind="code", start=100, end=100 + len(original),
                              line_start=20, line_end=20, precision="exact")],
    ), end=len(statement) - int(cut_content))
    canonical_line, = source_line_records(evidence)
    assert canonical_line["partial_end"] is True
    sent = []

    def respond(messages):
        source, = _record(messages, "[Evidence Packet]")
        sent.append(source)
        return text_document(source["excerpt"], basis="excerpt", refs=[source["id"]])

    state.update(make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state)))

    source, = sent
    line, = source["source_lines"]
    assert line["partial_end"] is cut_content
    assert (line["line_number"], line["start"], line["end"]) == (20, 0, len(statement) - int(cut_content))
    assert source["excerpt"] == (statement[:-1] if cut_content else statement)
    assert state["response"].result.citations[0].evidence == evidence
    assert source_line_records(evidence) == [canonical_line]


def test_selected_table_values_are_sent_once_with_merged_cell_structure():
    repeated = "TABLE_SENTINEL | 한글\n\\escaped"
    element = DocumentElement(element_id="table", kind="table", table=TableData(cells=[
        TableCell(cell_id="heading", row=0, col=0, col_span=2, text="Counts", is_header=True),
        TableCell(cell_id="first", row=1, col=0, row_span=2, text=repeated),
        TableCell(cell_id="second", row=1, col=1, text=repeated),
        TableCell(cell_id="omitted", row=2, col=1, text="OMITTED_VALUE"),
    ]))
    state, evidence = _source_state(element, cell_ids=["heading", "first", "second"])
    saved = evidence.model_dump(mode="json")
    sent = []

    def respond(messages):
        source, = _record(messages, "[Evidence Packet]")
        sent.append(source)
        return text_document(source["excerpt"], basis="excerpt", refs=[source["id"]])

    state.update(make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state)))

    source, = sent
    assert json.dumps(source).count("TABLE_SENTINEL") == 2
    assert "OMITTED_VALUE" not in json.dumps(source)
    cells = {cell["cell_id"]: cell for cell in source["table_cells"]}
    assert cells["heading"]["col_span"] == 2
    assert cells["first"]["row_span"] == 2
    assert cells["heading"]["is_header"] is True
    for cell_id in ("first", "second"):
        cell = cells[cell_id]
        start, end = cell["excerpt_range"]
        assert source["excerpt"][start:end] == repeated
        assert cell.get("col_span", 1) == 1
        assert cell.get("is_header", False) is False
    assert state["response"].result.citations[0].evidence == evidence
    assert state["response"].result.checks[0].support_status == "exact_match"
    assert evidence.model_dump(mode="json") == saved


def test_transform_input_only_replaces_its_bound_occurrence_in_history():
    selected = "SELECTED_INPUT"
    request = f"Keep the first {selected}. Translate only the second:\n{selected}"
    start = request.rindex(selected)
    state = _state([], query=request)
    state["messages"] = [HumanMessage(id="source-turn", content=request)]
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": RequestContract(
        body=TransformInputBody(instruction="Translate the selected second occurrence.", source=BoundInputText(
            turn_id="source-turn", text=selected, start=start, end=len(request),
            content_hash=hashlib.sha256(selected.encode()).hexdigest(),
        )),
    )})
    sent = []

    def respond(messages):
        sent.extend(messages)
        return text_document("선택된 입력")

    state.update(make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state)))

    human, = [message for message in sent if isinstance(message, HumanMessage)]
    assert human.content.count(selected) == 1
    assert human.content.startswith(f"Keep the first {selected}. Translate only the second:\n")
    assert _record(sent, "[Bound User Input]")["text"] == selected
    assert state["messages"][0].content == request
    assert export_answer_text(state["response"].result) == "선택된 입력"


def test_transform_answer_reuses_bound_body_and_preserves_only_semantic_limits():
    previous = finalize_answer(text_document("BOUND_ANSWER_BODY"), [], issues=[
        ResponseIssue(code="answer_incomplete", message="The upload remains unavailable."),
    ])
    previous = previous.model_copy(update={"semantic_validation": SemanticValidation(
        content_hash=previous.content_hash, evidence_fingerprint=semantic_evidence_fingerprint([]), threshold=0.8,
        checks=[SemanticCheck(name="grounded", probability=0.99),
                SemanticCheck(name="request_fulfilled", probability=0.4),
                SemanticCheck(name="detail", unit_id="b0.content.0", status="refused", probability=None)],
    )})
    saved = previous.model_dump(mode="json")
    history_text = export_answer_text(previous) + "\n\nAdditional delivery limitation remains."
    state = _state([], query="Summarize the previous answer.")
    state["messages"] = [HumanMessage(content="Earlier request."), AIMessage(content=history_text),
                         HumanMessage(content="Summarize the previous answer.")]
    state["runtime"] = state["runtime"].model_copy(update={
        "previous_response": previous,
        "request_contract": RequestContract(body=TransformAnswerBody(
            source=BoundAnswerReference(ref="previous", response_hash=previous.content_hash),
            instruction="Summarize while preserving its limitations.",
        )),
    })
    sent = []

    def respond(messages):
        sent.extend(messages)
        return text_document("The upload remains unavailable.")

    state.update(make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state)))

    assert sum(str(message.content).count("BOUND_ANSWER_BODY") for message in sent) == 1
    assert any("Additional delivery limitation remains." in str(message.content) for message in sent)
    bound = _record(sent, "[Bound Source Answer]")
    assert bound["issues"] == saved["issues"]
    assert "semantic_validation" not in bound
    assert bound["semantic_limitations"] == [
        {"name": "request_fulfilled", "unit_id": None, "status": "below_threshold"},
        {"name": "detail", "unit_id": "b0.content.0", "status": "refused"},
    ]
    assert previous.model_dump(mode="json") == saved
    assert state["messages"][1].content == history_text


def test_synthesis_can_cite_api_relationship_and_behavior_on_the_same_claim():
    relationship = _hit("parse_text(text) uses the same decoding rules as parse_stream(stream).")
    behavior = _hit("parse_stream(stream) raises DecodeError when the input is not a valid encoded document.")
    state = _state([relationship, behavior], query="Explain parse_text errors.")
    sent = []

    def respond(messages):
        sent.extend(messages)
        packet = _record(messages, "[Evidence Packet]")
        return text_document("parse_text raises DecodeError for an invalid encoded document.",
                             basis="source", refs=[source["id"] for source in packet])

    updates = make_synthesize_node(mock_chat_model(respond))(prepare_synthesis_state(state))

    assert {source["excerpt"] for source in _record(sent, "[Evidence Packet]")} == {
        relationship.evidence.excerpt, behavior.evidence.excerpt,
    }
    assert {citation.evidence.id for citation in updates["response"].result.citations} == {
        relationship.evidence.id, behavior.evidence.id,
    }
