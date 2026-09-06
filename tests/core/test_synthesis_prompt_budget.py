import json

from src.core.documents import DocumentElement, TableCell, TableData, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.nodes.synthesis.budgets import resolve_synthesis_budget_profile
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages, prepare_evidence_packet
from src.core.contracts.boundary.graph import build_graph_state_input


def evidence(text, *, source="upload"):
    snapshot = build_snapshot(
        source_uri="uploads/example.py", title="example.py", media_type="text/x-python",
        source_type=source, content=text, parser="test", parser_version="1",
    )
    return build_evidence(snapshot=snapshot, element=DocumentElement(element_id="code", kind="code", text=text))


def test_prompt_budget_changes_the_actual_reference_range_and_preserves_code():
    """A shortened prompt reference resolves to precisely its visible original source range."""
    original = evidence("def run():\n    retries = 3\n    return retries\n")
    packet = prepare_evidence_packet([original], max_items=6, snippet_char_limit=26, evidence_char_budget=26)
    assert len(packet) == 1
    selected = packet[0]
    assert selected.id != original.id
    assert selected.snapshot == original.snapshot
    assert selected.excerpt == original.element.text[:26]
    assert selected.selection.start == 0
    assert selected.selection.end == 26
    assert "\n    " in selected.excerpt
    messages, _, _ = build_synthesis_messages(
        state=build_graph_state_input(user_input="Explain run"), action_rules=[],
        evidence_packet=packet, attempt=1, max_turns=6,
    )
    rendered = str(messages[-1].content)
    prompt_packet = json.loads(rendered[rendered.index("[", len("[Evidence Packet]")):])
    assert [(item["id"], item["excerpt"]) for item in prompt_packet] == [(selected.id, selected.excerpt)]
    assert original.id not in rendered


def test_zero_remaining_budget_never_expands_to_the_full_source():
    """An exhausted evidence budget excludes source text instead of accidentally disabling truncation."""
    assert prepare_evidence_packet([evidence("private source" )], max_items=6, snippet_char_limit=100, evidence_char_budget=0) == []


def test_source_packet_can_contain_multiple_parts_of_a_document():
    """The answer may use more than two original ranges when the evidence budget allows it."""
    entries = [evidence(f"setting_{number} = {number}\n") for number in range(5)]
    packet = prepare_evidence_packet(entries, max_items=6, snippet_char_limit=100, evidence_char_budget=1000)
    assert packet == entries


def test_saving_a_researched_answer_keeps_the_retrieval_budget():
    """A save request does not discard supporting material from the answer being saved."""
    planner = PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="settings", k=4)])
    normal = resolve_synthesis_budget_profile(user_input="Explain settings", planner_output=planner, synthesis_max_tokens=1800)
    save = resolve_synthesis_budget_profile(user_input="Explain settings and save to txt", planner_output=planner, synthesis_max_tokens=1800)
    assert save == normal
    assert normal.max_tokens == 1800
    assert normal.max_evidence_items >= 4


def test_selected_table_cells_reach_generation_with_their_structure():
    """A table citation carries selected cells and spans without exposing unselected source values."""
    snapshot = evidence("table source").snapshot
    element = DocumentElement(element_id="table", kind="table", table=TableData(cells=[
        TableCell(cell_id="label", row=0, col=0, col_span=2, text="Retry settings", is_header=True),
        TableCell(cell_id="value", row=1, col=0, text="3"),
        TableCell(cell_id="unselected", row=1, col=1, text="not supplied"),
    ]))
    selected = build_evidence(snapshot=snapshot, element=element, cell_ids=["label", "value"])
    messages, _, _ = build_synthesis_messages(
        state=build_graph_state_input(user_input="Explain retry settings"), action_rules=[],
        evidence_packet=[selected], attempt=1, max_turns=6,
    )
    raw = str(messages[-1].content)
    packet = json.loads(raw[raw.index("[", len("[Evidence Packet]")):])
    assert packet[0]["table_cells"] == [
        {"cell_id": "label", "row": 0, "col": 0, "row_span": 1, "col_span": 2, "text": "Retry settings", "is_header": True},
        {"cell_id": "value", "row": 1, "col": 0, "row_span": 1, "col_span": 1, "text": "3", "is_header": False},
    ]
    assert "not supplied" not in raw
