import json

import pytest

from src.core.documents import DocumentElement, TableCell, TableData, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalRequirement, RetrievalTask
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


@pytest.mark.parametrize("boilerplate_blocks", [5, 15])
def test_parameter_explanation_survives_document_prefix_length(boilerplate_blocks):
    """A requested parameter's complete paragraph survives the same 960-character prompt budget."""
    prefix = (
        "# numpy.reshape\n\nnumpy.reshape(a, shape, order='C')\n\n"
        + ("Documentation navigation and release links. " * 8 + "\n\n") * boilerplate_blocks
    )
    explanation = (
        "**order**: {'C', 'F', 'A'}, optional\n"
        "C visits the last axis first. F visits the first axis first. "
        "A uses F for a Fortran-contiguous input and C otherwise."
    )
    text = prefix + explanation + "\n\n**copy**: optional\nAn unrelated parameter."
    snapshot = build_snapshot(
        source_uri="https://numpy.org/doc/stable/reshape.html", title="numpy.reshape",
        media_type="text/markdown", source_type="official", content=text,
        parser="test", parser_version="1",
    )
    original = build_evidence(snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text))

    packet = prepare_evidence_packet(
        [original], max_items=6, snippet_char_limit=960, evidence_char_budget=6000,
        query="NumPy numpy.reshape order parameter official documentation",
    )

    assert len(packet) == 1
    selected = packet[0]
    assert explanation in selected.excerpt
    assert len(selected.excerpt) <= 960
    assert selected.snapshot == original.snapshot
    assert selected.element == original.element
    assert selected.excerpt == text[selected.selection.start:selected.selection.end]


def test_query_selection_never_reads_outside_the_retrieved_source_range():
    """A focused packet selects within the authorized evidence range, leaving omitted source text unseen."""
    text = "Secret order policy that was not selected.\n\nPublic description.\n\nAnother public paragraph."
    snapshot = build_snapshot(
        source_uri="https://example.com/docs", title="Settings", media_type="text/plain",
        source_type="official", content=text, parser="test", parser_version="1",
    )
    start = text.index("Public description")
    original = build_evidence(
        snapshot=snapshot, element=DocumentElement(element_id="body", kind="paragraph", text=text),
        start=start,
    )
    packet = prepare_evidence_packet(
        [original], max_items=1, snippet_char_limit=25, evidence_char_budget=25, query="order policy",
    )

    assert len(packet) == 1
    assert packet[0].selection.start >= start
    assert "Secret" not in packet[0].excerpt


def test_partial_source_range_is_explicit_in_the_model_packet():
    """A model can distinguish a partial selection from the complete captured source."""
    original = evidence("def run():\n    retries = 3\n    return retries\n")
    packet = prepare_evidence_packet([original], max_items=1, snippet_char_limit=26, evidence_char_budget=26)
    messages, _, _ = build_synthesis_messages(
        state=build_graph_state_input(user_input="Explain run"), action_rules=[],
        evidence_packet=packet, attempt=1, max_turns=6,
    )
    raw = str(messages[-1].content)
    serialized = json.loads(raw[raw.index("[", len("[Evidence Packet]")):])

    assert serialized[0]["selection"] == packet[0].selection.model_dump(mode="json")
    assert serialized[0]["is_partial"] is True


def test_specific_code_aspect_preserves_the_complete_statement_near_a_long_function_end():
    """An explanation of a late call receives its original complete lines within the existing budget."""
    statement = "        transaction.commit_changes(\n            payload,\n            retries=3,\n        )\n"
    text = (
        "def persist(payload):\n    label = '설명'\n"
        + "    payload = normalize(payload)\n" * 60
        + "    if payload:\n" + statement + "    return payload\n"
    )
    snapshot = build_snapshot(
        source_uri="uploads/example.py", title="persist", media_type="text/x-python",
        source_type="upload", content=text, parser="test", parser_version="1",
    )
    original = build_evidence(
        snapshot=snapshot, element=DocumentElement(element_id="persist", kind="code", language="python", text=text),
    )
    task = RetrievalTask(
        route="upload", query="Explain transaction.commit_changes in persist", k=1,
        requirement=RetrievalRequirement(symbols=["persist"], aspects=["commit_changes"]),
    )

    packet = prepare_evidence_packet(
        [original], max_items=1, snippet_char_limit=200, evidence_char_budget=200,
        query=task.query, requirements_by_evidence={original.id: [task]},
    )

    assert len(packet) == 1
    selected = packet[0]
    assert statement in selected.excerpt
    assert len(selected.excerpt) <= 200
    assert selected.selection.start > 0
    assert text[selected.selection.start - 1] == "\n"
    assert selected.excerpt.endswith("\n")
    assert selected.excerpt == text[selected.selection.start:selected.selection.end]
    assert selected.snapshot == original.snapshot
    assert selected.element == original.element


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_ast_definition_end_before_a_line_ending_keeps_the_last_requested_call(newline):
    """A real AST definition range ending before its newline still exposes its complete final statement."""
    from src.infra.chunking import chunk_python_text
    from src.infra.tools.local_rag.requirements import resolve_source_requirement

    source = (
        "def persist(payload):\n    label = '설명'\n"
        + "    payload = normalize(payload)\n" * 60
        + "    transaction.commit_changes(payload)\n\ndef other():\n    return 0\n"
    ).replace("\n", newline)
    document = chunk_python_text(
        path="uploads/selection/example.py", text=source, chunk_size=800, chunk_overlap=120,
    )
    requirement = RetrievalRequirement(symbols=["persist"], aspects=["commit_changes"], match="definition")
    result = resolve_source_requirement(requirement=requirement, source_document=document.parsed, candidate_rows=[])
    task = RetrievalTask(route="upload", query="Explain commit_changes in persist", k=1, requirement=requirement)
    original = result.hits[0].evidence
    assert result.answerability == "covered"
    assert source[original.selection.end:].startswith(newline)

    packet = prepare_evidence_packet(
        [original], max_items=1, snippet_char_limit=200, evidence_char_budget=200,
        query=task.query, requirements_by_evidence={original.id: [task]},
    )

    assert len(packet) == 1
    selected = packet[0]
    assert "    transaction.commit_changes(payload)" in selected.excerpt
    assert selected.selection.end == original.selection.end
    assert selected.selection.start >= original.selection.start
    assert len(selected.excerpt) <= 200
    assert selected.excerpt == source[selected.selection.start:selected.selection.end]
    assert selected.snapshot == original.snapshot
    assert selected.element == original.element
