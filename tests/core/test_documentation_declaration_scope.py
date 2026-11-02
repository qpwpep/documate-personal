"""Public validation inputs preserve the syntax of selected official declarations."""

import json

import pytest

from src.core.contracts import PlannerState, RetrievalState
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages
from src.runtime.nodes.validation import make_pre_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, _state, read_decision_input
from tests.synthesis_fixtures import synthesis_excerpt_limits


def _source_view(text, *, start=0, end=None, parser="official_html"):
    state = _state()
    task = state["planner"].output.tasks[0].model_copy(deep=True)
    task.requirement.symbols = ["example.Reader.read"]
    task.requirement.version = "3.9"
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task]))
    evidence = build_evidence(
        snapshot=build_snapshot(source_uri="https://docs.python.org/3.9/library/example.html",
            title="example — Python 3.9 documentation", media_type="text/html", source_type="official",
            content=text, parser=parser, parser_version="2"),
        element=DocumentElement(element_id="body", kind="paragraph", text=text), start=start, end=end,
    )
    state["retrieval"] = RetrievalState(hit_log=[
        state["retrieval"].hit_log[0].model_copy(update={"evidence": evidence}),
    ])
    boundary = DecisionsBoundary()

    make_pre_synthesis_validation_node(False, decisions_client=boundary,
        excerpt_limits=synthesis_excerpt_limits())(state)

    source = read_decision_input(boundary.requests[0]["input"])["evidence"][0]["source"]
    assert source["excerpt"] == evidence.excerpt
    assert source["selection"] == evidence.selection.model_dump(mode="json")
    messages, _, _ = build_synthesis_messages(
        state=state, request_contract=state["runtime"].request_contract, action_rules=[],
        evidence_packet=[evidence], attempt=1, max_turns=3,
        requirement_ids_by_evidence={evidence.id: [task.requirement_id]},
        reference_aliases={"e1": evidence.id}, retrieval_required=True,
    )
    packet_message = next(message.content for message in messages if message.content.startswith("[Evidence Packet]"))
    synthesis_source, = json.loads(packet_message.split("\n", 2)[2])
    assert synthesis_source.get("selection_scope") == source.get("selection_scope")
    return source


@pytest.mark.parametrize("signature, keywords, arbitrary", [
    ("Reader.read()", [], False),
    ("Reader.read(path, /, *, mode='r')", ["mode"], False),
    ("Reader.read(*items)", [], False),
    ("Reader.read(**options)", [], True),
    ("async Reader.read(path)", ["path"], False),
])
def test_selected_official_declaration_exposes_keyword_syntax(signature, keywords, arbitrary):
    text = signature + "¶\n\nRead one or more inputs.\n"

    source = _source_view(text)

    scope = source["selection_scope"]
    assert scope["kind"] == "documentation_declarations"
    assert scope["declarations"] == [{"name": "example.Reader.read", "start": 0,
        "end": len(signature) + 1, "fully_selected": True,
        "keyword_parameters": keywords, "accepts_arbitrary_keywords": arbitrary}]


def test_clipped_declaration_does_not_disclose_omitted_arguments():
    signature = "Reader.read(path, *, hidden_option=False)"
    text = signature + "¶\n\nRead one input.\n"
    selection_end = len("Reader.read(path,")

    source = _source_view(text, end=selection_end)

    declaration, = source["selection_scope"]["declarations"]
    assert declaration["fully_selected"] is False
    assert "keyword_parameters" not in declaration
    assert "accepts_arbitrary_keywords" not in declaration
    assert "hidden_option" not in str(source)


@pytest.mark.parametrize("signature", ["Reader.read(...)", "Reader.read([path])"])
def test_unsupported_documentation_notation_has_no_keyword_contract(signature):
    source = _source_view(signature + "¶\n\nRead one input.\n")

    declaration, = source["selection_scope"]["declarations"]
    assert "keyword_parameters" not in declaration
    assert "accepts_arbitrary_keywords" not in declaration


@pytest.mark.parametrize("text, parser", [
    ("# Examples\n\n```python\nReader.read()¶\n```\n", "official_html"),
    ("Reader.read()\n\nA call shown in ordinary prose.\n", "official_html"),
    ("Reader.read()¶\n\nRead one input.\n", "provider_excerpt"),
])
def test_example_calls_and_unmarked_or_provider_text_are_not_official_declarations(text, parser):
    assert "selection_scope" not in _source_view(text, parser=parser)


@pytest.mark.parametrize("heading", ["", "## Parameters\n\n"])
def test_parameter_excerpt_preserves_its_actual_api_owner_without_unselected_details(heading):
    selected = "Use compact output when compact is false.\n\n"
    text = ("Reader.read()¶\n\nRead one input.\n\n"
            "class Writer(secret_default='UNSELECTED_DEFAULT')¶\n\n" + heading + selected
            + "Writer.flush()¶\n\nFlush buffered output.\n")
    start = text.index(selected)

    source = _source_view(text, start=start, end=start + len(selected))

    assert source["selection_scope"]["api_owners"] == [{
        "name": "example.Writer",
        "section_start": text.index(heading) if heading else text.index("class Writer"),
        "section_end": text.index("Writer.flush()¶"),
    }]
    assert source["selection_scope"]["declarations"] == []
    assert "secret_default" not in str(source)
    assert "UNSELECTED_DEFAULT" not in str(source)
    assert "Read one input" not in str(source)
    assert "keyword_parameters" not in str(source)


def test_bare_index_before_the_actual_definition_does_not_gain_ownership():
    index = "Reader.read()\n\nAn index entry.\n\n"
    text = index + "Reader.read(path)¶\n\nRead one input.\n"

    source = _source_view(text, end=len(index))

    assert "selection_scope" not in source


def test_fenced_call_inside_a_definition_cannot_create_another_api_declaration():
    example = "```python\nWriter.write()¶\n```\n"
    text = "Reader.read(path)¶\n\nRead one input.\n\n" + example

    source = _source_view(text, start=text.index(example))

    assert source["selection_scope"]["declarations"] == []
    assert source["selection_scope"]["api_owners"] == [{
        "name": "example.Reader.read", "section_start": 0, "section_end": len(text),
    }]


@pytest.mark.parametrize("heading", [
    "Pattern language¶", "Comparison to the glob module¶",
    "## Storage formats", "<h2>Storage formats</h2>", "Storage formats\n---------------",
])
def test_general_section_heading_ends_the_previous_definition_owner(heading):
    selected = "These rules describe the document's shared formats.\n"
    text = ("# Reader\n\nReader.read(path)¶\n\nRead one input.\n\n"
            + heading + "\n\n" + selected)

    source = _source_view(text, start=text.index(selected))

    assert "selection_scope" not in source
