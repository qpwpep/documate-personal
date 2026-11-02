from hashlib import sha256
from pathlib import Path

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import AnswerDocument, CodeBlock, ContentUnit, finalize_answer, text_document
from src.core.answer_export import exact_copy_ready
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import UserTurnSnapshot, WireRequestContract
from src.infra.tools.save_text import build_save_text_tool
from src.runtime.nodes.actions import make_action_postprocess_node
from src.runtime.nodes.planner import make_planner_node
from src.runtime.nodes.synthesis import make_synthesize_node
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.helpers import make_decisions_client, mock_chat_model, structured_model
from tests.synthesis_fixtures import prepare_synthesis_state


def _document_model(document=None):
    def respond(_messages):
        assert document is not None, "Exact copy must not generate a new body"
        return document

    return mock_chat_model(respond)


def _plan_copy(text, *, formats, content=(), previous=None, scoped_evidence=False,
               instruction="아래 원문은 내용 그대로 두고 요청한 형식으로 표시하고 저장해줘."):
    query = instruction + "\n" + text
    proposal = WireRequestContract.model_validate({
        "body": ({"kind": "copy_input", "source": {"turn_id": "current", "quote": text}}
                 if previous is None else {"kind": "copy_answer", "source": {"ref": "previous"}}),
        "slack_recipient": {"state": "omitted"},
        "actions": {"save_text": {"intent": "requested", "evidence_ids": ["request"]}},
        "answer": {
            "format": [{**item, "evidence_ids": ["request"]} for item in formats],
            "content": [{**item, "evidence_ids": ["request"]} for item in content],
        },
        "evidence": [{"id": "request", "turn_id": "current", "quote": query,
                      "scope": "current_request", "interpretation": "instruction"}],
    })
    if scoped_evidence:
        payload = proposal.model_dump(mode="python")
        payload["actions"]["save_text"]["evidence_ids"] = ("save",)
        for item in payload["answer"]["format"]:
            item["evidence_ids"] = ("format",)
        for item in payload["answer"]["content"]:
            item["evidence_ids"] = ("content",)
        payload["evidence"] = [
            {"id": name, "turn_id": "current", "quote": quote, "scope": scope, "interpretation": "instruction"}
            for name, quote, scope in [
                ("save", "저장해줘.", "actions.save_text"),
                ("format", "요청한 형식으로 표시", "answer.format"),
                ("content", "내용 그대로 두고", "answer.content"),
            ]
        ]
        proposal = WireRequestContract.model_validate(payload)
    state = build_graph_state_input(
        user_input=query, current_turn_id="current",
        user_turns=(UserTurnSnapshot(turn_id="current", text=query),),
        messages=[HumanMessage(content=query, id="current")], previous_response=previous,
    )
    planner = structured_model(
        lambda _messages: InitialPlannerOutput(use_retrieval=False, tasks=[], request_contract=proposal),
        InitialPlannerOutput,
    )
    state.update(make_planner_node(planner, False)(state))
    return state


def test_explicit_block_layout_changes_copy_operation_but_preserves_bound_input():
    text = "  request timeout\r\nagent execution failed\r\nunexpected error  "

    state = _plan_copy(text, formats=[{"kind": "code_block", "mode": "required"}], content=[
        {"kind": "example", "mode": "forbidden"},
        {"kind": "explanation", "mode": "forbidden"},
    ])

    contract = state["runtime"].request_contract
    assert contract.body.kind == "transform_input"
    assert contract.body.source.text == text
    assert contract.body.source.content_hash == sha256(text.encode("utf-8")).hexdigest()
    assert contract.body_request.kind == "transform_input"
    assert contract.body_request.source.quote == text
    assert [(item.kind, item.mode) for item in contract.answer.content] == [
        ("example", "forbidden"), ("explanation", "forbidden"),
    ]


def test_explicit_block_layout_changes_answer_copy_without_rebinding_its_source():
    text = "request timeout\nagent execution failed\nunexpected error"
    previous = finalize_answer(text_document(text), [])

    state = _plan_copy(text, formats=[{"kind": "code_block", "mode": "required"}], previous=previous)

    contract = state["runtime"].request_contract
    assert contract.body.kind == "transform_answer"
    assert contract.body.source.ref == "previous"
    assert contract.body.source.response_hash == previous.content_hash
    assert contract.body_request.kind == "transform_answer"
    assert contract.body_request.source.ref == "previous"
    assert state["runtime"].previous_response.content == previous.content


@pytest.mark.parametrize("previous_answer", [False, True])
@pytest.mark.parametrize("rendered, probability, expected_kind", [
    ("`alpha`와 beta", 0.01, "failure"),
    ("`alpha`와 `beta`", 0.99, "answer"),
])
def test_layout_scope_is_checked_when_the_source_is_only_partly_formatted(
    previous_answer, rendered, probability, expected_kind,
):
    text = "`alpha`와 beta"
    previous = finalize_answer(text_document(text), []) if previous_answer else None
    state = _plan_copy(
        text, formats=[{"kind": "inline_code", "mode": "required"}], previous=previous,
        instruction="원문 내용은 그대로 두고 alpha와 beta를 모두 인라인 코드로 표시하고 저장해줘.",
    )
    contract = state["runtime"].request_contract
    assert contract.body.kind == ("transform_answer" if previous_answer else "transform_input")
    assert not exact_copy_ready(contract, previous)
    state["retry"] = state["retry"].model_copy(update={"max_retries": 0, "max_content_repairs": 0})
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(_document_model(text_document(rendered)))(state))

    state.update(make_post_synthesis_validation_node(
        False, decisions_client=make_decisions_client({"format_fulfilled": probability}),
    )(state))

    assert state["response"].kind == expected_kind
    if expected_kind == "answer":
        report = state["response"].result.semantic_validation
        assert any(check.name == "format_fulfilled" and check.probability == probability for check in report.checks)


@pytest.mark.parametrize("previous_answer", [False, True])
@pytest.mark.parametrize("text, layout", [
    ("```text\nrequest timeout\n```", "code_block"),
    ("1. alpha\n2. beta", "ordered_list"),
    ("- alpha\n- beta", "list"),
    ("| item | value |\n| --- | --- |\n| alpha | 1 |", "table"),
    ("# Original heading\n\nOriginal text", "headings"),
])
def test_existing_layout_does_not_bypass_transformation_scope_validation(previous_answer, text, layout):
    previous = finalize_answer(text_document(text), []) if previous_answer else None

    state = _plan_copy(text, formats=[{"kind": layout, "mode": "required"}], previous=previous)

    contract = state["runtime"].request_contract
    assert contract.body.kind == ("transform_answer" if previous_answer else "transform_input")
    if previous_answer:
        assert contract.body.source.response_hash == previous.content_hash
    else:
        assert contract.body.source.text == text
        assert contract.body.source.content_hash == sha256(text.encode("utf-8")).hexdigest()
    prepare_synthesis_state(state)
    updates = make_synthesize_node(_document_model(text_document(text)))(state)
    assert updates["response"].kind != "clarification"
    assert updates["response"].result.content == text_document(text)


@pytest.mark.parametrize("previous_answer", [False, True])
@pytest.mark.parametrize("formats", [[], [{"kind": "inline_code", "mode": "preferred"}]])
def test_pure_copy_and_soft_layout_preferences_preserve_the_exact_copy_path(previous_answer, formats):
    text = "  original text\r\nwith whitespace  "
    previous = finalize_answer(text_document(text), []) if previous_answer else None
    state = _plan_copy(text, formats=formats, previous=previous)

    assert exact_copy_ready(state["runtime"].request_contract, previous)
    prepare_synthesis_state(state)
    updates = make_synthesize_node(_document_model())(state)

    assert updates["response"].result.content == text_document(text)


@pytest.mark.parametrize("formats", [
    [{"kind": "line_count", "mode": "required", "value": 1}],
    [{"kind": "code_block", "mode": "required"}, {"kind": "line_count", "mode": "required", "value": 1}],
])
def test_line_count_conflict_cannot_authorize_changing_literal_content(formats):
    state = _plan_copy("first line\nsecond line", formats=formats)

    assert state["runtime"].request_contract.body.kind == "copy_input"
    prepare_synthesis_state(state)
    updates = make_synthesize_node(_document_model())(state)
    assert updates["response"].kind == "clarification"


def test_content_conflict_alone_cannot_authorize_rewriting_a_copied_answer():
    code = "value = 1"
    source = build_snapshot(source_uri="https://example.test/source.py", title="source.py",
                            media_type="text/x-python", source_type="official", content=code,
                            parser="python", parser_version="1")
    evidence = build_evidence(snapshot=source,
                              element=DocumentElement(element_id="source", kind="code", text=code, language="python"))
    previous = finalize_answer(text_document(code, basis="excerpt", refs=[evidence.id]), [evidence])

    state = _plan_copy(code, formats=[], content=[{"kind": "code", "mode": "forbidden"}], previous=previous)

    assert state["runtime"].request_contract.body.kind == "copy_answer"
    prepare_synthesis_state(state)
    updates = make_synthesize_node(_document_model())(state)
    assert updates["response"].kind == "clarification"


@pytest.mark.parametrize("previous_answer", [False, True])
@pytest.mark.parametrize("scoped_evidence", [False, True])
def test_layout_only_transform_reaches_a_saved_answer_without_changing_the_payload(tmp_path, monkeypatch, previous_answer, scoped_evidence):
    text = "  request timeout\r\nagent execution failed\r\nunexpected error  "
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    previous = finalize_answer(text_document(text), []) if previous_answer else None
    state = _plan_copy(text, formats=[{"kind": "code_block", "mode": "required"}], content=[
        {"kind": "example", "mode": "forbidden"}, {"kind": "explanation", "mode": "forbidden"},
    ], previous=previous, scoped_evidence=scoped_evidence)
    document = AnswerDocument(blocks=[CodeBlock(language="text", content=ContentUnit(text=text, basis="interaction"))])
    prepare_synthesis_state(state)
    state.update(make_synthesize_node(_document_model(document))(state))
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))

    state.update(make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state))

    assert state["response"].kind == "answer"
    assert state["response"].result.content == document
    receipt = state["response"].result.actions[0]
    assert receipt.status == "success"
    assert ("```text\n" + text + "\n```").encode("utf-8") in Path(receipt.file_path).read_bytes()
