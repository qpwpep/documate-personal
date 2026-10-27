from pathlib import Path

import pytest
from pydantic import ValidationError

from src.core.answer_schema import (
    AnswerDocument, CodeBlock, ContentUnit, SemanticCheck, SemanticValidation,
    finalize_answer, semantic_evidence_fingerprint, text_document,
)
from src.core.contracts import PlannerState, ResponseState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import build_evidence
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    ActionContract, ActionRequest, AnswerContract, ContentRequirement, ContractEvidence, LayoutRequirement, RequestContract,
    check_answer_contract,
)
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from src.runtime.nodes.actions import make_action_postprocess_node
from src.infra.tools.save_text import build_save_text_tool
from tests.core.helpers import make_decisions_client


@pytest.mark.parametrize(("text", "present"), [
    ("1행은 x = 1이고 2행은 y = 2입니다.", False),
    ("1행은 `x = 1`이고 2행은 `y = 2`입니다.", True),
    (r"Show \`x = 1\` literally.", False),
    ("The unmatched delimiter is `x = 1.", False),
    ("```python\nprint('`label`')\n```", False),
    ("    print('`label`')", False),
    ("Use ``value = '`'`` here.", True),
])
def test_inline_code_requirement_observes_rendered_markdown_spans(text, present):
    document = text_document(text)
    required = AnswerContract(format=(LayoutRequirement(kind="inline_code", mode="required"),))
    forbidden = AnswerContract(format=(LayoutRequirement(kind="inline_code", mode="forbidden"),))

    assert check_answer_contract(required, document).missing_required == ([] if present else ["inline_code"])
    assert check_answer_contract(forbidden, document).forbidden_present == (["inline_code"] if present else [])


@pytest.mark.parametrize("inline", [False, True])
def test_structured_code_layout_is_not_inferred_from_its_literal_contents(inline):
    document = AnswerDocument(blocks=[CodeBlock(
        inline=inline, content=ContentUnit(text="print('`label`')", basis="interaction"),
    )])
    contract = AnswerContract(format=(LayoutRequirement(kind="inline_code", mode="required"),))

    assert check_answer_contract(contract, document).missing_required == ([] if inline else ["inline_code"])








def test_inline_code_preference_does_not_become_a_required_layout():
    contract = AnswerContract(format=(LayoutRequirement(kind="inline_code", mode="preferred"),))

    assert check_answer_contract(contract, text_document("Plain text is acceptable.")).valid


def test_present_layout_still_requires_semantic_scope_validation():
    contract = AnswerContract(format=(LayoutRequirement(kind="inline_code", mode="required"),))

    check = check_answer_contract(contract, text_document("`alpha` and beta"))

    assert check.missing_required == []
    assert check.unchecked_semantic == ["required:inline_code"]


@pytest.mark.parametrize("forbidden_content", ["example", "code_example"])
def test_required_code_block_can_present_input_without_examples(forbidden_content):
    """코드 블록은 로그나 원문도 표현하므로 예시 금지와 양립한다."""
    contract = AnswerContract(
        content=(ContentRequirement(kind=forbidden_content, mode="forbidden"),),
        format=(LayoutRequirement(kind="code_block", mode="required"),),
    )

    assert contract.content[0].kind == forbidden_content
    assert contract.format[0].kind == "code_block"


@pytest.mark.parametrize("kind", ["example", "code_example"])
def test_input_log_block_requires_semantic_example_judgment(kind):
    """표현 형식은 관찰하지만 로그가 예시인지는 의미 검증에 맡긴다."""
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="2026-10-23 INFO request completed", basis="interaction"),
    )])
    contract = AnswerContract(
        content=(ContentRequirement(kind=kind, mode="forbidden"),),
        format=(LayoutRequirement(kind="code_block", mode="required"),),
    )

    check = check_answer_contract(contract, document)

    assert check.forbidden_present == []
    assert check.missing_required == []
    assert check.unchecked_semantic == [f"forbidden:{kind}", "required:code_block"]


def _validation_state(document, *, kind="example", mode="forbidden", format_mode="required"):
    request = "입력 로그를 코드 블록으로 표시하되 예시는 추가하지 마."
    contract = RequestContract(
        answer=AnswerContract(
            content=(ContentRequirement(kind=kind, mode=mode, evidence_ids=("content",)),),
            format=(LayoutRequirement(kind="code_block", mode=format_mode, evidence_ids=("format",)),),
        ),
        evidence=(
            ContractEvidence(id="content", turn_id="user:0", quote=request,
                             scope="answer.content", interpretation="instruction"),
            ContractEvidence(id="format", turn_id="user:0", quote=request,
                             scope="answer.format", interpretation="instruction"),
        ),
    )
    return build_graph_state_input(
        user_input=request, request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        retry=RetryState(max_retries=0, max_content_repairs=0),
        response=ResponseState(result=finalize_answer(document, []), kind="answer",
                               request_id=contract.request_id, contract_revision=contract.revision),
    )


def test_validated_input_log_remains_an_answer_when_examples_are_forbidden():
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="INFO request completed", basis="interaction"),
    )])

    updates = make_post_synthesis_validation_node(
        False, decisions_client=make_decisions_client(),
    )(_validation_state(document))

    assert updates["response"].kind == "answer"
    assert updates["response"].result.content == document
    assert updates["response"].result.semantic_validation.passed


@pytest.mark.parametrize("basis", ["example", "interaction", "inference"])
@pytest.mark.parametrize("kind", ["example", "code_example"])
def test_forbidden_example_cannot_bypass_semantics_with_a_different_basis(kind, basis):
    document = AnswerDocument(blocks=[CodeBlock(
        language="python", content=ContentUnit(text="print('illustrative example')", basis=basis),
    )])

    updates = make_post_synthesis_validation_node(
        False, decisions_client=make_decisions_client({"request_fulfilled": 0.1}),
    )(_validation_state(document, kind=kind))

    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None


def test_code_block_prohibition_still_blocks_semantically_accepted_content():
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="INFO request completed", basis="interaction"),
    )])

    updates = make_post_synthesis_validation_node(
        False, decisions_client=make_decisions_client(),
    )(_validation_state(document, format_mode="forbidden"))

    assert updates["response"].kind == "failure"


def test_text_log_block_does_not_prove_program_code_is_present():
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="INFO request completed", basis="interaction"),
    )])
    contract = AnswerContract(content=(ContentRequirement(kind="code", mode="forbidden"),))

    check = check_answer_contract(contract, document)

    assert check.forbidden_present == []
    assert check.unchecked_semantic == ["forbidden:code"]


def test_required_code_example_conflicts_with_forbidden_program_code():
    with pytest.raises(ValidationError, match="conflicting required and forbidden content"):
        AnswerContract(content=(
            ContentRequirement(kind="code_example", mode="required"),
            ContentRequirement(kind="code", mode="forbidden"),
        ))


def _request_save(state):
    contract = state["runtime"].request_contract
    contract = RequestContract.model_validate({
        **contract.model_dump(mode="python"),
        "actions": ActionContract(save_text=ActionRequest(intent="requested", evidence_ids=("save",))),
        "evidence": (*contract.evidence, ContractEvidence(
            id="save", turn_id="user:0", quote="저장해줘", scope="actions.save_text", interpretation="instruction",
        )),
    })
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    return state


def test_unchecked_example_condition_cannot_be_saved_without_semantic_validation(tmp_path, monkeypatch):
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    document = AnswerDocument(blocks=[CodeBlock(
        language="python", content=ContentUnit(text="print('illustrative example')", basis="interaction"),
    )])
    state = _request_save(_validation_state(document))

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state)

    assert not output.exists()
    assert "response" not in updates


@pytest.mark.parametrize("judgment", ["rejected", "refused", "missing_request_check"])
def test_delivery_requires_a_passing_request_fulfillment_check(tmp_path, monkeypatch, judgment):
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="INFO request completed", basis="interaction"),
    )])
    state = _request_save(_validation_state(document))
    answer = state["response"].result
    check = SemanticCheck(
        name="grounded" if judgment == "missing_request_check" else "request_fulfilled",
        status="refused" if judgment == "refused" else "evaluated",
        probability=None if judgment == "refused" else (0.1 if judgment == "rejected" else 0.99),
    )
    report = SemanticValidation(content_hash=answer.content_hash,
                                evidence_fingerprint=semantic_evidence_fingerprint(answer.citations),
                                threshold=0.8, checks=[check])
    state["response"] = state["response"].model_copy(update={
        "result": answer.model_copy(update={"semantic_validation": report}),
    })

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state)

    assert not output.exists()
    assert "response" not in updates


@pytest.mark.parametrize("stale_field", ["content_hash", "evidence_fingerprint"])
def test_delivery_rejects_semantic_report_for_another_answer_revision(tmp_path, monkeypatch, stale_field):
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="INFO request completed", basis="interaction"),
    )])
    state = _request_save(_validation_state(document))
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))
    answer = state["response"].result
    report = answer.semantic_validation.model_copy(update={stale_field: "another-revision"})
    state["response"] = state["response"].model_copy(update={
        "result": answer.model_copy(update={"semantic_validation": report}),
    })

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state)

    assert not output.exists()
    assert "response" not in updates


@pytest.mark.parametrize("kind", ["code", "example", "code_example"])
def test_semantically_accepted_log_block_can_be_saved_with_content_prohibitions(tmp_path, monkeypatch, kind):
    output = tmp_path / "saved"
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: output)
    document = AnswerDocument(blocks=[CodeBlock(
        language="text", content=ContentUnit(text="INFO request completed", basis="interaction"),
    )])
    state = _request_save(_validation_state(document, kind=kind))
    state.update(make_post_synthesis_validation_node(False, decisions_client=make_decisions_client())(state))

    updates = make_action_postprocess_node(build_save_text_tool(), lambda **kwargs: {}, False)(state)

    receipt = updates["response"].result.actions[0]
    assert receipt.status == "success"
    assert updates["response"].result.content == document
    assert Path(receipt.file_path).read_text(encoding="utf-8-sig").endswith("```text\nINFO request completed\n```")


def test_cited_code_does_not_prove_a_more_specific_content_condition():
    text = "value = 1"
    source = build_snapshot(source_uri="https://example.test/source.py", title="source.py",
                            media_type="text/x-python", source_type="official", content=text,
                            parser="python", parser_version="1")
    evidence = build_evidence(snapshot=source,
                              element=DocumentElement(element_id="source", kind="code", text=text, language="python"))
    document = AnswerDocument(blocks=[CodeBlock(
        language="python", content=ContentUnit(text=text, basis="excerpt", refs=[evidence.id]),
    )])
    contract = AnswerContract(content=(ContentRequirement(
        kind="code", mode="forbidden", description="새로 작성한 구현 코드",
    ),))

    check = check_answer_contract(contract, document, evidence=[evidence])

    assert check.forbidden_present == []
    assert check.unchecked_semantic == ["forbidden:code"]


@pytest.mark.parametrize("kind", ["example", "code_example"])
def test_original_code_excerpt_does_not_by_itself_prove_an_example(kind):
    text = "value = 1"
    source = build_snapshot(source_uri="https://example.test/source.py", title="source.py",
                            media_type="text/x-python", source_type="official", content=text,
                            parser="python", parser_version="1")
    evidence = build_evidence(snapshot=source,
                              element=DocumentElement(element_id="source", kind="code", text=text, language="python"))
    document = AnswerDocument(blocks=[CodeBlock(
        language="python", content=ContentUnit(text=text, basis="excerpt", refs=[evidence.id]),
    )])
    contract = AnswerContract(content=(ContentRequirement(kind=kind, mode="forbidden"),))

    check = check_answer_contract(contract, document, evidence=[evidence])

    assert check.forbidden_present == []
    assert check.unchecked_semantic == [f"forbidden:{kind}"]


@pytest.mark.parametrize("source_text, selected", [
    ("amount = 1\n", "amount"),
    ('filename = "sample.py"\n', "sample.py"),
    ('log = """\nINFO request completed\n"""\n', "INFO request completed"),
    ("amount\n", "amount"),
])
@pytest.mark.parametrize("inline", [False, True])
def test_exact_selected_text_does_not_make_a_fragment_or_literal_program_code(source_text, selected, inline):
    source = build_snapshot(source_uri="https://example.test/source.py", title="source.py",
                            media_type="text/x-python", source_type="official", content=source_text,
                            parser="python", parser_version="1")
    start = source_text.index(selected)
    evidence = build_evidence(
        snapshot=source, start=start, end=start + len(selected),
        element=DocumentElement(element_id="source", kind="code", text=source_text, language="python"),
    )
    document = text_document(f"`{selected}`" if inline else selected, basis="source", refs=[evidence.id])
    contract = AnswerContract(content=(ContentRequirement(kind="code", mode="forbidden"),))

    check = check_answer_contract(contract, document, evidence=[evidence])

    assert check.forbidden_present == []
    assert check.unchecked_semantic == ["forbidden:code"]


@pytest.mark.parametrize("container", ["code_record", "fenced_markdown", "indented_markdown"])
def test_layout_syntax_inside_literal_code_does_not_count_as_displayed_layout(container):
    text = "# Heading\n\n1. alpha\n2. beta\n\n| item | value |\n| --- | --- |\n| alpha | 1 |"
    if container == "code_record":
        document = AnswerDocument(blocks=[CodeBlock(language="text", content=ContentUnit(text=text))])
    elif container == "fenced_markdown":
        document = text_document("```text\n" + text + "\n```")
    else:
        document = text_document("\n".join("    " + line for line in text.splitlines()))
    required = AnswerContract(format=tuple(LayoutRequirement(kind=kind, mode="required")
                                          for kind in ("ordered_list", "list", "table", "headings")))
    forbidden = AnswerContract(format=tuple(LayoutRequirement(kind=kind, mode="forbidden")
                                           for kind in ("ordered_list", "list", "table", "headings")))

    assert check_answer_contract(required, document).missing_required == ["ordered_list", "list", "table", "headings"]
    assert check_answer_contract(forbidden, document).forbidden_present == []
