"""Validation preserves the presentation and operation contracts it judges."""
import hashlib
import json

import pytest

from src.core.answer_schema import AnswerDocument, finalize_answer, text_document
from src.core.answer_schema.models import CodeBlock, ContentUnit, ParagraphBlock, TableBlock
from src.core.request_bodies import BoundInputText, ComposeBody, TransformInputBody
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.core.test_bound_transform_validation_scope import _bound_edit_state
from tests.core.test_decisions_validation import DecisionsBoundary, _evidence, _state
from tests.core import test_post_reference_projection
from tests.synthesis_fixtures import synthesis_excerpt_limits


@pytest.mark.parametrize("metadata_only", [False, True])
@pytest.mark.parametrize("source_kind", ["answer", "input"])
def test_required_transform_rejects_unchanged_display_even_when_model_would_approve(metadata_only, source_kind):
    state = _bound_edit_state()
    source = state["runtime"].previous_response
    if source_kind == "input":
        text = "마지막 줄: 42"
        contract = state["runtime"].request_contract
        state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract.model_copy(update={
            "body": TransformInputBody(source=BoundInputText(turn_id="current", text=text,
                start=0, end=len(text), content_hash=hashlib.sha256(text.encode()).hexdigest()),
                instruction=contract.body.instruction, change_requirement="required"),
        })})
    document = source.content
    if metadata_only:
        document = AnswerDocument(blocks=[ParagraphBlock(content=[
            ContentUnit(text="마지막 줄:", basis="example"), ContentUnit(text="42", basis="interaction"),
        ])])
    state["response"] = state["response"].model_copy(update={"result": finalize_answer(document, [])})
    boundary = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry
    state.update(updates)
    assert state["response"].kind == "draft"
    assert updates["retry"].retry_reason == "missing_content"
    assert boundary.requests == []


def test_required_layout_transform_accepts_changed_structure_with_unchanged_wording():
    state = _bound_edit_state()
    contract = state["runtime"].request_contract
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract.model_copy(update={
        "body": contract.body.model_copy(update={"instruction": "원문을 그대로 코드 블록으로 표시한다."}),
    })})
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(AnswerDocument(blocks=[CodeBlock(content=ContentUnit(text="마지막 줄: 42"))]), []),
    })

    updates = make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert updates["response"].kind == "answer"


def test_unchanged_required_transform_exhausts_repair_as_validation_failure():
    state = _bound_edit_state()
    state["retry"] = state["retry"].model_copy(update={"max_content_repairs": 0})
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(state["runtime"].previous_response.content, []),
    })

    updates = make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert updates["response"].kind == "failure"
    assert updates["response"].problem.code == "answer_validation_failed"


def test_conditional_transform_allows_an_already_satisfied_unchanged_document():
    state = _bound_edit_state()
    contract = state["runtime"].request_contract
    body = contract.body.model_copy(update={"instruction": "이미 한 문장이면 그대로 유지한다.",
                                            "change_requirement": "if_needed"})
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": contract.model_copy(update={"body": body}),
    })
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(state["runtime"].previous_response.content, []),
    })

    updates = make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert updates["response"].kind == "answer"


def test_table_cell_grounding_receives_its_headers_and_row_without_borrowing_citations():
    state = _state()
    evidence = state["response"].evidence_packet[0]
    other = _evidence("Other source only supports the reason.", subject="other")
    document = AnswerDocument(blocks=[TableBlock(
        columns=[ContentUnit(text="센서"), ContentUnit(text="판정"), ContentUnit(text="사유")],
        rows=[[ContentUnit(text="C", basis="source", refs=[evidence.id]),
               ContentUnit(text="제외", basis="source", refs=[evidence.id]),
               ContentUnit(text="결측", basis="source", refs=[other.id])]],
    )])
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(document, [evidence, other], retrieval_required=True),
        "evidence_packet": [evidence, other],
        "evidence_requirement_map": {evidence.id: ["path_exists"], other.id: ["path_exists"]},
    })
    boundary = DecisionsBoundary()

    make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    cell, = [json.loads(call["input"])["units"][0] for call in boundary.requests
             if [question["name"] for question in call["questions"]] == ["grounded_3"]]
    assert cell["presentation_context"] == {
        "columns": ["센서", "판정", "사유"], "row": ["C", "제외", "결측"], "column_index": 0,
    }
    assert [source["id"] for source in cell["cited_sources"]] == [cell["refs"][0]]
    assert other.id not in json.dumps(cell, ensure_ascii=False)


def test_post_reuses_the_identical_prepared_requirement_judgments():
    state = _state()
    pre = DecisionsBoundary({"requirement_0_target_identified": 0.94,
                             "requirement_0_target_analysis_supported": 0.93})
    draft = state["response"]
    state.update(make_pre_synthesis_validation_node(False, decisions_client=pre,
        excerpt_limits=synthesis_excerpt_limits())(state))
    state["response"] = draft
    post = DecisionsBoundary({"requirement_0_target_identified": 0.1,
                              "requirement_0_target_analysis_supported": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=post)(state)

    assert updates["response"].kind == "answer"
    assert all("requirements" not in json.loads(call["input"]) for call in post.requests)
    checks = {check.name: check.probability for check in updates["response"].result.semantic_validation.checks}
    assert checks["requirement_target_identified:path_exists"] == 0.94
    assert checks["requirement_target_analysis_supported:path_exists"] == 0.93


@pytest.mark.parametrize("stage", ["pre", "post"])
def test_requirement_judgment_preserves_explicit_unknown_outcome_across_search_rewrites(stage):
    state = _state()
    instruction = "자료의 값을 알려줘. 자료에 정해지지 않았다면 확인할 수 없다고 알려줘."
    task = state["planner"].output.tasks[0].model_copy(update={"query": instruction})
    state["planner"] = state["planner"].model_copy(update={
        "output": state["planner"].output.model_copy(update={"tasks": [task.model_copy(update={"query": "value lookup"})]}),
    })
    state["retry"] = state["retry"].model_copy(update={"original_tasks": [task]})
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": state["runtime"].request_contract.model_copy(update={
            "body": ComposeBody(instruction="Also explain the unrelated second task."),
        }),
    })
    boundary = DecisionsBoundary()
    node = (make_pre_synthesis_validation_node(False, decisions_client=boundary,
        excerpt_limits=synthesis_excerpt_limits()) if stage == "pre"
        else make_post_synthesis_validation_node(False, decisions_client=boundary))

    node(state)

    payload, = [json.loads(call["input"]) for call in boundary.requests
                if "requirements" in json.loads(call["input"])]
    assert payload["tasks"]["path_exists"]["query"] == instruction
    assert "unrelated second task" not in json.dumps(payload)


def test_post_checks_new_cited_selection_instead_of_borrowing_prepared_approval():
    state = _state()
    draft = state["response"]
    state.update(make_pre_synthesis_validation_node(False, decisions_client=DecisionsBoundary(),
        excerpt_limits=synthesis_excerpt_limits())(state))
    different = _evidence("Path.exists()", subject="pathlib.Path.exists")
    state["response"] = draft.model_copy(update={
        "result": finalize_answer(text_document("Path.exists() returns True", basis="source", refs=[different.id]),
                                  [different], retrieval_required=True),
        "evidence_packet": [different], "evidence_requirement_map": {different.id: ["path_exists"]},
    })
    boundary = DecisionsBoundary({"requirement_0_target_analysis_supported": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry
    assert updates["retry"].failed_requirement_ids == ["path_exists"]
    assert any("requirements" in json.loads(call["input"]) for call in boundary.requests)


def test_citation_scope_keeps_actual_source_kind_and_independently_blocks_truthful_body():
    state = _state()
    instruction = "첨부 코드만 인용해서 설명해줘."
    state["runtime"] = state["runtime"].model_copy(update={
        "request_contract": state["runtime"].request_contract.model_copy(update={
            "body": ComposeBody(instruction=instruction),
        }),
    })
    boundary = DecisionsBoundary({"request_fulfilled": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    body, = [json.loads(call["input"]) for call in boundary.requests
             if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    assert body["confirmed_request"]["body"]["instruction"] == instruction
    source, = body["units"][0]["cited_sources"]
    assert source["kind"] == "paragraph"
    assert source["route"] == "docs"
    assert updates["retry"].needs_retry


def test_verified_source_location_uses_local_proof_and_keeps_full_request_context():
    state = test_post_reference_projection._state(bound_previous=False, terminal_partial=False)
    boundary = DecisionsBoundary({"grounded_1": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["response"].kind == "answer"
    questions = {question["name"] for call in boundary.requests for question in call["questions"]}
    assert "grounded_1" not in questions
    assert {"grounded_0", "grounded_2", "request_fulfilled", "requirement_0_target_identified",
            "requirement_0_target_analysis_supported"} <= questions
    body, = [json.loads(call["input"]) for call in boundary.requests
             if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    location = body["units"][1]
    assert location["text"] == "first.py:1"
    assert location["source_location"] == {"ref": "e1", "start": 0, "end": 10}
    assert location["cited_sources"][0]["source_lines"][0]["line_number"] == 1
    report = updates["response"].result.semantic_validation
    assert not any(check.name == "grounded" and check.unit_id == "b1.content.0" for check in report.checks)


def test_source_location_records_deterministic_proof_without_an_excerpt_match_or_model_score():
    state = test_post_reference_projection._state(bound_previous=False, terminal_partial=False)
    result = state["response"].result

    assert result.checks[1].support_status == "verified_location"
    assert result.checks[0].support_status == "not_evaluated"
    assert result.semantic_validation is None


@pytest.mark.parametrize("invalid", ["outside_selection", "mixed_prose"])
def test_invalid_source_location_remains_a_local_failure_even_if_model_approves(invalid):
    state = test_post_reference_projection._state(bound_previous=False, terminal_partial=False)
    response = state["response"]
    document = response.result.content.model_copy(deep=True)
    unit = document.blocks[1].content[0]
    if invalid == "outside_selection":
        unit.source_location = unit.source_location.model_copy(update={"end": 11})
    else:
        unit.text = "first.py:1, and the value is 999."
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, response.evidence_packet, retrieval_required=True),
    })

    updates = make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state)

    assert updates["retry"].needs_retry
    check, = [check for check in state["response"].result.checks if check.unit_id == "b1.content.0"]
    assert check.support_status == "unsupported"
    assert check.issues == ["source_location_mismatch"]


def test_verified_source_location_does_not_discharge_whole_request_fulfillment():
    state = test_post_reference_projection._state(bound_previous=False, terminal_partial=False)
    boundary = DecisionsBoundary({"request_fulfilled": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry
    assert updates["retry"].retry_reason == "missing_content"
