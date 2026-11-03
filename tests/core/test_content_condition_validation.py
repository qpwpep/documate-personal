"""Confirmed content conditions independently control answer acceptance and repair."""
from hashlib import sha256

import pytest
from langchain_core.messages import HumanMessage

from src.core.answer_schema import export_answer_text, finalize_answer, text_document
from src.core.contracts import GraphState, PlannerState, ResponseState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import (
    AnswerContract, BoundInputText, ContentRequirement, ContractEvidence, CopyInputBody, RequestContract,
)
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary, read_decision_input


def _state(contract: RequestContract, text: str, *, repairs: int = 1) -> GraphState:
    request = " ".join(evidence.quote for evidence in contract.evidence)
    return build_graph_state_input(
        user_input=request,
        messages=[HumanMessage(content=request, id="current")],
        request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=False, tasks=[])),
        retry=RetryState(max_retries=0, max_content_repairs=repairs),
        response=ResponseState(
            result=finalize_answer(text_document(text), []), synthesis_attempt=1,
            request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


def _contract(*conditions: ContentRequirement) -> RequestContract:
    return RequestContract(
        answer=AnswerContract(content=tuple(
            condition.model_copy(update={"evidence_ids": ("instruction",)}) for condition in conditions
        )),
        evidence=(ContractEvidence(
            id="instruction", turn_id="current", scope="answer.content", interpretation="instruction",
            quote=" ".join(condition.description for condition in conditions),
        ),),
    )


@pytest.mark.parametrize(("mode", "description", "text", "refused"), [
    ("required", "각 메서드가 True를 반환하는 경우를 각각 한 문장으로 설명한다.",
     "is_file은 일반 파일에서 True를 반환하고, is_dir은 디렉터리에서 True를 반환합니다.", False),
    ("required", "각 선택지마다 독립된 사용 예시를 두 개 제시한다.",
     "첫 번째 선택지는 짧은 작업에 적합하고 두 번째 선택지는 긴 작업에 적합합니다.", False),
    ("forbidden", "각 항목의 설명에 확인하지 않은 실행 결과를 추가하지 않는다.",
     "첫 번째 항목은 직접 실행해 보니 항상 10초 안에 완료되었습니다.", False),
    ("required", "각 항목의 목적을 설명한다.", "첫 번째 항목은 입력을 읽습니다.", True),
], ids=["sentences-per-target", "examples-per-target", "forbidden-execution-claim", "refused-condition"])
def test_rejected_condition_requests_repair_despite_passing_overall_judgment(
    mode, description, text, refused,
):
    condition = ContentRequirement(kind="custom", mode=mode, description=description)
    contract = _contract(condition)
    decisions = DecisionsBoundary(
        {"content_condition_0": 0.1},
        refused_names={"content_condition_0"} if refused else (),
    )

    updates = make_post_synthesis_validation_node(False, decisions_client=decisions)(
        _state(contract, text),
    )

    assert updates["retry"].needs_retry
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert updates["retry"].failed_routes == []
    assert "semantic_content_condition:0" in updates["retry"].retrieval_feedback
    assert "response" not in updates


def test_preference_is_not_a_gate_and_same_kind_conditions_keep_their_contract_indexes():
    contract = _contract(
        ContentRequirement(kind="custom", mode="preferred", description="가능하면 친근한 어조를 쓴다."),
        ContentRequirement(kind="custom", mode="required", description="각 대상의 목적을 따로 설명한다."),
        ContentRequirement(kind="custom", mode="forbidden", description="확인하지 않은 실행 결과를 주장하지 않는다."),
    )
    text = "첫 번째 대상은 입력을 읽습니다. 두 번째 대상은 결과를 저장합니다."
    decisions = DecisionsBoundary({"content_condition_0": 0.01})

    updates = make_post_synthesis_validation_node(False, decisions_client=decisions)(_state(contract, text))

    assert not updates["retry"].needs_retry
    response = updates["response"]
    assert response.kind == "answer"
    assert export_answer_text(response.result) == text
    report = response.result.semantic_validation
    assert report is not None and report.passed
    assert [check.name for check in report.checks if check.name.startswith("content_condition:")] == [
        "content_condition:1", "content_condition:2",
    ]
    request = decisions.requests[0]
    condition_questions = {
        question["name"]: question for question in request["questions"]
        if question["name"].startswith("content_condition_")
    }
    assert set(condition_questions) == {"content_condition_1", "content_condition_2"}
    for index in (1, 2):
        instructions = condition_questions[f"content_condition_{index}"]["instructions"]
        assert f"confirmed_request.answer.content[{index}]" in instructions
    payload = read_decision_input(request["input"])
    assert payload["confirmed_request"]["answer"]["content"] == [
        condition.model_dump(mode="json") for condition in contract.answer.content
    ]
    assert len(decisions.requests) == 1


def test_failed_condition_cannot_become_an_answer_when_repair_budget_is_exhausted():
    contract = _contract(ContentRequirement(
        kind="explanation", mode="required", description="각 대상의 이유를 각각 설명한다.",
    ))
    decisions = DecisionsBoundary({"content_condition_0": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=decisions)(
        _state(contract, "대상 이름만 나열합니다.", repairs=0),
    )

    assert not updates["retry"].needs_retry
    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None
    assert "semantic_content_condition:0" in updates["retry"].retrieval_feedback


def test_condition_check_preserves_exact_input_with_dots_code_and_whitespace():
    original = '  pathlib.Path("report.v1.txt").is_file()\nvalue = 3.14  \n'
    prefix = "Copy this input exactly, preserving the path and decimal literal:\n"
    request = prefix + original
    contract = RequestContract(
        body=CopyInputBody(source=BoundInputText(
            turn_id="current", text=original, start=len(prefix), end=len(request),
            content_hash=sha256(original.encode("utf-8")).hexdigest(),
        )),
        answer=AnswerContract(content=(ContentRequirement(
            kind="custom", mode="required",
            description="Preserve the path report.v1.txt and decimal literal 3.14.",
            evidence_ids=("instruction",),
        ),)),
        evidence=(ContractEvidence(
            id="instruction", turn_id="current", quote=request,
            scope="current_request", interpretation="instruction",
        ),),
    )
    decisions = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=decisions)(
        _state(contract, original),
    )

    response = updates["response"]
    assert response.kind == "answer"
    assert export_answer_text(response.result) == original
    report = response.result.semantic_validation
    assert report is not None and report.passed
    assert [check.name for check in report.checks] == ["request_fulfilled", "content_condition:0"]
    payload = read_decision_input(decisions.requests[0]["input"])
    assert payload["rendered_response"]["body"]["blocks"][0]["content"][0]["text"] == original
