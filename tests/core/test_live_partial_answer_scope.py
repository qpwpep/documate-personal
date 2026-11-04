"""Structured sentence quotas are local gates, including when a model approves everything.

From the repository root in Git Bash:
    RUN_LIVE_PARTIAL_ANSWER_SCOPE=1 uv run --locked pytest -q tests/core/test_live_partial_answer_scope.py

The default suite uses an approving provider boundary and real owned validation.
Opt-in cases send fixed official-document excerpts and candidate answers to the
configured Decisions model, permitting at most three paid calls per case. Sentence
quotas remain local checks regardless of semantic probability. Punctuation marks
bound sentences; version/API dots and inline-code text do not create boundaries.
No retrieval, generation, saving or delivery runs in this module.
"""
from __future__ import annotations

import json
import os
from collections.abc import Iterator

import pytest
from langchain_core.messages import HumanMessage
from openai import OpenAI

from src.core.answer_schema import AnswerDocument, finalize_answer
from src.core.contracts import (
    DebugState, GraphState, PlannerState, ResponseState, RetrievalDiagnostic, RetrievalState, RetryState,
)
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.decisions import DecisionResult
from src.core.documents import DocumentElement, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import (
    AnswerContract, ComposeBody, ContentRequirement, ContractEvidence, LayoutRequirement, RequestContract,
)
from src.infra.decisions import DecisionsClient
from src.infra.llm_boundary import capture_llm_diagnostics, request_call_budget
from src.infra.settings import get_settings
from src.runtime.agent_runtime.response_assembler import ResponseAssembler
from src.runtime.nodes.validation import make_post_synthesis_validation_node
from tests.core.test_decisions_validation import DecisionsBoundary

REQUEST = (
    "pathlib.Path.is_file()과 Path.is_dir()의 차이를 Python 공식 문서로 확인해 줘. "
    "각 메서드가 True를 반환하는 경우를 각각 한 문장으로 설명해 줘. "
    "코드 예시나 표는 넣지 마."
)
# Retained verbatim from the official page captured during the reproduction.
SOURCE_TEXT = """Path.is_file(*, follow_symlinks=True)¶

Return True if the path points to a regular file. False will be returned if the path is invalid, inaccessible or missing, or if it points to something other than a regular file. Use Path.stat() to distinguish between these cases.

This method normally follows symlinks; to exclude symlinks, add the argument follow_symlinks=False.

버전 3.13에서 변경: The follow_symlinks parameter was added.
"""
DIRECTORY_SOURCE_TEXT = """Path.is_dir(*, follow_symlinks=True)¶

Return True if the path points to a directory.
"""
ONE_SENTENCE = (
    "Python 3.15.2 공식 문서에 따르면, `Path.is_file()`은 경로가 일반 파일을 가리킬 때 `True`를 반환합니다."
)
TWO_SENTENCES = (
    ONE_SENTENCE + " 심볼릭 링크가 일반 파일을 가리키는 경우에도 기본 설정에서는 `Path.is_file()`이 `True`를 반환합니다."
)
INTRODUCTION = "`Path.is_file()`은 주어진 경로가 일반 파일인지 판별하는 메서드입니다."
DIRECTORY_SENTENCE = "`Path.is_dir()`은 경로가 디렉터리를 가리킬 때 `True`를 반환합니다."


@pytest.fixture(scope="module")
def live_decisions_client() -> Iterator[DecisionsClient]:
    settings = get_settings()
    with OpenAI(api_key=settings.openai_api_key, max_retries=0) as sdk:
        yield DecisionsClient(sdk, model=settings.validation_model,
                              timeout_seconds=settings.validation_timeout_seconds)


def _partial_state(
    text: str, *, complete: bool = False, sentence_scope: int | None = 1,
    following_parts: tuple[tuple[str, int | None], ...] = (),
) -> GraphState:
    contract = RequestContract(
        body=ComposeBody(instruction=REQUEST),
        answer=AnswerContract(content=(
            ContentRequirement(kind="explanation", mode="required", evidence_ids=("request",),
                description="Explain when each requested method returns True."),
            ContentRequirement(kind="code_example", mode="forbidden", evidence_ids=("request",),
                description="Do not include code examples."),
        ), format=(LayoutRequirement(kind="table", mode="forbidden", evidence_ids=("request",)),),
            sentence_counts=tuple({
                "scope": scope, "minimum": 1, "maximum": 1, "mode": "required", "evidence_ids": ["request"],
            } for scope in ("Path.is_file()이 True를 반환하는 경우", "Path.is_dir()이 True를 반환하는 경우")),
        ),
        evidence=(ContractEvidence(id="request", turn_id="current", quote=REQUEST,
            scope="current_request", interpretation="instruction"),),
    )
    source_sections = {"is_file": SOURCE_TEXT}
    if complete:
        source_sections["is_dir"] = DIRECTORY_SOURCE_TEXT
    snapshot = build_snapshot(
        source_uri="https://docs.python.org/ko/3/library/pathlib.html",
        title="pathlib — Object-oriented filesystem paths — Python 3.15.2 문서",
        media_type="text/html", source_type="official", content="\n".join(source_sections.values()),
        parser="official_html", parser_version="1", quality_issues=["html_structure_flattened"],
    )
    evidence = {method: build_evidence(snapshot=snapshot, element=DocumentElement(
        element_id=f"path_{method}", kind="paragraph", text=source_text,
    )) for method, source_text in source_sections.items()}
    tasks = [RetrievalTask(
        route="docs", query=f"Python 공식 문서에서 pathlib.Path.{method}가 True를 반환하는 경우", k=1,
        requirement_id=method, requirement={"library": "Python", "symbols": [f"pathlib.Path.{method}"],
                                           "match": "symbol", "evidence_needs": ["explanation"]},
    ) for method in ("is_file", "is_dir")]
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [
        {"text": part, "basis": "source", "refs": [item.id for item in evidence.values()], "sentence_scope": scope}
        for part, scope in ((text, sentence_scope), *following_parts)
    ]}]})
    return build_graph_state_input(
        user_input=REQUEST, messages=[HumanMessage(content=REQUEST, id="current")],
        request_contract=contract, planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=tasks)),
        retrieval=RetrievalState(hit_log=[SearchHit(evidence=item, rank=1, requirement_id=method,
            score=RetrievalScore(metric="captured_source", direction="higher"))
            for method, item in evidence.items()]),
        debug=DebugState(retrieval_diagnostics=[
            RetrievalDiagnostic(route="docs", requirement_id="is_file", status="success",
                                answerability="covered", evidence_count=1),
            RetrievalDiagnostic(route="docs", requirement_id="is_dir",
                                status="success" if complete else "no_result",
                                answerability="covered" if complete else "missing",
                                evidence_count=1 if complete else 0),
        ]),
        retry=RetryState(max_retries=0, max_content_repairs=0,
                        terminal_failed_requirement_ids=[] if complete else ["is_dir"],
                        terminal_retrieval_reason=None if complete else "no_evidence"),
        response=ResponseState(
            result=finalize_answer(document, list(evidence.values()), retrieval_required=True),
            evidence_packet=list(evidence.values()),
            evidence_requirement_map={item.id: [method] for method, item in evidence.items()},
            normal_evidence_missing_requirement_ids=[] if complete else ["is_dir"], synthesis_attempt=1,
            request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


@pytest.mark.parametrize(("incomplete_task_indices", "expected_notice"), [
    pytest.param((1,), "공식 문서에서 pathlib.Path.is_dir에 필요한 근거를 충분히 확인하지 못했습니다.",
                 id="unsupported-target"),
    pytest.param((0, 1),
                 "공식 문서에서 pathlib.Path.is_file에 필요한 근거를 충분히 확인하지 못했습니다. "
                 "공식 문서에서 pathlib.Path.is_dir에 필요한 근거를 충분히 확인하지 못했습니다.",
                 id="supported-target-also-incomplete"),
])
def test_partial_sentence_quota_keeps_supported_body_and_contract(
    incomplete_task_indices: tuple[int, ...], expected_notice: str,
):
    state = _partial_state(ONE_SENTENCE)
    original = state["response"].result.content.model_copy(deep=True)
    contract = state["runtime"].request_contract
    incomplete_tasks = [state["planner"].output.tasks[index] for index in incomplete_task_indices]
    state["retry"].terminal_failed_requirement_ids = [task.requirement_id for task in incomplete_tasks]
    state["response"].normal_evidence_missing_requirement_ids = [task.requirement_id for task in incomplete_tasks]
    if 0 in incomplete_task_indices:
        state["debug"].retrieval_diagnostics[0].answerability = "partial"
    decisions = DecisionsBoundary()

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))

    body_request, = [call for call in decisions.requests
                    if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    payload = json.loads(body_request["input"])
    assert payload["confirmed_request"]["answer"] == contract.answer.model_dump(
        mode="json", exclude={"sentence_counts"},
    )
    assert "sentence_scopes" not in payload["answer_scope"]
    assert payload["answer_scope"]["sentence_scope_bindings"] == [{
        "unit_id": "b0.content.0", "text": ONE_SENTENCE,
        "scope": "Path.is_file()이 True를 반환하는 경우",
    }]
    assert payload["terminal_failed_requirement_ids"] == [task.requirement_id for task in incomplete_tasks]
    assert payload["rendered_response"]["notices"] == [expected_notice]
    scope = {item["task"]["requirement_id"]: item for item in payload["answer_scope"]["requirements"]}
    assert scope["is_file"]["coverage"] == ("partial" if 0 in incomplete_task_indices else "required")
    assert scope["is_file"]["available_source_ids"] == ["e1"]
    assert scope["is_file"]["cited_source_ids"] == scope["is_file"]["available_source_ids"]
    assert scope["is_dir"]["coverage"] == "unavailable"
    assert scope["is_dir"]["available_source_ids"] == scope["is_dir"]["cited_source_ids"] == []
    outcome = ResponseAssembler().assemble(response=state)
    assert outcome.status == "partial"
    assert state["response"].result.content == original
    assert state["response"].result.semantic_validation.passed
    assert state["runtime"].request_contract == contract


@pytest.mark.parametrize("supported_target_is_terminal", [False, True])
def test_uncited_supported_target_is_not_exempted_by_a_terminal_gap(supported_target_is_terminal):
    state = _partial_state(ONE_SENTENCE)
    if supported_target_is_terminal:
        state["retry"].terminal_failed_requirement_ids = ["is_file", "is_dir"]
    evidence = state["response"].evidence_packet[0]
    document = AnswerDocument.model_validate({"blocks": [{"type": "paragraph", "content": [{
        "text": "확인한 내용은 없습니다.", "basis": "interaction", "refs": [], "sentence_scope": 1,
    }]}]})
    state["response"].result = finalize_answer(document, [evidence], retrieval_required=True)
    boundary = DecisionsBoundary({"request_fulfilled": 0.1, "requirement_0_target_analysis_supported": 0.1})

    state.update(make_post_synthesis_validation_node(False, decisions_client=boundary)(state))

    body_request, = [call for call in boundary.requests
                    if any(question["name"] == "request_fulfilled" for question in call["questions"])]
    payload = json.loads(body_request["input"])
    scope = {item["task"]["requirement_id"]: item for item in payload["answer_scope"]["requirements"]}
    assert scope["is_file"]["coverage"] == ("partial" if supported_target_is_terminal else "required")
    assert scope["is_file"]["available_source_ids"] == ["e1"]
    assert scope["is_file"]["cited_source_ids"] == []
    assert scope["is_file"]["uncited_sources"][0]["excerpt"] == SOURCE_TEXT
    assert scope["is_dir"]["coverage"] == "unavailable"
    outcome = ResponseAssembler().assemble(response=state)
    assert outcome.status == "failed"
    assert "semantic_request_fulfillment" in state["retry"].retrieval_feedback


@pytest.mark.parametrize(("text", "sentence_scope", "following_parts", "complete", "missing"), [
    pytest.param(TWO_SENTENCES, 1, (), False, "sentence_count:1", id="additional-supported-detail"),
    pytest.param(INTRODUCTION + " " + ONE_SENTENCE, 1, (), False, "sentence_count:1", id="same-unit-introduction"),
    pytest.param(INTRODUCTION, 1, ((ONE_SENTENCE, 1),), False, "sentence_count:1", id="separate-unit-introduction"),
    pytest.param(INTRODUCTION, None, ((ONE_SENTENCE, 1),), False, "sentence_scope", id="unassigned-introduction"),
    pytest.param("`Path.is_file()`은 일반 파일에서 True를 반환하며, `Path.is_dir()`은 디렉터리에서 True를 반환합니다.",
                 1, (), True, "sentence_count:2", id="two-targets-one-assigned-sentence"),
    pytest.param("`Path.is_file()`은 일반 파일에서 True를 반환하며,", 1,
                 (("`Path.is_dir()`은 디렉터리에서 True를 반환합니다.", 2),), True, "sentence_scope",
                 id="one-sentence-crossing-two-content-units"),
])
def test_local_sentence_violation_is_not_overridden_by_approving_decisions(
    text: str, sentence_scope: int | None, following_parts: tuple[tuple[str, int | None], ...],
    complete: bool, missing: str,
):
    state = _partial_state(text, complete=complete, sentence_scope=sentence_scope, following_parts=following_parts)
    original = state["response"].result.content.model_copy(deep=True)
    decisions = DecisionsBoundary()

    state.update(make_post_synthesis_validation_node(False, decisions_client=decisions)(state))

    outcome = ResponseAssembler().assemble(response=state)
    assert outcome.status == "failed"
    assert state["response"].result.content != original
    assert missing in state["retry"].retrieval_feedback
    assert state["response"].result.actions == []
    assert decisions.requests == []


def test_two_supported_targets_keep_their_own_sentence_quotas():
    state = _partial_state(ONE_SENTENCE, complete=True, following_parts=((DIRECTORY_SENTENCE, 2),))
    original = state["response"].result.content.model_copy(deep=True)

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))

    assert ResponseAssembler().assemble(response=state).status == "completed"
    assert state["response"].result.content == original


def test_partial_answer_does_not_waive_a_separate_whole_body_quota():
    state = _partial_state(ONE_SENTENCE)
    contract = state["runtime"].request_contract.model_dump(mode="json")
    contract["answer"]["sentence_counts"].append({
        "scope": None, "minimum": 2, "maximum": 2, "mode": "required", "evidence_ids": ["request"],
    })
    state["runtime"].request_contract = RequestContract.model_validate(contract)

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))

    assert ResponseAssembler().assemble(response=state).status == "failed"
    assert "sentence_count:3" in state["retry"].retrieval_feedback


def test_independent_true_and_false_conditions_for_one_target_have_separate_quotas():
    state = _partial_state(ONE_SENTENCE, following_parts=((
        "`Path.is_file()`은 경로가 없거나 일반 파일이 아니면 `False`를 반환합니다.", 2,
    ),))
    request = "Python 공식 문서로 Path.is_file()의 True 반환 조건과 False 반환 조건을 각각 한 문장으로 설명해 줘."
    contract = state["runtime"].request_contract.model_dump(mode="json")
    contract["body"]["instruction"] = request
    contract["answer"]["content"][0]["description"] = "Explain when Path.is_file() returns True and False."
    contract["answer"]["sentence_counts"][1]["scope"] = "Path.is_file()이 False를 반환하는 경우"
    contract["evidence"][0]["quote"] = request
    state["runtime"].request_contract = RequestContract.model_validate(contract)
    state["runtime"].user_input = request
    state["messages"] = [HumanMessage(content=request, id="current")]
    state["planner"].output.tasks = state["planner"].output.tasks[:1]
    state["debug"].retrieval_diagnostics = state["debug"].retrieval_diagnostics[:1]
    state["retry"].terminal_failed_requirement_ids = []
    state["retry"].terminal_retrieval_reason = None
    state["response"].normal_evidence_missing_requirement_ids = []
    original = state["response"].result.content.model_copy(deep=True)

    state.update(make_post_synthesis_validation_node(False, decisions_client=DecisionsBoundary())(state))

    assert ResponseAssembler().assemble(response=state).status == "completed"
    assert state["response"].result.content == original


@pytest.mark.skipif(
    os.environ.get("RUN_LIVE_PARTIAL_ANSWER_SCOPE") != "1",
    reason="Set RUN_LIVE_PARTIAL_ANSWER_SCOPE=1 to call the configured Decisions provider.",
)
@pytest.mark.parametrize(("text", "expected", "complete"), [
    pytest.param(ONE_SENTENCE, "partial", False, id="supported-one-sentence"),
    pytest.param("`Path.is_file()`은 지정한 경로가 일반 파일을 가리키는 경우에 `True`를 돌려줍니다.",
                 "partial", False, id="supported-paraphrase-returns-true"),
    pytest.param("경로가 일반 파일을 가리키면 `Path.is_file()`의 반환값은 `True`입니다.",
                 "partial", False, id="supported-paraphrase-condition-first"),
    pytest.param("일반 파일을 가리키는 경로에 대해 `Path.is_file()`을 호출하면 `True`를 얻습니다.",
                 "partial", False, id="supported-paraphrase-call-result"),
    pytest.param("`Path.is_file()`에는 `follow_symlinks` 매개변수가 있습니다.",
                 "semantic_rejection", False, id="omitted-supported-fact"),
    pytest.param("경로가 일반 파일을 가리키면 `Path.is_file()`의 반환값은 `False`입니다.",
                 "semantic_rejection", False, id="contradicted-return-value"),
    pytest.param("경로가 디렉터리를 가리키면 `Path.is_file()`의 반환값은 `True`입니다.",
                 "semantic_rejection", False, id="wrong-supported-target"),
    pytest.param(TWO_SENTENCES, "sentence_count", False, id="supported-two-sentences-violate-per-target-count"),
    pytest.param(INTRODUCTION + " " + ONE_SENTENCE,
                 "sentence_count", False, id="introduction-repeats-per-target-condition"),
    pytest.param("`Path.is_file()`은 경로가 일반 파일을 가리킬 때 True를 반환하며, "
                 "`Path.is_dir()`은 경로가 디렉터리를 가리킬 때 True를 반환합니다.",
                 "sentence_count", True, id="complete-two-targets-merged-into-one-sentence"),
])
def test_live_answer_applies_sentence_count_to_supported_targets(
    text: str, expected: str, complete: bool,
    live_decisions_client: DecisionsClient, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = _partial_state(text, complete=complete)
    original = state["response"].result.content.model_copy(deep=True)
    evaluations: list[DecisionResult] = []
    evaluate = live_decisions_client.evaluate

    def record_evaluation(
        *, input: str, questions: list[dict[str, str]], stage: str,
        attempt: int = 1, reserve_calls: int = 0,
    ) -> DecisionResult:
        result = evaluate(input=input, questions=questions, stage=stage,
                          attempt=attempt, reserve_calls=reserve_calls)
        evaluations.append(result)
        return result

    monkeypatch.setattr(live_decisions_client, "evaluate", record_evaluation)
    with capture_llm_diagnostics():
        request_call_budget("post_synthesis_validation").max_calls = 3
        state.update(make_post_synthesis_validation_node(False, decisions_client=live_decisions_client)(state))

    outcome = ResponseAssembler().assemble(response=state)
    response = state["response"].result
    probabilities = {name: value for evaluation in evaluations for name, value in evaluation.probabilities.items()}
    refused_names = {name for evaluation in evaluations for name in evaluation.refused_names}
    diagnostics = "\n".join([state["retry"].retrieval_feedback, *state["debug"].validation_events])
    print({
        "candidate": text,
        "status": outcome.status,
        "probabilities": probabilities,
        "refused_names": sorted(refused_names),
        "validation_events": state["debug"].validation_events,
    })
    if expected == "partial":
        assert len(evaluations) == 3
        assert outcome.status == "partial", diagnostics
        assert outcome.problem.code == "evidence_insufficient"
        assert response.content == original
        assert response.semantic_validation is not None and response.semantic_validation.passed
        assert any("pathlib.Path.is_dir" in issue.message for issue in response.issues)
    elif expected == "sentence_count":
        assert evaluations == []
        assert response.content != original, diagnostics
        assert outcome.status == "failed"
        assert "sentence_count" in state["retry"].retrieval_feedback, diagnostics
    else:
        assert len(evaluations) == 3
        assert response.content != original, diagnostics
        assert outcome.status == "failed", diagnostics
        assert "semantic_" in state["retry"].retrieval_feedback, diagnostics
