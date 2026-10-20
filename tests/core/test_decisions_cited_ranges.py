"""Public validation nodes trust semantic decisions and retain exact source scope."""
from __future__ import annotations

import json
from functools import partial

import httpx
import pytest
from openai import OpenAI

from src.core.answer_schema import finalize_answer, text_document
from src.core.contracts import DebugState, PlannerState, ResponseState, RetrievalDiagnostic, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.decisions import DecisionsClient
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.synthesis_fixtures import synthesis_excerpt_limits


make_pre_synthesis_validation_node = partial(make_pre_synthesis_validation_node, excerpt_limits=synthesis_excerpt_limits())


class PassingDecisions(DecisionsClient):
    """Exercise the production SDK/client against a provider HTTP boundary."""

    def __init__(self, probabilities=None, *, refused_names=()):
        self.requests = []
        self.probabilities = probabilities or {}
        self.refused_names = set(refused_names)
        sdk = OpenAI(
            api_key="test-key", base_url="https://provider.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._respond)), max_retries=0,
        )
        super().__init__(sdk)

    def _respond(self, request):
        assert request.url.path == "/v1/decisions"
        payload = json.loads(request.content)
        self.requests.append(payload)
        return httpx.Response(200, json={
            "model": self.model_name,
            "answers": [{"type": "refusal", "name": question["name"]}
                        if question["name"] in self.refused_names else
                        {"type": "predicate", "name": question["name"],
                         "probability": self.probabilities.get(question["name"], 0.99)}
                        for question in payload["questions"]],
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })


def _state(*, route="docs", source_text="None으로 주면 저장하는 항목 수에 상한을 두지 않습니다."):
    contract = RequestContract(request_id="cited-ranges")
    requirement = ({
        "library": "Python", "symbols": ["functools.lru_cache"], "aspects": ["maxsize"],
        "evidence_needs": ["parameters"], "match": "symbol",
    } if route == "docs" else {
        "file_ids": ["upload-1"], "match": "topic",
    })
    task = RetrievalTask(
        route=route, query="캐시가 항목을 무제한으로 보관하도록 설정하는 방법", k=2,
        requirement_id="cache_limit", requirement=requirement,
    )
    source = build_snapshot(
        source_uri="https://docs.python.org/3/library/functools.html" if route == "docs" else "upload://upload-1",
        title="functools.lru_cache", media_type="text/plain",
        source_type="official" if route == "docs" else "upload", content=source_text,
        parser="test", parser_version="1",
    )
    element = DocumentElement(
        element_id="cache", kind="paragraph", text=source_text,
        anchors=[SourceAnchor(kind="web" if route == "docs" else "text", start=0,
                              end=len(source_text), precision="exact")],
        metadata={"file_id": "upload-1"} if route == "upload" else {},
    )
    evidence = build_evidence(snapshot=source, element=element)
    hit = SearchHit(evidence=evidence, requirement_id=task.requirement_id, rank=1,
                    score=RetrievalScore(metric="test", direction="higher"))
    document = text_document("None을 지정하면 저장 항목 수를 제한하지 않습니다.", basis="source", refs=[evidence.id])
    return build_graph_state_input(
        user_input="캐시에 상한을 두지 않으려면 어떻게 설정하나요?", request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task])),
        retrieval=RetrievalState(hit_log=[hit]), retry=RetryState(max_retries=1, max_content_repairs=1),
        debug=DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
            route=route, requirement_id=task.requirement_id, status="success",
            answerability="partial", missing_requirements=["aspect:maxsize"], evidence_count=1,
        )]),
        response=ResponseState(
            result=finalize_answer(document, [evidence], retrieval_required=True),
            evidence_packet=[evidence], evidence_requirement_map={evidence.id: [task.requirement_id]},
            synthesis_attempt=1, request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


@pytest.mark.parametrize("factory", [make_pre_synthesis_validation_node, make_post_synthesis_validation_node])
@pytest.mark.parametrize("route", ["docs", "upload"])
def test_semantically_accepted_korean_paraphrase_is_not_rejected_by_literal_coverage(factory, route):
    state = _state(route=route)
    boundary = PassingDecisions()

    updates = factory(False, decisions_client=boundary)(state)

    assert boundary.requests
    assert updates["retry"].needs_retry is False
    assert updates["retry"].failed_requirement_ids == []
    if factory is make_post_synthesis_validation_node:
        assert updates["response"].kind == "answer"
        assert updates["response"].result.citations == state["response"].result.citations


@pytest.mark.parametrize("factory", [make_pre_synthesis_validation_node, make_post_synthesis_validation_node])
def test_decisions_cannot_override_explicit_upload_file_scope(factory):
    state = _state(route="upload")
    task = state["planner"].output.tasks[0]
    task = task.model_copy(update={"requirement": task.requirement.model_copy(update={"file_ids": ["upload-2"]})})
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task]))

    updates = factory(False, decisions_client=PassingDecisions())(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].failed_requirement_ids == ["cache_limit"]


def test_post_decisions_cannot_override_a_truncated_python_definition():
    source_text = "def cached(value):\n    return value + 1\n"
    state = _state(route="upload", source_text=source_text)
    task = state["planner"].output.tasks[0]
    task = task.model_copy(update={"requirement": task.requirement.model_copy(update={
        "match": "definition", "symbols": ["cached"], "aspects": [],
    })})
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task]))
    original = state["response"].evidence_packet[0]
    evidence = build_evidence(
        snapshot=original.snapshot, element=original.element.model_copy(update={"kind": "code"}),
        start=0, end=source_text.index("\n") + 1,
    )
    hit = state["retrieval"].hit_log[0].model_copy(update={"evidence": evidence})
    state["retrieval"] = RetrievalState(hit_log=[hit])
    document = text_document("cached 함수의 정의입니다.", basis="source", refs=[evidence.id])
    state["response"] = state["response"].model_copy(update={
        "result": finalize_answer(document, [evidence], retrieval_required=True),
        "evidence_packet": [evidence], "evidence_requirement_map": {evidence.id: [task.requirement_id]},
    })

    updates = make_post_synthesis_validation_node(False, decisions_client=PassingDecisions())(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].failed_requirement_ids == ["cache_limit"]


def test_post_requirement_batch_sees_only_actual_citations_and_records_requirement_check():
    state = _state()
    cited = state["response"].evidence_packet[0]
    uncited = _state(source_text="UNCITED_SOURCE contains additional required facts.")["response"].evidence_packet[0]
    state["response"] = state["response"].model_copy(update={
        "evidence_packet": [cited, uncited],
        "evidence_requirement_map": {cited.id: ["cache_limit"], uncited.id: ["cache_limit"]},
    })
    boundary = PassingDecisions()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert len(boundary.requests) == 1
    request = boundary.requests[0]
    payload = json.loads(request["input"])
    assert {question["name"] for question in request["questions"]} == {
        "grounded_0", "request_fulfilled", "requirement_0",
    }
    assert [source["id"] for source in payload["requirements"][0]["sources"]] == [cited.id]
    assert "UNCITED_SOURCE" not in request["input"]
    report = updates["response"].result.semantic_validation
    assert report.passed
    assert next(check for check in report.checks if check.name == "requirement:cache_limit").probability == 0.99


@pytest.mark.parametrize("refused", [False, True])
def test_insufficient_actual_citations_trigger_requirement_repair_even_when_body_checks_pass(refused):
    state = _state()
    boundary = PassingDecisions(
        {} if refused else {"requirement_0": 0.1}, refused_names={"requirement_0"} if refused else (),
    )

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert len(boundary.requests) == 1
    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert updates["retry"].failed_requirement_ids == ["cache_limit"]
    assert "semantic_cited_requirement:cache_limit" in updates["retry"].retrieval_feedback


def test_terminal_requirement_gap_preserves_grounded_partial_answer_policy():
    state = _state()
    state["retry"] = state["retry"].model_copy(update={
        "terminal_failed_requirement_ids": ["cache_limit"], "terminal_retrieval_reason": "no_evidence",
    })
    boundary = PassingDecisions()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert json.loads(boundary.requests[0]["input"])["requirements"] == []
    assert updates["retry"].needs_retry is False
    assert updates["response"].kind == "failure"
    assert updates["response"].problem.code == "evidence_insufficient"
    assert updates["response"].result.content == state["response"].result.content
    assert "retrieved_evidence_incomplete" in {issue.code for issue in updates["response"].result.issues}
