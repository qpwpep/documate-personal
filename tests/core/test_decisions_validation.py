"""Decisions judgments control retrieval and answer recovery through public nodes."""
from __future__ import annotations

from functools import partial
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from openai import OpenAI

from src.core.answer_schema import AnswerDocument, export_answer_text, finalize_answer, text_document
from src.core.contracts import DebugState, PlannerState, ResponseState, RetrievalDiagnostic, RetrievalState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.documents import DocumentElement, SourceAnchor, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.llm_errors import LLMCallError
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.infra.decisions import DecisionsClient
from src.runtime.nodes.validation import make_post_synthesis_validation_node, make_pre_synthesis_validation_node
from tests.synthesis_fixtures import synthesis_excerpt_limits


make_pre_synthesis_validation_node = partial(make_pre_synthesis_validation_node, excerpt_limits=synthesis_excerpt_limits())

class DecisionsBoundary(DecisionsClient):
    """Keep the SDK and Decisions client real while replacing provider HTTP."""

    def __init__(self, probabilities=None, *, refused_names=(), error_status=None):
        self.probabilities = probabilities or {}
        self.refused_names = set(refused_names)
        self.error_status = error_status
        self.requests = []
        sdk = OpenAI(
            api_key="test-key", base_url="https://provider.test/v1",
            http_client=httpx.Client(transport=httpx.MockTransport(self._respond)), max_retries=0,
        )
        super().__init__(sdk)

    def evaluate(self, *, input, questions, stage, attempt=1, reserve_calls=0):
        self._stage, self._attempt = stage, attempt
        return super().evaluate(input=input, questions=questions, stage=stage, attempt=attempt,
                                reserve_calls=reserve_calls)

    def _respond(self, request):
        assert request.method == "POST"
        assert request.url.path == "/v1/decisions"
        payload = json.loads(request.content)
        self.requests.append({**payload, "stage": self._stage, "attempt": self._attempt})
        if self.error_status is not None:
            return httpx.Response(self.error_status, json={"error": {
                "message": "Temporary provider failure", "type": "server_error", "code": "server_error",
            }})
        return httpx.Response(200, headers={"x-request-id": "req_test_decisions_validation"}, json={
            "model": self.model_name,
            "answers": [{"type": "refusal", "name": question["name"]}
                        if question["name"] in self.refused_names else
                        {"type": "predicate", "name": question["name"],
                         "probability": self.probabilities.get(question["name"], 0.99)}
                        for question in payload["questions"]],
            "usage": {"input_tokens": 32, "output_tokens": 0, "total_tokens": 32},
        })


def _evidence(source_text, *, subject="pathlib.Path.exists", source_uri=None):
    source = build_snapshot(
        source_uri=source_uri or f"https://docs.python.org/3/library/pathlib.html#{subject}",
        title=subject, media_type="text/plain", source_type="official",
        content=source_text, parser="test", parser_version="1",
    )
    element = DocumentElement(
        element_id=subject, kind="paragraph", text=source_text,
        anchors=[SourceAnchor(kind="web", start=0, end=len(source_text), precision="exact")],
    )
    return build_evidence(snapshot=source, element=element)


def _state(*, retrieval=True, answer="Path.exists()는 파일이나 디렉터리가 존재하면 True를 반환합니다.", repairs=1):
    contract = RequestContract(request_id="decisions-validation")
    task = RetrievalTask(
        route="docs", query="pathlib.Path.exists behavior", k=2, requirement_id="path_exists",
        requirement={"library": "Python", "symbols": ["pathlib.Path.exists"], "match": "symbol"},
    )
    evidence = _evidence("Path.exists() returns True if the path points to an existing file or directory.")
    hit = SearchHit(
        evidence=evidence, requirement_id=task.requirement_id, rank=1,
        score=RetrievalScore(metric="test", raw=0.99, normalized=0.99, direction="higher"),
    )
    packet = [evidence] if retrieval else []
    document = text_document(answer, basis="source", refs=[evidence.id]) if retrieval else text_document(answer)
    return build_graph_state_input(
        user_input="Path.exists()의 반환값을 설명해 주세요." if retrieval else "간단한 인사말을 작성해 주세요.",
        request_contract=contract,
        planner=PlannerState(output=PlannerOutput(use_retrieval=retrieval, tasks=[task] if retrieval else [])),
        retrieval=RetrievalState(hit_log=[hit.model_dump(mode="json")] if retrieval else []),
        debug=DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
            route="docs", requirement_id=task.requirement_id, status="success", answerability="covered", evidence_count=1,
        )] if retrieval else []),
        retry=RetryState(max_retries=1, max_content_repairs=repairs),
        response=ResponseState(
            result=finalize_answer(document, packet, retrieval_required=retrieval),
            evidence_packet=packet,
            evidence_requirement_map={evidence.id: [task.requirement_id]} if retrieval else {},
            synthesis_attempt=1, request_id=contract.request_id, contract_revision=contract.revision,
        ),
    )


@pytest.mark.parametrize("probability, needs_retry", [(0.79, True), (0.80, False)])
def test_pre_validation_uses_decisions_probability_for_covered_evidence(probability, needs_retry):
    """검색 진단이 covered여도 Decisions 임계값에 따라 재검색하거나 진행한다."""
    state = _state()
    boundary = DecisionsBoundary({"requirement_0": probability})

    updates = make_pre_synthesis_validation_node(False, decisions_client=boundary, pass_threshold=0.8)(state)

    assert updates["retry"].needs_retry is needs_retry
    assert updates["retry"].failed_requirement_ids == (["path_exists"] if needs_retry else [])
    if needs_retry:
        assert updates["retry"].retry_reason == "no_evidence"
        assert updates["retry"].retry_scope == "refresh_routes"
        assert updates["retry"].failed_routes == ["docs"]
    request = boundary.requests[0]
    assert request["stage"] == "pre_synthesis_validation"
    assert state["runtime"].user_input in request["input"]
    assert state["response"].evidence_packet[0].excerpt in request["input"]


@pytest.mark.parametrize("max_retries, remaining_failure", [
    (1, None),
    (0, "requirement_0"),
    (1, "requirement_0"),
    (1, "request_ready"),
    (0, "tool_error"),
    (1, "tool_error"),
], ids=[
    "covered", "semantic-gap-terminal", "semantic-gap-retry",
    "request-not-ready", "provider-error-terminal", "provider-error-retry",
])
def test_pre_validation_recomputes_static_failures_after_rejecting_a_source(max_retries, remaining_failure):
    """Rejected API headings cannot leave stale failures or erase independent failures."""
    task = RetrievalTask(
        route="docs", query="mean", k=2, requirement_id="mean_parameter",
        requirement={"library": "numpy", "evidence_needs": ["parameters"]},
    )
    good = _evidence(
        "# numpy.var\nnumpy.var(a, mean=None)\n\nParameters\n"
        "mean: Supplies the mean value used to compute the variance.\n",
        subject="numpy.var", source_uri="https://numpy.org/doc/stable/reference/generated/numpy.var.html",
    )
    rejected = _evidence(
        "# numpy.mean\n", subject="numpy.mean",
        source_uri="https://numpy.org/doc/stable/reference/generated/numpy.mean.html",
    )
    state = _state()
    state["runtime"] = state["runtime"].model_copy(update={"user_input": "numpy.var의 mean 매개변수를 설명해 주세요."})
    state["planner"] = PlannerState(output=PlannerOutput(use_retrieval=True, tasks=[task]))
    state["retrieval"] = RetrievalState(hit_log=[
        SearchHit(evidence=evidence, requirement_id=task.requirement_id, rank=index + 1,
                  score=RetrievalScore(metric="rank", raw=index + 1, direction="lower"))
        for index, evidence in enumerate([good, rejected])
    ])
    state["debug"] = DebugState(retrieval_diagnostics=[RetrievalDiagnostic(
        route="docs", requirement_id=task.requirement_id, evidence_count=2,
        status="error" if remaining_failure == "tool_error" else "success", answerability="partial",
        error_code="RETRIEVAL_DOCS_FAILED" if remaining_failure == "tool_error" else None,
    )])
    state["retry"] = RetryState(max_retries=max_retries)
    state["response"] = ResponseState()
    probabilities = {"evidence_1": 0.1}
    if remaining_failure in {"requirement_0", "request_ready"}:
        probabilities[remaining_failure] = 0.1
    boundary = DecisionsBoundary(probabilities)

    updates = make_pre_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert [verdict.status for verdict in updates["retrieval"].evidence_verdicts] == ["passed", "failed"]
    requirements = json.loads(boundary.requests[-1]["input"])["requirements"]
    assert [source["id"] for source in requirements[0]["sources"]] == [good.id]
    retry = updates["retry"]
    if remaining_failure in {"requirement_0", "tool_error"}:
        assert retry.needs_retry is bool(max_retries)
        assert retry.retry_reason == ("tool_error" if remaining_failure == "tool_error" else "no_evidence")
        assert retry.failed_requirement_ids == [task.requirement_id]
        assert retry.terminal_failed_requirement_ids == ([] if max_retries else [task.requirement_id])
        assert [hit.evidence.id for hit in retry.preserved_hits] == [good.id]
    else:
        assert retry.needs_retry is False
        assert retry.attempt == 0
        assert retry.failed_requirement_ids == []
        assert retry.failed_routes == []
        assert retry.terminal_failed_requirement_ids == []
        assert retry.terminal_retrieval_reason is None
        if remaining_failure == "request_ready":
            assert retry.retry_reason == "missing_content"
            assert updates["response"].kind == "failure"
        else:
            assert retry.retry_reason is None
            assert "response" not in updates


def test_pre_validation_checks_readiness_without_retrieval():
    """검색 없는 요청도 Decisions 준비 판정을 거쳐 부적합한 요청의 합성을 막는다."""
    state = _state(retrieval=False, answer="안녕하세요.")
    boundary = DecisionsBoundary({"request_ready": 0.2})

    updates = make_pre_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert boundary.requests
    assert updates["retry"].needs_retry is False
    assert updates["retry"].retry_reason == "missing_content"
    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None


def test_request_readiness_refusal_stops_even_when_a_requirement_lacks_coverage():
    """A refused whole request cannot become a partial body through an unrelated source gap."""
    from src.runtime.make_graph import _pre_synthesis_route

    state = _state()
    accepted_hit = state["retrieval"].hit_log[0]
    state["response"] = ResponseState()
    state["retry"] = RetryState(max_retries=1)
    boundary = DecisionsBoundary({"requirement_0": 0.1}, refused_names={"request_ready"})

    state.update(make_pre_synthesis_validation_node(False, decisions_client=boundary)(state))

    assert state["retry"].needs_retry is False
    assert state["retry"].attempt == 0
    assert state["response"].kind == "failure"
    assert state["response"].problem.code == "model_refusal"
    assert _pre_synthesis_route(state)[0] == "action_postprocess"
    verdict = state["retrieval"].evidence_verdicts[0]
    assert (verdict.requirement_id, verdict.evidence_id, verdict.status) == (
        accepted_hit.requirement_id, accepted_hit.evidence.id, "passed",
    )


@pytest.mark.parametrize("factory", [make_pre_synthesis_validation_node, make_post_synthesis_validation_node])
@pytest.mark.parametrize("remembered_in", ["messages", "summary"])
def test_conversation_followups_send_the_user_facts_used_by_synthesis(factory, remembered_in):
    """검색 없는 후속 질문도 합성과 같은 대화 사실로 준비 가능성과 답변을 검증한다."""
    request = "제가 앞서 알려 준 이름이 뭐였죠?"
    state = _state(retrieval=False, answer="알려 주신 이름은 민수입니다.")
    contract = state["runtime"].request_contract.model_copy(update={
        "body": RequestContract(body={"kind": "compose", "instruction": "대화에서 사용자가 알려 준 이름을 답한다."}).body,
    })
    history = [
        HumanMessage(content="제 이름은 민수입니다."),
        AIMessage(content="반갑습니다."),
    ] if remembered_in == "messages" else []
    summary = "사용자가 자신의 이름을 민수라고 알려 주었다." if remembered_in == "summary" else None
    state["messages"] = [*history, HumanMessage(content=request)]
    state["runtime"] = state["runtime"].model_copy(update={
        "user_input": request, "request_contract": contract, "memory_summary": summary,
    })
    boundary = DecisionsBoundary()

    updates = factory(False, decisions_client=boundary)(state)

    context = json.loads(boundary.requests[0]["input"])["conversation_context"]
    assert context["memory_summary"] == summary
    assert context["messages"] == [
        *([{"role": "user", "content": "제 이름은 민수입니다."},
           {"role": "assistant", "content": "반갑습니다."}] if history else []),
        {"role": "user", "content": request},
    ]
    assert "민수" in json.dumps(context, ensure_ascii=False)
    assert updates["retry"].needs_retry is False


@pytest.mark.parametrize("factory", [make_pre_synthesis_validation_node, make_post_synthesis_validation_node])
def test_validation_conversation_uses_the_synthesis_turn_limit_and_excludes_tools(factory):
    """검증에 대화 전체나 검색 도구 메시지를 추가하지 않고 합성의 대화 범위를 유지한다."""
    from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages

    state = _state(retrieval=False, answer="알려 주신 이름은 민수입니다.")
    request = "제가 앞서 알려 준 이름이 뭐였죠?"
    state["messages"] = [
        HumanMessage(content="이미 대화 범위에서 제외된 발언"),
        AIMessage(content="이미 대화 범위에서 제외된 답변"),
        HumanMessage(content="제 이름은 민수입니다."),
        AIMessage(content="반갑습니다."),
        HumanMessage(content=request),
        ToolMessage(content="현재 검색 도구의 원문", tool_call_id="unused-search"),
    ]
    state["runtime"] = state["runtime"].model_copy(update={"user_input": request})
    boundary = DecisionsBoundary()

    factory(False, decisions_client=boundary, max_turns=1)(state)

    context = json.loads(boundary.requests[0]["input"])["conversation_context"]
    messages, _, _ = build_synthesis_messages(
        state=state, request_contract=state["runtime"].request_contract,
        action_rules=[], evidence_packet=[], attempt=1, max_turns=1,
        requirement_ids_by_evidence={}, reference_aliases={}, retrieval_required=False,
    )
    synthesis_dialogue = [message.content for message in messages if isinstance(message, (HumanMessage, AIMessage))]
    assert [message["content"] for message in context["messages"]] == synthesis_dialogue
    assert synthesis_dialogue == ["제 이름은 민수입니다.", "반갑습니다.", request]
    assert "이미 대화 범위에서 제외" not in json.dumps(context, ensure_ascii=False)
    assert "현재 검색 도구의 원문" not in json.dumps(context, ensure_ascii=False)


@pytest.mark.parametrize("contract", [
    RequestContract(body={"kind": "acknowledge"}),
    RequestContract(relation="cancel", target_request_id="pending-request"),
], ids=["acknowledge", "cancel"])
def test_bodyless_instructions_preserve_the_acknowledgment_and_cancellation_paths(contract):
    """본문 없는 확인·취소 지시를 답변 준비 확률로 차단하거나 합성 재시도하지 않는다."""
    state = _state(retrieval=False)
    state["runtime"] = state["runtime"].model_copy(update={"request_contract": contract})
    state["retry"] = RetryState(
        needs_retry=True, retry_reason="missing_content", failed_routes=["docs"],
        failed_requirement_ids=["previous-requirement"], retrieval_feedback="Previous failure",
    )
    boundary = DecisionsBoundary({"request_ready": 0.1})

    updates = make_pre_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert boundary.requests == []
    assert "response" not in updates
    assert updates["retry"].needs_retry is False
    assert updates["retry"].retry_reason is None
    assert updates["retry"].failed_routes == []
    assert updates["retry"].failed_requirement_ids == []
    assert updates["retry"].retrieval_feedback == ""


def test_post_validation_rejects_semantically_unsupported_text_with_valid_references():
    """유효한 출처 ID만으로 통과시키지 않고 의미적 근거 부족 시 같은 자료로 재합성한다."""
    claim = "Path.exists()는 존재하지 않는 경로를 자동으로 생성합니다."
    state = _state(answer=claim)
    boundary = DecisionsBoundary({"grounded_0": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_reason == "missing_content"
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert updates["retry"].content_repair_attempt == 1
    assert updates["retry"].preserved_hits[0].evidence == state["response"].evidence_packet[0]
    request = boundary.requests[0]
    assert request["stage"] == "post_synthesis_validation"
    assert claim in request["input"]
    assert state["response"].evidence_packet[0].excerpt in request["input"]


def test_post_validation_repairs_unfulfilled_request_without_retrieval():
    """형식 오류가 없는 일반 답변도 요청 충족 확률이 낮으면 한 번 재합성한다."""
    state = _state(retrieval=False, answer="요청과 관계없는 문장입니다.")
    boundary = DecisionsBoundary({"request_fulfilled": 0.15})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_reason == "missing_content"
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert updates["retry"].failed_routes == []


def test_post_validation_repairs_only_a_refused_unit_and_preserves_accepted_sources():
    state = _state()
    response = state["response"]
    refused_text = "Path.exists()에 관한 검증되지 않은 추가 주장입니다."
    document = AnswerDocument(blocks=[
        *response.result.content.blocks,
        *text_document(refused_text, basis="source", refs=[response.evidence_packet[0].id]).blocks,
    ])
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, response.evidence_packet, retrieval_required=True),
    })
    boundary = DecisionsBoundary(refused_names={"grounded_1"})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert updates["retry"].content_repair_attempt == 1
    assert updates["retry"].preserved_hits[0].evidence == response.evidence_packet[0]
    assert "b1.content.0" in updates["retry"].retrieval_feedback
    assert "b0.content.0" not in updates["retry"].retrieval_feedback


def test_post_validation_filters_a_refused_unit_then_checks_the_retained_body():
    state = _state(repairs=0)
    response = state["response"]
    refused_text = "Path.exists()에 관한 검증되지 않은 추가 주장입니다."
    document = AnswerDocument(blocks=[
        *response.result.content.blocks,
        *text_document(refused_text, basis="source", refs=[response.evidence_packet[0].id]).blocks,
    ])
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, response.evidence_packet, retrieval_required=True),
    })
    boundary = DecisionsBoundary(refused_names={"grounded_1"})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert len(boundary.requests) == 2
    assert refused_text in boundary.requests[0]["input"]
    assert refused_text not in boundary.requests[1]["input"]
    assert updates["response"].kind == "answer"
    assert updates["response"].result.content == response.result.content
    assert updates["response"].result.citations == response.result.citations
    assert updates["response"].result.semantic_validation.passed


def test_post_validation_request_fulfillment_refusal_uses_body_repair_without_research():
    state = _state()
    boundary = DecisionsBoundary(refused_names={"request_fulfilled"})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is True
    assert updates["retry"].retry_scope == "reuse_hits_resynthesize"
    assert updates["retry"].content_repair_attempt == 1
    assert updates["retry"].failed_routes == []


def test_post_validation_preserves_accepted_answer_and_citations():
    state = _state()
    boundary = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert boundary.requests
    assert updates["retry"].needs_retry is False
    assert updates["response"].kind == "answer"
    assert updates["response"].problem is None
    assert updates["response"].result.content == state["response"].result.content
    assert updates["response"].result.citations == state["response"].result.citations
    report = updates["response"].result.semantic_validation
    assert report is not None and report.passed
    assert report.model == boundary.model_name
    assert report.content_hash == updates["response"].result.content_hash


def test_exhausted_post_validation_does_not_publish_rejected_claim():
    """재합성 예산이 없으면 Decisions가 거절한 주장을 최종 답변에 남기지 않는다."""
    claim = "Path.exists()는 존재하지 않는 경로를 자동으로 생성합니다."
    state = _state(answer=claim, repairs=0)
    boundary = DecisionsBoundary({"grounded_0": 0.1})

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert updates["retry"].needs_retry is False
    assert claim not in export_answer_text(updates["response"].result)
    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None
    assert state["response"].evidence_packet[0].excerpt in export_answer_text(updates["response"].result)


def test_filtered_answer_must_pass_decisions_again_before_becoming_an_answer():
    """주장을 제거한 본문이 요청을 충족하지 못하면 이전 판정으로 답변을 승인하지 않는다."""
    state = _state(repairs=0)
    response = state["response"]
    invalid_claim = "Path.exists()는 존재하지 않는 경로를 자동으로 생성합니다."
    invalid_document = text_document(invalid_claim, basis="source", refs=[response.evidence_packet[0].id])
    document = AnswerDocument(blocks=[*response.result.content.blocks, *invalid_document.blocks])
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, response.evidence_packet, retrieval_required=True),
    })

    class SequentialDecisions(DecisionsBoundary):
        def evaluate(self, **kwargs):
            self.probabilities = ({"grounded_1": 0.1} if not self.requests else {"request_fulfilled": 0.1})
            return super().evaluate(**kwargs)

    boundary = SequentialDecisions()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert len(boundary.requests) == 2
    assert invalid_claim in boundary.requests[0]["input"]
    assert invalid_claim not in boundary.requests[1]["input"]
    result = updates["response"].result
    assert invalid_claim not in export_answer_text(result)
    assert updates["retry"].needs_retry is False
    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None
    assert result.semantic_validation is None
    assert response.evidence_packet[0].excerpt in export_answer_text(result)
    assert any("revision=filtered" in event and "request_fulfilled" in event
               for event in updates["debug"].validation_events)


def test_second_grounding_rejection_removes_the_retained_claim_too():
    """필터된 본문의 근거까지 재검증에서 거절되면 두 주장을 모두 원문 발췌로 대체한다."""
    retained_claim = "Path.exists()는 파일 또는 디렉터리의 존재를 확인합니다."
    rejected_claim = "Path.exists()는 존재하지 않는 경로를 자동으로 생성합니다."
    state = _state(answer=retained_claim, repairs=0)
    response = state["response"]
    rejected_document = text_document(rejected_claim, basis="source", refs=[response.evidence_packet[0].id])
    document = AnswerDocument(blocks=[*response.result.content.blocks, *rejected_document.blocks])
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, response.evidence_packet, retrieval_required=True),
    })

    class SequentialDecisions(DecisionsBoundary):
        def evaluate(self, **kwargs):
            self.probabilities = ({"grounded_1": 0.1} if not self.requests else {"grounded_0": 0.1})
            return super().evaluate(**kwargs)

    boundary = SequentialDecisions()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    assert len(boundary.requests) == 2
    final_text = export_answer_text(updates["response"].result)
    assert retained_claim not in final_text
    assert rejected_claim not in final_text
    assert response.evidence_packet[0].excerpt in final_text
    assert updates["response"].kind == "failure"
    assert updates["response"].problem is not None
    assert any("revision=filtered" in event and "grounded_0" in event and "0.1" in event
               for event in updates["debug"].validation_events)


def test_post_validation_keeps_each_units_cited_sources_separate():
    """다른 문장의 출처를 빌려 근거를 판정하지 않도록 각 문장에 자체 인용만 보낸다."""
    state = _state()
    response = state["response"]
    second_evidence = _evidence("Path.is_file() returns True if the path is a regular file.", subject="pathlib.Path.is_file")
    second_document = text_document("Path.is_file()은 일반 파일인지 확인합니다.", basis="source", refs=[second_evidence.id])
    packet = [*response.evidence_packet, second_evidence]
    document = AnswerDocument(blocks=[*response.result.content.blocks, *second_document.blocks])
    state["response"] = response.model_copy(update={
        "result": finalize_answer(document, packet, retrieval_required=True), "evidence_packet": packet,
    })
    boundary = DecisionsBoundary()

    updates = make_post_synthesis_validation_node(False, decisions_client=boundary)(state)

    units = json.loads(boundary.requests[0]["input"])["units"]
    assert len(units) == 2
    assert [source["id"] for source in units[0]["cited_sources"]] == [packet[0].id]
    assert [source["id"] for source in units[1]["cited_sources"]] == [packet[1].id]
    assert updates["response"].kind == "answer"


def _previous_answer_state(body_kind, *, matching_revision=True):
    previous = _state(answer="이전 답변: Path.exists()는 파일 또는 디렉터리의 존재를 확인합니다.")["response"].result
    state = _state(retrieval=False, answer="현재 답변을 준비합니다.")
    body = {
        "kind": body_kind,
        "source": {"ref": "previous", "response_hash": previous.content_hash if matching_revision else "0" * 64},
    }
    if body_kind == "transform_answer":
        body["instruction"] = "이전 답변을 한 문장으로 요약해 주세요."
    contract = RequestContract(request_id="decisions-validation", body=body)
    state["runtime"] = state["runtime"].model_copy(update={
        "user_input": "이전 답변을 그대로 다시 보여 주세요." if body_kind == "copy_answer" else body["instruction"],
        "previous_response": previous, "request_contract": contract,
    })
    if body_kind == "copy_answer" and matching_revision:
        state["response"] = state["response"].model_copy(update={
            "result": previous,
            "evidence_packet": [citation.evidence for citation in previous.citations],
        })
    return state, previous


@pytest.mark.parametrize("factory", [make_pre_synthesis_validation_node, make_post_synthesis_validation_node])
@pytest.mark.parametrize("body_kind", ["copy_answer", "transform_answer"])
def test_answer_followups_send_bound_original_content_and_cited_sources(factory, body_kind):
    """이전 답변의 복사·변환 검증은 해시뿐 아니라 결합된 원문과 인용 근거를 받는다."""
    state, previous = _previous_answer_state(body_kind)
    boundary = DecisionsBoundary()

    factory(False, decisions_client=boundary)(state)

    bound = json.loads(boundary.requests[0]["input"])["bound_source_response"]
    assert bound["content_hash"] == previous.content_hash
    assert bound["content"] == previous.content.model_dump(mode="json")
    for source, citation in zip(bound["sources"], previous.citations, strict=True):
        assert {key: source[key] for key in ("id", "route", "title", "uri", "excerpt")} == {
            "id": citation.evidence.id,
            "route": citation.evidence.route,
            "title": citation.evidence.snapshot.title,
            "uri": citation.evidence.snapshot.source_uri,
            "excerpt": citation.evidence.excerpt,
        }


@pytest.mark.parametrize("factory", [make_pre_synthesis_validation_node, make_post_synthesis_validation_node])
@pytest.mark.parametrize("body_kind", ["copy_answer", "transform_answer"])
def test_answer_followups_do_not_send_a_different_previous_revision(factory, body_kind):
    """결합된 해시와 다른 원본은 API 호출 전에 거절해 본문·출처를 전송하지 않는다."""
    state, _previous = _previous_answer_state(body_kind, matching_revision=False)
    boundary = DecisionsBoundary()

    with pytest.raises(LLMCallError) as raised:
        factory(False, decisions_client=boundary)(state)

    assert raised.value.problem.code == "evidence_insufficient"
    assert boundary.requests == []


@pytest.mark.parametrize("factory, stage", [
    (make_pre_synthesis_validation_node, "pre_synthesis_validation"),
    (make_post_synthesis_validation_node, "post_synthesis_validation"),
])
def test_decisions_failure_cannot_silently_pass_validation(factory, stage):
    state = _state()
    boundary = DecisionsBoundary(error_status=503)

    with pytest.raises(LLMCallError) as raised:
        factory(False, decisions_client=boundary)(state)

    assert raised.value.problem.code == "provider_unavailable"
    assert raised.value.problem.stage == stage
