"""Judge transport keeps exact answer evidence without resending whole sources."""
from __future__ import annotations

import copy
import json
import socket

import httpx
import pytest
import requests
from langchain_openai import ChatOpenAI

from src.app.transport_contracts import AgentResponse
from src.core.answer_schema import AnswerDocument, SemanticValidation, finalize_answer, semantic_evidence_fingerprint
from src.core.answer_schema.source_locations import SOURCE_LOCATION_PLACEHOLDER
from src.core.answer_schema.wire import SourcedAnswerPayload
from src.core.contracts.debug import DebugPayload
from src.core.contracts.outcome import TurnResult
from src.core.contracts.provenance import AnswerProvenance, AnswerSource
from src.core.documents import DocumentElement, SourceAnchor, TableCell, TableData, build_snapshot
from src.core.evidence import RetrievalScore, SearchHit, build_evidence
from src.core.llm_errors import make_problem
from src.eval.config_models import BenchmarkCase, BenchmarkConfig, CaseOracle
from src.eval.judge_llm import LLMJudge
from src.eval.online_runner.case_runner import run_online_benchmark
from src.eval.result_models import CaseResult
from src.infra.chunking import chunk_python_text
from tests.eval.test_release_eval_contract import _response_payload


@pytest.fixture
def provider(monkeypatch):
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(request.content) > 100_000:
            return httpx.Response(400, json={"error": {
                "message": "Judge input exceeds the provider context limit.",
                "type": "invalid_request_error", "param": "messages",
                "code": "context_length_exceeded",
            }})
        return httpx.Response(200, json={
            "id": "chatcmpl-evidence-budget", "object": "chat.completion", "created": 1,
            "model": payload["model"],
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant",
                "content": json.dumps({
                    "score": 0.9, "reason": "The selected source supports the answer.",
                    "subscores": {name: 0.9 for name in (
                        "answer_quality", "groundedness", "citation_traceability",
                        "tool_choice", "format_language",
                    )},
                }),
            }}],
            "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
        })

    def deny_network(*args, **kwargs):
        raise AssertionError("Judge budget tests must use the HTTP fake.")

    monkeypatch.setattr(socket.socket, "connect", deny_network)
    monkeypatch.setattr(socket.socket, "connect_ex", deny_network)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        monkeypatch.setattr(
            "src.eval.judge_llm.ChatOpenAI",
            lambda **kwargs: ChatOpenAI(**(kwargs | {
                "api_key": "test-key", "base_url": "https://judge-budget.invalid/v1",
                "http_client": client, "max_retries": 0, "use_responses_api": False,
            })),
        )
        yield requests


def _inputs(*, excerpt="선택한 근거는 α와 🧪 및 <|endoftext|> 문자를 보존합니다.", large_source=False,
            table=False, table_partial=True):
    omitted = "UNSELECTED_SOURCE_DO_NOT_SEND "
    text = omitted * (4_000 if large_source else 3) + excerpt + omitted * 3
    start = text.index(excerpt)
    snapshot = build_snapshot(
        source_uri="https://example.org/official/reference", title="Reference",
        media_type="text/plain", source_type="official", content=text,
        parser="test", parser_version="1",
    )
    anchors = [SourceAnchor(kind="web", start=start, end=start + len(excerpt))]
    if table:
        cells = [
            TableCell(cell_id="header", row=0, col=0, text="항목", is_header=True),
            TableCell(cell_id="selected", row=1, col=0, text=excerpt, anchors=anchors),
            TableCell(cell_id="unselected", row=2, col=0, text=omitted * 4_000),
        ]
        if not table_partial:
            cells = cells[:2]
        element = DocumentElement(
            element_id="table", kind="table", table=TableData(cells=cells),
            anchors=anchors,
        )
        evidence = build_evidence(snapshot=snapshot, element=element, cell_ids=["header", "selected"])
    else:
        element = DocumentElement(
            element_id="paragraph", kind="paragraph", text=text, anchors=anchors,
        )
        evidence = build_evidence(
            snapshot=snapshot, element=element, start=start, end=start + len(excerpt),
        )
    answer = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{
            "text": "요청한 내용을 근거에 따라 설명합니다.", "basis": "source", "refs": [evidence.id],
        }]}],
    }), [evidence])
    case = BenchmarkCase(
        case_id="judge-input-budget", category="docs_only", query="선택한 근거를 설명해 줘.",
        expected_tools=["tavily_search"], require_official_citation=True,
        oracle=CaseOracle(required_facts=["원문 선택 범위를 정확히 보존한다."]),
    )
    hit = SearchHit(
        evidence=evidence, rank=1, requirement_id="source-explanation",
        score=RetrievalScore(metric="rank", raw=1, direction="lower"),
    )
    return {
        "case": case, "tool_calls": ["tavily_search"], "response": answer,
        "turn_result": TurnResult(status="completed", response=answer, execution_id="request-budget"),
        "observed_hits": [hit, hit.model_copy(deep=True)],
        "evidence_scope": {
            "status": "complete", "errors": [], "verified_evidence": [evidence.model_dump(mode="json")],
        },
        "answer_provenance": AnswerProvenance(
            body_kind="compose", response_hash=answer.content_hash, evidence_packet=[evidence],
        ),
    }, evidence


def _wire_payload(request):
    return json.loads(next(message["content"] for message in request["messages"] if message["role"] == "user"))


@pytest.mark.parametrize(("final_status", "setup_status"), [
    ("completed", None), ("partial", None), ("needs_input", None),
    ("completed", "completed"), ("completed", "partial"), ("completed", "needs_input"),
])
def test_online_judge_uses_the_same_domain_turns_before_and_after_storage(
    provider, monkeypatch, tmp_path, final_status, setup_status,
):
    body = _response_payload()
    answer = SourcedAnswerPayload.model_validate(body["response"]).to_answer()
    case = BenchmarkCase(
        case_id="online-judge-contract", category="docs_only", query="pandas merge를 설명해 줘.",
        expected_tools=["tavily_search"], require_official_citation=True,
        setup_turns=["어떤 문서를 찾아야 해?"] if setup_status else [],
        setup_forbidden_tools=[[]] if setup_status else None,
        setup_expectations=[{
            "status": setup_status,
            **({"problem_code": "evidence_insufficient"} if setup_status == "partial" else {}),
        }] if setup_status else None,
    )
    fixtures = tmp_path / "cases.jsonl"
    fixtures.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    sent = []

    def respond(_adapter, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if request.method == "GET":
            response._content = json.dumps({"epoch": "session", "revision": 0, "files": []}).encode()
            return response
        sent.append(json.loads(request.body))
        status = setup_status if setup_status and len(sent) == 1 else final_status
        debug = DebugPayload.model_validate(body["debug"])
        if status == "needs_input":
            debug.answer_provenance = None
        envelope = AgentResponse(
            status=status, response=None if status == "needs_input" else answer,
            answer_id=None if status == "needs_input" else "answer-1",
            message="어떤 문서를 설명할까요?" if status == "needs_input" else "",
            missing_slots=["document"] if status == "needs_input" else [],
            problem=make_problem("evidence_insufficient", "retrieval") if status == "partial" else None,
            execution_id="req-1", trace=body["trace"], debug=debug,
            upload_manifest={"epoch": "session", "revision": 0, "files": []},
        )
        response.headers["Content-Type"] = "text/event-stream"
        response._content = ("event: final_response\ndata: " + envelope.model_dump_json() + "\n\n").encode()
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", respond)
    run_dir, results, _ = run_online_benchmark(
        fixtures_path=fixtures, endpoint="http://benchmark.invalid", config=BenchmarkConfig(),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke",
    )
    result, = results

    if final_status == "partial":
        assert result.judge_status == "not_run"
        assert result.judge_status_reason == "product_failure"
        assert result.judge_input_complete is None
        assert result.composite_quality_score is None
        assert provider == []
    else:
        assert result.judge_status == "succeeded", result.judge_input_issues
        assert result.judge_input_complete is True
        assert result.composite_quality_score is not None
        assert len(provider) == 1
        judge_payload = _wire_payload(provider[0])
        assert judge_payload["turn_result"]["status"] == final_status
        if setup_status:
            assert judge_payload["conversation"][0]["turn_result"]["status"] == setup_status
    if final_status == "completed":
        assert result.release_pass is True, result.gate_failures
    assert len(sent) == 1 + bool(setup_status)
    reloaded = CaseResult.model_validate_json((run_dir / "raw_results.jsonl").read_text(encoding="utf-8"))
    assert type(result.turn_result) is TurnResult
    assert result.turn_result.model_dump(mode="json") == reloaded.turn_result.model_dump(mode="json")
    for live, saved in zip(result.scenario_turns, reloaded.scenario_turns, strict=True):
        expected_status = setup_status if live.role == "setup" else final_status
        assert type(live.turn_result) is TurnResult
        assert live.turn_result.model_dump(mode="json") == saved.turn_result.model_dump(mode="json")
        assert live.turn_result.status == expected_status
        assert live.turn_result.execution_id == "req-1"
        assert live.turn_result.message == ("어떤 문서를 설명할까요?" if expected_status == "needs_input" else "")
        assert live.turn_result.missing_slots == (["document"] if expected_status == "needs_input" else [])
        if expected_status == "partial":
            assert live.turn_result.problem.code == "evidence_insufficient"
        else:
            assert live.turn_result.problem is None
        assert live.upload_manifest == saved.upload_manifest
        assert live.upload_manifest.epoch == "session"
        assert live.debug == saved.debug
        assert live.trace == saved.trace == body["trace"]
        if live.response is not None:
            assert live.turn_result.response == answer
            assert live.turn_result.answer_id == "answer-1"


def test_judge_input_rejects_unsupported_internal_values():
    inputs, _ = _inputs()
    inputs["planner_diagnostics"] = {"unexpected": object()}

    with pytest.raises(TypeError, match="Unsupported judge input value: object"):
        LLMJudge.build_case_payload(**inputs)


@pytest.mark.parametrize("assessed_explanation", [False, True])
def test_verified_source_location_is_observed_without_a_fake_grounding_probability(
    provider, monkeypatch, tmp_path, assessed_explanation,
):
    text = "value = 1\n"
    parsed = chunk_python_text(path="sample.py", text=text, source_content=text.encode(),
                               source_uri="upload:///sample.py", chunk_size=2000, chunk_overlap=0).parsed
    evidence = build_evidence(snapshot=parsed.snapshot, element=parsed.elements[0])
    answer = finalize_answer(AnswerDocument.model_validate({"blocks": [
        {"type": "paragraph", "content": [{"text": "value에는 1을 저장합니다.", "basis": "source", "refs": [evidence.id]}]},
        {"type": "paragraph", "content": [{"text": SOURCE_LOCATION_PLACEHOLDER, "basis": "source", "refs": [evidence.id],
          "source_location": {"ref": evidence.id, "start": 0, "end": len(text)}}]},
    ]}), [evidence])
    if assessed_explanation:
        answer.semantic_validation = SemanticValidation(
            content_hash=answer.content_hash, evidence_fingerprint=semantic_evidence_fingerprint(answer.citations),
            model="fixture", threshold=0.8, checks=[
                {"name": "request_fulfilled", "probability": 0.99},
                {"name": "grounded", "unit_id": "b0.content.0", "probability": 0.99},
            ],
        )
    debug = DebugPayload(
        tool_calls=["upload_search"], tool_call_count=1, llm_calls=[],
        observed_hits=[SearchHit(evidence=evidence, rank=1, score=RetrievalScore(
            metric="source_symbol", direction="higher")).model_dump(mode="json")],
        answer_provenance=AnswerProvenance(body_kind="compose", response_hash=answer.content_hash,
                                           evidence_packet=[evidence]),
        execution_evidence={"schema_version": 1, "execution_id": "location-execution", "status": "complete", "events": [
            {"sequence": index, "invocation_id": "search", "tool_name": "upload_search", "phase": phase}
            for index, phase in enumerate(("started", "succeeded"), 1)
        ]},
    )

    def respond(_adapter, request, **kwargs):
        response = requests.Response()
        response.status_code = 200
        response._content_consumed = True
        if request.method == "GET":
            response._content = json.dumps({"epoch": "session", "revision": 0, "files": []}).encode()
        else:
            envelope = AgentResponse(
                status="completed", response=answer, execution_id="location-execution", debug=debug,
                trace="Execution ID: location-execution",
                upload_manifest={"epoch": "session", "revision": 0, "files": []},
            )
            response.headers["Content-Type"] = "text/event-stream"
            response._content = ("event: final_response\ndata: " + envelope.model_dump_json() + "\n\n").encode()
        return response

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", respond)
    case = BenchmarkCase(case_id="location-statistics", category="rag_only",
                         query="value의 값과 원본 줄 위치를 설명해줘.", expected_tools=["upload_search"],
                         require_local_citation=True)
    fixture = tmp_path / "cases.jsonl"
    fixture.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    run_dir, results, _ = run_online_benchmark(
        fixtures_path=fixture, endpoint="http://location.invalid", config=BenchmarkConfig(),
        config_path=tmp_path / "config.toml", output_root=tmp_path / "runs", track="smoke",
    )
    result, = results
    assert result.judge_status == "succeeded", result.judge_input_issues
    assert result.unchecked_unit_count == (0 if assessed_explanation else 1)
    assert result.response.checks[1].support_status == "verified_location"
    assert result.response.content.blocks[1].content[0].text == "sample.py:1"
    assert len(provider) == 1
    judge_payload = _wire_payload(provider[0])
    assert judge_payload["content_stats"]["unchecked_unit_count"] == result.unchecked_unit_count
    report = result.response.semantic_validation
    assert report is None or all(check.unit_id != "b1.content.0" for check in report.checks)
    saved = CaseResult.model_validate_json((run_dir / "raw_results.jsonl").read_text(encoding="utf-8"))
    assert saved.unchecked_unit_count == result.unchecked_unit_count
    assert saved.response.checks[1].support_status == "verified_location"


def test_repeated_large_source_is_sent_once_as_exact_selected_evidence(provider):
    inputs, evidence = _inputs(large_source=True)
    inputs["turn_result"] = TurnResult(
        status="partial", response=inputs["response"], execution_id="request-budget",
        problem=make_problem("evidence_insufficient", "retrieval"),
    )
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(model_name="gpt-6-luna").score_case(**inputs)

    assert outcome.status == "succeeded", outcome.error
    assert len(provider) == 1
    payload = _wire_payload(provider[0])
    assert set(payload["evidence"]) == {evidence.id}
    record = payload["evidence"][evidence.id]
    assert record["id"] == evidence.id
    assert record["snapshot"] == evidence.snapshot.model_dump(mode="json")
    assert record["selection"] == evidence.selection.model_dump(mode="json")
    assert record["excerpt"] == evidence.excerpt
    assert record["element"]["element_id"] == evidence.element.element_id
    assert "text" not in record["element"]
    assert "table" not in record["element"]
    assert "UNSELECTED_SOURCE_DO_NOT_SEND" not in json.dumps(payload, ensure_ascii=False)
    assert payload["response"]["content"] == inputs["response"].content.model_dump(mode="json")
    assert payload["response"]["content_hash"] == inputs["response"].content_hash
    assert payload["case"]["oracle"] == inputs["case"].oracle.model_dump(mode="json")
    reference = {"evidence_id": evidence.id}
    assert payload["response"]["citations"][0]["evidence"] == reference
    assert payload["evidence_scope"]["verified_evidence"] == [reference]
    assert payload["answer_provenance"]["evidence_packet"] == [reference]
    assert [hit["evidence"] for hit in payload["observed_hits"]] == [reference, reference]
    assert record["anchors"] == [anchor.model_dump(mode="json") for anchor in evidence.element.anchors]
    assert payload["turn_result"]["status"] == "partial"
    assert payload["turn_result"]["problem"] == inputs["turn_result"].problem.model_dump(mode="json")
    assert payload["turn_result"]["response"] == {"response_content_hash": inputs["response"].content_hash}
    assert inputs == original
    assert outcome.input_stats["evidence_count"] == 1
    assert outcome.input_stats["estimated_input_tokens"] <= outcome.input_stats["max_input_tokens"]
    assert outcome.input_stats["output_token_limit"] == 4096
    assert provider[0].get("max_completion_tokens", provider[0].get("max_tokens")) == 4096


@pytest.mark.parametrize("table_partial", [True, False])
def test_selected_table_cells_and_excerpt_survive_without_unselected_cells(provider, table_partial):
    inputs, evidence = _inputs(table=True, table_partial=table_partial)
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(model_name="gpt-6-luna").score_case(**inputs)

    assert outcome.status == "succeeded", outcome.error
    payload = _wire_payload(provider[0])
    record = payload["evidence"][evidence.id]
    assert record["selection"] == evidence.selection.model_dump(mode="json")
    assert record["excerpt"] == evidence.excerpt
    assert record["is_partial"] is table_partial
    assert "UNSELECTED_SOURCE_DO_NOT_SEND" not in json.dumps(payload, ensure_ascii=False)
    assert inputs == original


@pytest.mark.parametrize("oversized_field", ["excerpt", "oracle"])
def test_required_input_over_budget_is_not_truncated_or_sent(provider, oversized_field):
    required = "필수 근거 α 中文 🧪 <|endoftext|> " * 2_000
    inputs, _ = _inputs(excerpt=required if oversized_field == "excerpt" else "짧은 필수 근거")
    if oversized_field == "oracle":
        inputs["case"].oracle.required_facts = [required]
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(
        model_name="gpt-6-luna", max_input_tokens=1_000, max_output_tokens=128,
    ).score_case(**inputs)

    assert outcome.status == "not_run"
    assert outcome.failure_kind == "input_budget_exceeded"
    assert outcome.score is None
    assert outcome.subscores is None
    assert outcome.input_stats["estimated_input_tokens"] > outcome.input_stats["max_input_tokens"]
    assert provider == []
    assert inputs == original


def test_conflicting_reference_identity_is_rejected_before_http(provider):
    inputs, _ = _inputs()
    conflicting = inputs["observed_hits"][0].model_dump(mode="json")
    original_text = conflicting["evidence"]["element"]["text"]
    conflicting["evidence"]["element"]["text"] = "X" + original_text[1:]
    inputs["observed_hits"] = [conflicting]
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(model_name="gpt-6-luna").score_case(**inputs)

    assert outcome.status == "not_run"
    assert outcome.failure_kind == "input_invalid"
    assert outcome.score is None
    assert provider == []
    assert inputs == original


def test_different_selections_of_one_source_remain_distinct(provider):
    inputs, first = _inputs(excerpt="첫 번째 사실과 두 번째 사실을 함께 설명합니다.")
    second = build_evidence(
        snapshot=first.snapshot, element=first.element,
        start=first.selection.start + 8, end=first.selection.end,
    )
    response = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{
            "text": "두 선택 범위를 설명합니다.", "basis": "source", "refs": [first.id, second.id],
        }]}],
    }), [first, second])
    inputs.update(
        response=response, turn_result=TurnResult(status="completed", response=response),
        observed_hits=[
            SearchHit(evidence=item, rank=index, score=RetrievalScore(metric="rank", raw=index, direction="lower"))
            for index, item in enumerate((first, second), 1)
        ],
        evidence_scope={"status": "complete", "errors": [], "verified_evidence": [
            item.model_dump(mode="json") for item in (first, second)
        ]},
        answer_provenance=AnswerProvenance(
            body_kind="compose", response_hash=response.content_hash, evidence_packet=[first, second],
        ),
    )
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(model_name="gpt-6-luna").score_case(**inputs)

    assert outcome.status == "succeeded", outcome.error
    payload = _wire_payload(provider[0])
    assert first.id != second.id
    assert set(payload["evidence"]) == {first.id, second.id}
    for item in (first, second):
        assert payload["evidence"][item.id]["selection"] == item.selection.model_dump(mode="json")
        assert payload["evidence"][item.id]["excerpt"] == item.excerpt
    assert payload["response"]["content"] == response.content.model_dump(mode="json")
    assert inputs == original


def test_transformed_answer_keeps_selected_parent_and_current_evidence_scope(provider):
    inputs, evidence = _inputs()
    previous = inputs["response"]
    prior_query = "선택한 근거를 설명해 줘."
    inputs["case"].setup_turns = [prior_query]
    inputs["case"].query = "이전 답변을 더 짧게 바꿔 줘."
    current = finalize_answer(AnswerDocument.model_validate({
        "blocks": [{"type": "paragraph", "content": [{
            "text": "근거를 짧게 설명합니다.", "basis": "source", "refs": [evidence.id],
        }]}],
    }), [evidence])
    old_unused = build_evidence(
        snapshot=evidence.snapshot, element=evidence.element, start=0, end=10,
    )
    inputs.update(
        response=current, turn_result=TurnResult(status="completed", response=current),
        tool_calls=[], observed_hits=[],
        conversation=[{
            "query": prior_query, "response": previous,
            "turn_result": TurnResult(status="completed", response=previous),
            "observed_hits": [SearchHit(
                evidence=old_unused, rank=1,
                score=RetrievalScore(metric="rank", raw=1, direction="lower"),
            )],
        }],
        answer_provenance=AnswerProvenance(
            body_kind="transform_answer", response_hash=current.content_hash,
            source=AnswerSource(ref="previous", response_hash=previous.content_hash, citation_ids=[evidence.id]),
            evidence_packet=[evidence],
        ),
    )
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(model_name="gpt-6-luna").score_case(**inputs)

    assert outcome.status == "succeeded", outcome.error
    payload = _wire_payload(provider[0])
    assert payload["conversation"][0]["query"] == prior_query
    assert payload["conversation"][0]["response"]["content"] == previous.content.model_dump(mode="json")
    assert payload["conversation"][0]["turn_result"]["response"] == {
        "response_content_hash": previous.content_hash,
    }
    assert payload["response"]["content"] == current.content.model_dump(mode="json")
    assert payload["answer_provenance"]["source"] == inputs["answer_provenance"].source.model_dump(mode="json")
    assert payload["evidence_scope"]["verified_evidence"] == [{"evidence_id": evidence.id}]
    assert payload["answer_provenance"]["evidence_packet"] == [{"evidence_id": evidence.id}]
    assert old_unused.id not in payload["answer_provenance"]["source"]["citation_ids"]
    assert inputs == original


def test_missing_input_clarification_does_not_require_a_response_document(provider):
    case = BenchmarkCase(
        case_id="missing-file", category="tool_action", query="첨부한 문서를 요약해 줘.",
        oracle=CaseOracle(expected_behaviors=["문서가 없으므로 업로드를 요청한다."]),
    )
    turn = TurnResult(
        status="needs_input", message="요약할 문서를 업로드해 주세요.",
        execution_id="clarification-request", missing_slots=["attachment"],
    )
    outcome = LLMJudge(model_name="gpt-6-luna").score_case(
        case=case, tool_calls=[], response=None, turn_result=turn,
    )

    assert outcome.status == "succeeded", outcome.error
    payload = _wire_payload(provider[0])
    assert payload["response"] is None
    assert payload["turn_result"] == turn.model_dump(mode="json")
    assert payload["evidence"] == {}
    assert payload["evidence_scope"] is None
    assert payload["answer_provenance"] is None
    assert payload["case"]["oracle"] == case.oracle.model_dump(mode="json")


def test_unavailable_tokenizer_uses_byte_bound_and_blocks_oversized_input(provider, monkeypatch):
    def unavailable(*args, **kwargs):
        raise OSError("Tokenizer vocabulary is unavailable.")

    monkeypatch.setattr("src.eval.judge_input.tiktoken.encoding_for_model", unavailable)
    monkeypatch.setattr("src.eval.judge_input.tiktoken.get_encoding", unavailable)
    inputs, _ = _inputs(excerpt="필수 근거 🧪 <|endoftext|> " * 100)
    original = copy.deepcopy(inputs)
    outcome = LLMJudge(
        model_name="gpt-6-luna", max_input_tokens=1_000, max_output_tokens=128,
    ).score_case(**inputs)

    assert outcome.status == "not_run"
    assert outcome.failure_kind == "input_budget_exceeded"
    assert outcome.score is None
    assert outcome.input_stats["tokenizer"] == "utf8_byte_upper_bound"
    assert outcome.input_stats["tokenizer_error"] == "OSError"
    assert outcome.input_stats["estimated_input_tokens"] > outcome.input_stats["max_input_tokens"]
    assert outcome.input_stats["estimated_input_tokens"] >= outcome.input_stats["payload_bytes"]
    assert provider == []
    assert inputs == original
