"""Physical source positions survive real upload, model, validation and HTTP delivery."""
from __future__ import annotations

import json
import socket
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from openai import OpenAI

from src.app.web.app import create_app
from src.core.answer_schema import AnswerDocument, AnswerResponse, export_answer_text
from src.core.planner_schema import InitialPlannerOutput
from src.core.request_contracts import WireRequestContract
from src.infra.settings import APP_ENV_SPEC_BY_NAME, AppSettings
from tests.web.test_multi_upload_api import context, final_response, staged, sync


SAMPLE = Path(__file__).parents[2] / "data/benchmarks/fixtures/uploads/sample_data_analysis.py"
CALCULATION = 'grouped = all_sales.groupby("region", as_index=False)["amount"].sum()'
QUESTION = "첨부한 sample_data_analysis.py에서 지역별 amount 합계를 계산하는 코드 한 줄과 원본 파일의 줄 번호를 알려줘."
INLINE_SOURCE = b"x = 1\ny = 2\n"
INLINE_QUESTION = "첨부한 two_lines.py의 1행과 2행 원문을 각각 인라인 코드로 표시하고 실제 줄 번호와 함께 한 문장으로 설명해줘."


def _chat_response(payload, document):
    if payload.get("response_format", {}).get("json_schema", {}).get("name") == "AnswerDocument":
        document = AnswerDocument.model_validate(document).model_dump(mode="json")
    return {
        "id": "chatcmpl-physical-lines", "object": "chat.completion", "created": 1,
        "model": payload["model"],
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": json.dumps(document, ensure_ascii=False),
        }}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
    }


@pytest.fixture
def physical_line_api(tmp_path, monkeypatch):
    """Retain the application and SDKs; replace only provider HTTP and owned storage."""
    for name in APP_ENV_SPEC_BY_NAME:
        monkeypatch.delenv(name, raising=False)
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        slack_bot_token="", slack_default_user_id="", slack_default_dm_email="",
        chat_model="gpt-4o-mini", planner_model="gpt-4o-mini", summary_model="gpt-4o-mini",
        synthesis_use_responses_api=False, synthesis_reasoning_effort=None,
        verbose=True,
    )
    monkeypatch.setattr("src.app.web.app.get_settings", lambda: settings)
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGCHAIN_TRACING_V2", "false")
    connect = socket.socket.connect

    def local_only(sock, address):
        if isinstance(address, tuple) and address[0] not in {"127.0.0.1", "::1", "localhost"}:
            raise AssertionError("External network is forbidden in physical line delivery tests")
        return connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", local_only)
    controls = SimpleNamespace(invalid_location=True, repair_location=False, paragraph_code=False,
                               excerpt_repair_required=False, location_repair_required=False,
                               inline_mode=None, invalid_second_location=False, format_probability=0.99, format_refused=False,
                               requests=[], synthesis_packets=[], decisions_inputs=[])

    def provider(request):
        payload = json.loads(request.content)
        controls.requests.append((request.url.path, payload))
        if request.url.path == "/v1/embeddings":
            return httpx.Response(200, json={
                "object": "list", "model": payload["model"],
                "data": [{"object": "embedding", "index": i, "embedding": [1.0, 0.0]}
                         for i in range(len(payload["input"]))],
                "usage": {"prompt_tokens": 1, "total_tokens": 1},
            })
        if request.url.path == "/v1/decisions":
            controls.decisions_inputs.append(json.loads(payload["input"]))
            answers = []
            for question in payload["questions"]:
                name = question["name"]
                if name == "format_fulfilled" and controls.format_refused:
                    answers.append({"type": "refusal", "name": name})
                else:
                    answers.append({"type": "predicate", "name": name,
                                    "probability": controls.format_probability if name == "format_fulfilled" else 0.99})
            return httpx.Response(200, json={
                "model": payload["model"],
                "answers": answers,
                "usage": {"input_tokens": 10, "output_tokens": 0, "total_tokens": 10},
            })
        assert request.url.path == "/v1/chat/completions"
        schema_name = payload.get("response_format", {}).get("json_schema", {}).get("name")
        if schema_name == "PlannerOutput":
            raw_context = next(item["content"] for item in payload["messages"]
                               if item.get("name") == "request_context")
            context_record = json.loads(raw_context.split("\n", 1)[1])
            question = context_record["user_turn_ledger"][-1]["text"]
            answer = ({"content": [{"kind": "code_example", "mode": "required", "evidence_ids": ["code"]}]}
                      if controls.paragraph_code else {})
            evidence = ([{"id": "code", "turn_id": context_record["user_turn_ledger"][-1]["turn_id"],
                          "quote": question, "scope": "answer.content.code_example", "interpretation": "instruction"}]
                        if controls.paragraph_code else [])
            if controls.inline_mode is not None:
                answer["format"] = [{"kind": "inline_code", "mode": "required", "evidence_ids": ["format"]}]
                evidence.append({"id": "format", "turn_id": context_record["user_turn_ledger"][-1]["turn_id"],
                                 "quote": question, "scope": "answer.format", "interpretation": "instruction"})
            plan = {
                "use_retrieval": True,
                "tasks": [{"route": "upload", "query": question if controls.inline_mode is not None else "region amount groupby sum", "k": 4,
                           "requirement": {"file_ids": [item["file_id"] for item in context_record["upload_files"]]}}],
                "request_contract": WireRequestContract(
                    body={"kind": "compose", "instruction": question}, answer=answer, evidence=evidence,
                    slack_recipient={"state": "omitted"},
                ).model_dump(mode="json"),
            }
            document = InitialPlannerOutput.model_validate(plan).model_dump(mode="json")
        elif schema_name == "AnswerDocument":
            packet_message = next(item["content"] for item in payload["messages"]
                                  if str(item["content"]).startswith("[Evidence Packet]"))
            packet = json.loads(packet_message.split("\n", 2)[2])
            controls.synthesis_packets.append(packet)
            if controls.inline_mode is not None:
                source = next(item for item in packet if "x = 1\ny = 2" in item["excerpt"])
                use_inline = controls.inline_mode == "correct" or (
                    controls.inline_mode == "repair" and len(controls.synthesis_packets) > 1
                )
                first = "`x = 1`" if use_inline or controls.inline_mode == "mixed" else "x = 1"
                second = "`y = 2`" if use_inline else "y = 2"
                first_start = source["selection"]["start"] + source["excerpt"].index("x = 1")
                second_start = (source["selection"]["end"] if controls.invalid_second_location
                                else source["selection"]["start"] + source["excerpt"].index("y = 2"))
                document = {"blocks": [{"type": "paragraph", "content": [
                    {"text": "first source position", "basis": "source", "refs": [source["id"]],
                     "source_location": {"ref": source["id"], "start": first_start, "end": first_start + 5}},
                    {"text": f"원문은 {first}이고", "basis": "source", "refs": [source["id"]]},
                    {"text": "second source position", "basis": "source", "refs": [source["id"]],
                     "source_location": {"ref": source["id"], "start": second_start, "end": second_start + 5}},
                    {"text": f"원문은 {second}입니다.", "basis": "source", "refs": [source["id"]]},
                ]}]}
                return httpx.Response(200, json=_chat_response(payload, document))
            evidence = next(item for item in packet if CALCULATION in item["excerpt"])
            feedback = next((item["content"] for item in payload["messages"]
                             if str(item["content"]).startswith("[Validation Repair]")), "")
            location_repair_ready = not controls.location_repair_required or (
                "source_location_mismatch" in feedback and "start/end offsets" in feedback
            )
            invalid_location = controls.invalid_location and not (
                len(controls.synthesis_packets) > 1 and controls.repair_location and location_repair_ready
            )
            start = (evidence["selection"]["end"] if invalid_location
                     else evidence["selection"]["start"] + evidence["excerpt"].index(CALCULATION))
            location = {"text": "source position", "basis": "source", "refs": [evidence["id"]],
                        "source_location": {"ref": evidence["id"], "start": start, "end": start + len(CALCULATION)}}
            basis = "source"
            if controls.excerpt_repair_required and not (
                "excerpt_does_not_match_source" in feedback and "basis=source" in feedback
            ):
                basis = "excerpt"
            document = {"blocks": [
                {"type": "code", "language": "python", "inline": False, "content": {
                    "text": CALCULATION, "basis": basis, "refs": [evidence["id"]],
                }},
                {"type": "paragraph", "content": [location]},
            ]}
            if controls.paragraph_code:
                document = {"blocks": [{"type": "paragraph", "content": [{
                    "text": f'`{CALCULATION}` —',
                    "basis": "source", "refs": [evidence["id"]],
                }, location]}]}
        else:
            raise AssertionError(f"Unexpected provider schema: {schema_name}")
        return httpx.Response(200, json=_chat_response(payload, document))

    with httpx.Client(transport=httpx.MockTransport(provider)) as provider_http:
        monkeypatch.setattr("src.infra.llm.ChatOpenAI", lambda **kwargs: ChatOpenAI(
            **kwargs, http_client=provider_http, base_url="https://physical-lines.test/v1",
        ))
        monkeypatch.setattr("src.infra.llm.OpenAI", lambda **kwargs: OpenAI(
            **kwargs, http_client=provider_http, base_url="https://physical-lines.test/v1",
        ))
        monkeypatch.setattr("src.infra.chroma_store.OpenAIEmbeddings", lambda **kwargs: OpenAIEmbeddings(
            **kwargs, http_client=provider_http, base_url="https://physical-lines.test/v1",
            check_embedding_ctx_length=False,
        ))
        with TestClient(create_app()) as client:
            yield SimpleNamespace(client=client, root=tmp_path, controls=controls)


def test_out_of_evidence_location_cannot_be_delivered_as_completed(physical_line_api):
    api = physical_line_api
    original = SAMPLE.read_bytes()
    assert original.decode("utf-8").splitlines()[29] == CALCULATION
    current = sync(api, add=[staged(api, SAMPLE.name, original)])

    result = final_response(api, query=QUESTION, uploads=context(current))

    assert api.controls.synthesis_packets, "The actual synthesis HTTP boundary was not exercised"
    assert api.controls.decisions_inputs, "The actual Decisions HTTP boundary was not exercised"
    assert result["status"] == "partial", result
    assert result["problem"]["code"] == "model_output_invalid"
    response = AnswerResponse.model_validate(result["response"])
    assert "29행" not in export_answer_text(response)
    assert CALCULATION in export_answer_text(response)
    limitation = next(issue for issue in response.issues if issue.code == "source_location_unverified")
    assert "원본" in limitation.message and "줄" in limitation.message
    assert response.actions == []


def test_location_recovers_when_provider_repairs_the_selected_range(physical_line_api):
    api = physical_line_api
    api.controls.repair_location = True
    api.controls.location_repair_required = True
    current = sync(api, add=[staged(api, SAMPLE.name, SAMPLE.read_bytes())])

    result = final_response(api, query=QUESTION, uploads=context(current))

    assert result["status"] == "completed", result
    response = AnswerResponse.model_validate(result["response"])
    displayed = export_answer_text(response)
    assert CALCULATION in displayed
    assert "sample_data_analysis.py:30" in displayed
    assert "29행" not in displayed
    assert not response.issues
    assert all(check.support_status != "unsupported" for check in response.checks)
    assert len(api.controls.synthesis_packets) == 2


def test_partial_source_quotation_recovers_from_exact_excerpt_diagnostic(physical_line_api, caplog):
    api = physical_line_api
    api.controls.invalid_location = False
    api.controls.excerpt_repair_required = True
    current = sync(api, add=[staged(api, SAMPLE.name, SAMPLE.read_bytes())])

    result = final_response(api, query=QUESTION, uploads=context(current))

    assert result["status"] == "completed", result
    response = AnswerResponse.model_validate(result["response"])
    displayed = export_answer_text(response)
    assert CALCULATION in displayed
    assert "sample_data_analysis.py:30" in displayed
    assert response.content.blocks[0].content.basis == "source"
    assert not response.issues
    assert all(check.support_status != "unsupported" for check in response.checks)
    assert len(api.controls.synthesis_packets) == 2
    saved = api.client.post(f"/sessions/session-a/answers/{result['answer_id']}/save")
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "completed", saved.json()
    saved_answer = AnswerResponse.model_validate(saved.json()["response"])
    artifact = saved_answer.actions[-1].artifact
    download = api.client.get(f"/download/{artifact.filename}")
    assert download.status_code == 200
    assert download.content == export_answer_text(response, include_sources=True).encode("utf-8-sig")
    validation_logs = [record.getMessage() for record in caplog.records
                       if getattr(record, "event", None) == "post_synthesis_validation"]
    assert validation_logs
    assert "excerpt_does_not_match_source" in validation_logs[0]
    assert "missing_content=[]" in validation_logs[0]
    assert "semantic_failures={}" in validation_logs[0]
    assert all(CALCULATION not in message and QUESTION not in message for message in validation_logs)


def test_physical_line_is_shared_by_model_validation_answer_and_download(physical_line_api):
    api = physical_line_api
    api.controls.invalid_location = False
    current = sync(api, add=[staged(api, SAMPLE.name, SAMPLE.read_bytes())])

    result = final_response(api, query=QUESTION, uploads=context(current))

    assert result["status"] == "completed", result
    packet_source = next(item for packet in api.controls.synthesis_packets for item in packet
                         if CALCULATION in item["excerpt"])
    expected_line = CALCULATION + "\n"
    record = next(item for item in packet_source["source_lines"] if item["line_number"] == 30)
    start = packet_source["selection"]["start"]
    assert packet_source["excerpt"][record["start"] - start:record["end"] - start] == expected_line
    assert "text" not in record
    assert record["partial_start"] is False
    assert record["partial_end"] is False

    def sources(value):
        if isinstance(value, dict):
            if value.get("excerpt") and CALCULATION in value["excerpt"]:
                yield value
            for child in value.values():
                yield from sources(child)
        elif isinstance(value, list):
            for child in value:
                yield from sources(child)

    validation_sources = [(payload, source) for payload in api.controls.decisions_inputs for source in sources(payload)]
    assert validation_sources
    for payload, source in validation_sources:
        record = next(item for item in source["source_lines"] if item["line_number"] == 30)
        start = source["selection"]["start"]
        assert source["excerpt"][record["start"] - start:record["end"] - start] == expected_line
        if "evaluation_policy" in payload:
            assert "text" not in record
        else:
            assert record["text"] == expected_line

    original = AnswerResponse.model_validate(result["response"])
    displayed = export_answer_text(original)
    assert CALCULATION in displayed
    assert "sample_data_analysis.py:30" in displayed
    location = original.content.blocks[1].content[0].source_location
    assert location.ref == original.citations[0].evidence.id
    assert location.ref == original.content.blocks[1].content[0].refs[0]
    saved = api.client.post(f"/sessions/session-a/answers/{result['answer_id']}/save")
    assert saved.status_code == 200, saved.text
    saved_result = saved.json()
    assert saved_result["status"] == "completed", saved_result
    saved_answer = AnswerResponse.model_validate(saved_result["response"])
    assert saved_answer.model_copy(update={"actions": []}) == original
    receipt = saved_answer.actions[-1]
    download = api.client.get(f"/download/{receipt.artifact.filename}")
    assert download.status_code == 200, download.text
    assert download.content == export_answer_text(original, include_sources=True).encode("utf-8-sig")
    assert CALCULATION in download.content.decode("utf-8-sig")
    assert "sample_data_analysis.py:30" in download.content.decode("utf-8-sig")


def test_exact_source_line_in_paragraph_completes_and_downloads(physical_line_api):
    api = physical_line_api
    api.controls.invalid_location = False
    api.controls.paragraph_code = True
    current = sync(api, add=[staged(api, SAMPLE.name, SAMPLE.read_bytes())])

    result = final_response(api, query=QUESTION, uploads=context(current))

    assert result["status"] == "completed", result
    response = AnswerResponse.model_validate(result["response"])
    assert response.content.blocks[0].type == "paragraph"
    assert not response.issues
    displayed = export_answer_text(response)
    assert f"`{CALCULATION}`" in displayed
    assert "sample_data_analysis.py:30" in displayed
    saved = api.client.post(f"/sessions/session-a/answers/{result['answer_id']}/save")
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "completed", saved.json()
    saved_answer = AnswerResponse.model_validate(saved.json()["response"])
    artifact = saved_answer.actions[-1].artifact
    download = api.client.get(f"/download/{artifact.filename}")
    assert download.status_code == 200
    assert download.content == export_answer_text(response, include_sources=True).encode("utf-8-sig")


def test_missing_required_inline_code_cannot_complete_with_high_model_scores(physical_line_api):
    api = physical_line_api
    api.controls.inline_mode = "missing"
    current = sync(api, add=[staged(api, "two_lines.py", INLINE_SOURCE)])

    result = final_response(api, query=INLINE_QUESTION, uploads=context(current))

    assert api.controls.synthesis_packets, "The real synthesis boundary must run before format rejection"
    assert api.controls.decisions_inputs
    assert result["status"] in {"partial", "failed"}, result
    assert result["problem"] is not None


@pytest.mark.parametrize("inline_mode", ["correct", "repair"])
def test_required_inline_code_with_correct_lines_completes_and_preserves_saved_bytes(physical_line_api, inline_mode):
    api = physical_line_api
    api.controls.inline_mode = inline_mode
    current = sync(api, add=[staged(api, "two_lines.py", INLINE_SOURCE)])

    result = final_response(api, query=INLINE_QUESTION, uploads=context(current))

    assert result["status"] == "completed", result
    response = AnswerResponse.model_validate(result["response"])
    paragraph, = response.content.blocks
    assert paragraph.type == "paragraph"
    assert " ".join(unit.text for unit in paragraph.content) == "two_lines.py:1 원문은 `x = 1`이고 two_lines.py:2 원문은 `y = 2`입니다."
    assert not response.issues
    assert all(not check.issues for check in response.checks)
    saved = api.client.post(f"/sessions/session-a/answers/{result['answer_id']}/save")
    assert saved.status_code == 200, saved.text
    assert saved.json()["status"] == "completed", saved.json()
    saved_answer = AnswerResponse.model_validate(saved.json()["response"])
    assert saved_answer.model_copy(update={"actions": []}) == response
    receipt = saved_answer.actions[-1]
    download = api.client.get(f"/download/{receipt.artifact.filename}")
    assert download.status_code == 200, download.text
    assert download.content == export_answer_text(response, include_sources=True).encode("utf-8-sig")
    assert download.content == Path(receipt.file_path).read_bytes()
    assert b"`x = 1`" in download.content and b"`y = 2`" in download.content


def test_inline_code_does_not_override_an_invalid_source_location(physical_line_api):
    api = physical_line_api
    api.controls.inline_mode = "correct"
    api.controls.invalid_second_location = True
    current = sync(api, add=[staged(api, "two_lines.py", INLINE_SOURCE)])

    result = final_response(api, query=INLINE_QUESTION, uploads=context(current))

    assert api.controls.synthesis_packets
    assert result["status"] == "partial", result
    assert result["problem"]["code"] == "model_output_invalid"
    response = AnswerResponse.model_validate(result["response"])
    displayed = export_answer_text(response)
    assert "two_lines.py:1" in displayed
    assert "`x = 1`" in displayed and "`y = 2`" in displayed
    assert "second source position" not in displayed
    assert any(issue.code == "source_location_unverified" for issue in response.issues)


@pytest.mark.parametrize(("probability", "refused"), [
    pytest.param(0.1, False, id="format-failed"),
    pytest.param(0.99, True, id="format-refused"),
])
def test_partial_inline_format_cannot_complete_when_separate_format_review_fails(physical_line_api, probability, refused):
    api = physical_line_api
    api.controls.inline_mode = "mixed"
    api.controls.format_probability = probability
    api.controls.format_refused = refused
    current = sync(api, add=[staged(api, "two_lines.py", INLINE_SOURCE)])

    result = final_response(api, query=INLINE_QUESTION, uploads=context(current))

    assert api.controls.synthesis_packets
    assert result["status"] in {"partial", "failed"}, result
    assert result["problem"] is not None
    questions = [question["name"] for path, payload in api.controls.requests if path == "/v1/decisions"
                 for question in payload["questions"]]
    assert "format_fulfilled" in questions
