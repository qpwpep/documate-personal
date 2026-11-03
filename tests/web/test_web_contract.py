"""Python and browser clients consume the same complete public response examples."""
from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path

import pytest
import requests
from jsonschema import Draft202012Validator

from src.app.client import AgentRequestContext, AgentSessionClient
from src.app.web.app import create_app
from src.app.web.schemas import AgentResponse
from src.core.slack_contract import RecipientInput
from src.core.uploads import UploadManifest


CONTRACT_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "web_contracts" / "agent_responses.json"
STATUSES = ("completed", "partial", "needs_input", "failed", "refused")


@pytest.mark.parametrize("field", ["status", "message", "missing_slots", "debug"])
def test_output_schema_rejects_missing_serialized_defaults(
    responses: dict[str, dict[str, object]], field: str,
) -> None:
    schema = create_app().openapi()
    validator = Draft202012Validator({
        "$ref": "#/components/schemas/AgentResponse", "components": schema["components"],
    })
    payload = {key: value for key, value in responses["completed"].items() if key != field}

    assert not validator.is_valid(payload)
    assert validator.is_valid(responses["completed"])


def test_output_defaults_do_not_change_optional_request_fields() -> None:
    schemas = create_app().openapi()["components"]["schemas"]

    assert "include_debug" not in schemas["AgentRequest"]["required"]
    assert "add" not in schemas["BrowserUploadSyncRequest"]["required"]
    assert schemas["RecipientInput"] == RecipientInput.model_json_schema()
    assert {"selector", "failure"} <= set(schemas["SlackDefault"]["required"])
    assert {"actions", "content_hash", "semantic_validation"} <= set(schemas["AnswerResponse"]["required"])


@pytest.fixture
def responses() -> dict[str, dict[str, object]]:
    with CONTRACT_PATH.open(encoding="utf-8") as contract_file:
        return json.load(contract_file)["responses"]


@pytest.fixture
def http_response(monkeypatch: pytest.MonkeyPatch) -> Callable[[bytes], list[requests.PreparedRequest]]:
    def install(payload: bytes) -> list[requests.PreparedRequest]:
        sent: list[requests.PreparedRequest] = []
        response = requests.Response()
        response.status_code = 200
        response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
        response.raw = io.BytesIO(payload)

        def send(_adapter: requests.adapters.HTTPAdapter, request: requests.PreparedRequest,
                 **_kwargs: object) -> requests.Response:
            sent.append(request)
            response.request = request
            response.url = request.url
            return response

        monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", send)
        return sent

    return install


@pytest.mark.parametrize("status", STATUSES)
def test_shared_responses_are_complete_server_outputs(
    responses: dict[str, dict[str, object]], status: str,
) -> None:
    payload = responses[status]
    result = AgentResponse.model_validate(payload)

    assert result.status == status
    assert result.model_dump(mode="json") == payload
    if status == "completed":
        assert result.response.content.blocks[0].content[0].text == "한글 답변 🧪"
        assert result.problem is None
    elif status == "partial":
        assert result.response.issues[0].message == "두 번째 자료는 확인하지 못했습니다."
        assert result.problem.code == "evidence_insufficient"
    elif status == "needs_input":
        assert result.message == "비교할 파일을 첨부해 주세요."
        assert result.missing_slots == ["upload"]
    else:
        assert result.response is None
        assert result.upload_manifest is None


@pytest.mark.parametrize("status", STATUSES)
def test_python_client_preserves_shared_result_and_confirmed_attachment_state(
    responses: dict[str, dict[str, object]], status: str,
    http_response: Callable[[bytes], list[requests.PreparedRequest]],
) -> None:
    payload = responses[status]
    encoded = json.dumps(payload, ensure_ascii=False)
    sent = http_response(f"event: final_response\ndata: {encoded}\n\nevent: done\ndata: {{}}\n\n".encode("utf-8"))
    client = AgentSessionClient(
        AgentRequestContext(fastapi_url="http://contract.test", session_id="contract-session"),
        manifest=UploadManifest(epoch="contract-epoch", revision=1, files=[]),
    )

    events = list(client.stream("  질문 원문 🧪\n"))

    assert len(sent) == 1
    assert sent[0].method == "POST"
    assert json.loads(sent[0].body)["query"] == "  질문 원문 🧪\n"
    assert len(events) == 1
    assert events[0].event == "final_response"
    assert events[0].data == payload
    assert events[0].result.status == status
    if status in {"failed", "refused"}:
        assert client.manifest is None
    else:
        assert client.manifest == UploadManifest(epoch="contract-epoch", revision=2, files=[])


def test_python_client_keeps_server_error_before_a_valid_final_response(
    responses: dict[str, dict[str, object]],
    http_response: Callable[[bytes], list[requests.PreparedRequest]],
) -> None:
    diagnostic = {"message": "앞선 단계의 진단을 확인해 주세요.", "stage": "planner"}
    frames = [
        f"event: error\ndata: {json.dumps(diagnostic, ensure_ascii=False)}\n\n",
        f"event: final_response\ndata: {json.dumps(responses['completed'], ensure_ascii=False)}\n\n",
    ]
    sent = http_response("".join(frames).encode("utf-8"))
    client = AgentSessionClient(
        AgentRequestContext(fastapi_url="http://contract.test", session_id="contract-session"),
        manifest=UploadManifest(epoch="contract-epoch", revision=1, files=[]),
    )

    events = list(client.stream("질문"))

    assert len(sent) == 1
    assert events[0].event == "error"
    assert events[0].data == diagnostic
    assert events[-1].result.status == "completed"
    assert events[-1].data == responses["completed"]
    assert client.manifest == UploadManifest(epoch="contract-epoch", revision=2, files=[])


@pytest.mark.parametrize("status", ["completed", "partial", "needs_input"])
def test_server_and_python_client_reject_unconfirmed_attachment_state(
    responses: dict[str, dict[str, object]], status: str,
    http_response: Callable[[bytes], list[requests.PreparedRequest]],
) -> None:
    payload = {**responses[status], "upload_manifest": None}
    with pytest.raises(ValueError, match="requires upload_manifest"):
        AgentResponse.model_validate(payload)

    sent = http_response(
        f"event: final_response\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"),
    )
    client = AgentSessionClient(
        AgentRequestContext(fastapi_url="http://contract.test", session_id="contract-session"),
        manifest=UploadManifest(epoch="contract-epoch", revision=1, files=[]),
    )

    events = list(client.stream("질문"))

    assert len(sent) == 1
    assert len(events) == 1
    assert events[0].event == "error"
    assert events[0].data["code"] == "invalid_stream"
    assert events[0].result is None
    assert client.manifest is None
