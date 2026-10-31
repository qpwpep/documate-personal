"""Saved answer exports retain their original identity across new copy requests."""
from pathlib import Path
from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.app.answer_exports import AnswerExportService
from src.app.web.routes import router
from src.app.web.session_store import InMemorySessionStore
from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.contracts.provenance import AnswerProvenance
from src.eval.config_models import BenchmarkCase
from src.eval.result_models import ScenarioTurnResult
from src.eval.save_outcomes import assess_save_outcome
from src.infra import saved_artifacts
from src.infra.tools.save_text import build_save_text_tool
from tests.core.helpers import SequentialGraph
from tests.core.test_exact_answer_export import export_graph
from tests.core.test_pending_action_delivery import _ConsumerGraph, _contract, _manager


@pytest.fixture
def saved_answer_flow(tmp_path, monkeypatch, request):
    origin, expired = getattr(request, "param", ("natural", False))
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    monkeypatch.setattr("src.app.web.routes.get_save_text_output_dir", lambda: tmp_path)
    manager = _manager(SequentialGraph(
        _ConsumerGraph([_contract(save="requested" if origin == "compose" else "not_requested")],
                       ["확정된 원본 답변입니다."], build_save_text_tool(), None),
        export_graph(), export_graph(),
    ))
    store = InMemorySessionStore(settings=manager.settings, agent_factory=lambda: manager)
    store.get_or_create("save-eval-session")
    app = FastAPI()
    app.state.answer_export_service = AnswerExportService(settings=manager.settings, session_store=store)
    app.include_router(router)
    with TestClient(app) as client:
        # Replace only the network boundary; downloads still validate real manifests and bytes.
        monkeypatch.setattr("src.eval.save_outcomes.requests.get",
                            lambda url, **kwargs: client.get(urlsplit(url).path))
        setup = manager.run_agent_flow("원본 답변을 작성해줘")
        original = AnswerResponse.model_validate(setup["response"])
        prior = ScenarioTurnResult(
            role="setup", query="원본 답변을 작성해줘", request_payload={"session_id": manager.session.session_id},
            response=original, answer_provenance=AnswerProvenance.model_validate(setup["debug"]["answer_provenance"]),
        )
        initial = original.actions[0] if origin == "compose" else None
        if origin == "direct":
            direct = client.post(f"/sessions/save-eval-session/answers/{setup['answer_id']}/save").json()
            initial = AnswerResponse.model_validate(direct["response"]).actions[0]
        if expired:
            monkeypatch.setattr(saved_artifacts, "time", lambda: initial.artifact.expires_at + 1)
        result = manager.run_agent_flow("방금 답변을 그대로 TXT 파일로 저장해줘")
        yield manager, prior, result, initial
    store.close_all()


def _assess(result, prior, session_id):
    response = AnswerResponse.model_validate(result["response"])
    return assess_save_outcome(
        case=BenchmarkCase(
            case_id="exact-answer-export", category="tool_action", query="방금 답변을 그대로 저장해줘",
            setup_turns=[prior.query],
            save_expectation={"outcome": "required_success", "target": {"kind": "setup_answer", "setup_turn_index": 0}},
        ),
        response=response, actions=response.actions, called_tools=result["debug"]["tool_calls"],
        prior_turns=[prior], session_id=session_id, endpoint="http://testserver",
        provenance=AnswerProvenance.model_validate(result["debug"]["answer_provenance"]),
    )


@pytest.mark.parametrize("saved_answer_flow", [
    ("natural", False), ("direct", False), ("direct", True), ("compose", False), ("compose", True),
], indirect=True)
def test_exact_answer_export_is_verified_across_request_and_retention_boundaries(saved_answer_flow):
    manager, prior, result, initial = saved_answer_flow
    receipt = AnswerResponse.model_validate(result["response"]).actions[0]
    assert result["status"] == "completed"
    assert receipt.operation.request_id != result["debug"]["answer_provenance"]["request_id"]
    assert _assess(result, prior, manager.session.session_id).status == "verified"
    assert Path(receipt.file_path).read_bytes() == export_answer_text(prior.response, include_sources=True).encode("utf-8-sig")
    if initial is not None:
        assert receipt.operation.request_id == initial.operation.request_id
    repeated = manager.run_agent_flow("방금 답변을 그대로 TXT 파일로 저장해줘")
    assert repeated["response"]["actions"][0]["operation"] == receipt.operation.model_dump(mode="json")
    assert _assess(repeated, prior, manager.session.session_id).status == "verified"


@pytest.mark.parametrize("defect, code", [
    ("session", "save_request_mismatch"),
    ("binding", "save_operation_mismatch"),
    ("source", "save_target_mismatch"),
    ("missing_source", "save_target_unverifiable"),
    ("file", "save_artifact_mismatch"),
])
def test_export_verification_rejects_wrong_source_binding_session_and_bytes(saved_answer_flow, defect, code):
    manager, prior, result, _ = saved_answer_flow
    receipt = result["response"]["actions"][0]
    provenance = result["debug"]["answer_provenance"]
    if defect == "session":
        receipt["operation"]["session_id"] = "another-session"
    elif defect == "binding":
        receipt["operation"]["operation_id"] = "another-operation"
    elif defect == "source":
        provenance["source"]["response_hash"] = "f" * 64
    elif defect == "missing_source":
        provenance["source"] = None
    else:
        Path(receipt["file_path"]).write_text("wrong bytes", encoding="utf-8")
    assert _assess(result, prior, manager.session.session_id).failure_codes == [code]


@pytest.mark.parametrize("saved_answer_flow", [("compose", False)], indirect=True)
def test_newly_composed_save_still_requires_the_current_request_identity(saved_answer_flow):
    manager, prior, _, _ = saved_answer_flow
    receipt = prior.response.actions[0]
    operation = receipt.operation.model_copy(update={"request_id": "unrelated-request"})
    changed = receipt.model_copy(update={"operation": operation})
    # Even a matching committed binding cannot authorize another request's
    # operation for a newly composed answer.
    provenance = prior.answer_provenance.model_copy(update={"save_operation_binding_sha256": operation.binding_sha256})
    assessment = assess_save_outcome(
        case=BenchmarkCase(
            case_id="composed-answer-save", category="tool_action", query="작성하고 저장해줘",
            save_expectation={"outcome": "required_success", "target": {"kind": "final_answer"}},
        ),
        response=prior.response, actions=[changed], called_tools=["save_text"], prior_turns=[],
        session_id=manager.session.session_id, endpoint="http://testserver", provenance=provenance,
    )
    assert assessment.failure_codes == ["save_request_mismatch"]
