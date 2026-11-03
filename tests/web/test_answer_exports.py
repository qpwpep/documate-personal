"""Answer exports keep server snapshots and verified downloadable bytes intact."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from src.app.answer_exports import AnswerExportService
from src.app.web.app import create_app
from src.app.web.session_store import InMemorySessionStore
from src.core.answer_schema import AnswerResponse, ResponseIssue, export_answer_text, finalize_answer, text_document
from src.core.answer_schema.models import SemanticCheck, SemanticValidation, semantic_evidence_fingerprint
from src.core.contracts.graph_state import PendingAction
from src.core.save_contract import SaveOperation
from src.infra.settings import AppSettings
from src.runtime.agent_runtime.session_context import SessionContext
from tests.core.test_answer_validation import _evidence
from tests.core.test_pending_action_delivery import _contract, _manager
from tests.core.test_exact_answer_export import checked_source, export_graph


@pytest.fixture
def exports(tmp_path, monkeypatch):
    monkeypatch.setattr("src.app.web.routes.get_save_text_output_dir", lambda: tmp_path)
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    settings = AppSettings(_env_file=None)

    def manager():
        session = SessionContext()
        return SimpleNamespace(session=session, close=session.close)

    store = InMemorySessionStore(settings=settings, agent_factory=manager)
    session = store.get_or_create("export-session").session
    evidence = _evidence("regional = df.groupby('region')['amount'].sum()")
    original = finalize_answer(
        text_document(evidence.excerpt, basis="excerpt", refs=[evidence.id]), [evidence],
        issues=[ResponseIssue(code="scope", message="첨부 파일의 코드만 확인했습니다.")],
    )
    answer_id = session.register_answer(original)
    app = create_app()
    app.router.lifespan_context = lambda _app: nullcontext()
    app.state.answer_export_service = AnswerExportService(settings=settings, session_store=store)
    with TestClient(app) as client:
        yield client, store, session, answer_id, original, tmp_path
    store.close_all()


def test_direct_save_preserves_server_answer_sources_and_limits(exports):
    client, _, session, answer_id, original, _ = exports
    # Client-controlled or subsequent displayed bodies must never replace the snapshot.
    session.previous_response = finalize_answer(text_document("another answer"), [])
    response = client.post(f"/sessions/export-session/answers/{answer_id}/save", json={"content": "tampered"})

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    saved = AnswerResponse.model_validate(result["response"])
    assert saved.model_copy(update={"actions": []}) == original
    receipt = saved.actions[-1]
    assert receipt.verification == "verified"
    download = client.get(f"/download/{receipt.artifact.filename}")
    assert download.status_code == 200
    assert session.get_answer(answer_id) == original


def test_sentence_scope_survives_saved_snapshot(exports):
    client, _, session, _, original, _ = exports
    document = original.content.model_copy(deep=True)
    document.blocks[0].content[0].sentence_scope = 1
    scoped = finalize_answer(
        document, [citation.evidence for citation in original.citations], issues=original.issues,
    )
    answer_id = session.register_answer(scoped)

    response = client.post(f"/sessions/export-session/answers/{answer_id}/save")

    assert response.status_code == 200
    result = response.json()
    assert result["status"] == "completed"
    saved = AnswerResponse.model_validate(result["response"])
    assert saved.content.blocks[0].content[0].sentence_scope == 1
    assert saved.citations == original.citations
    assert saved.model_copy(update={"actions": []}) == scoped
    download = client.get(f"/download/{saved.actions[-1].artifact.filename}")
    assert download.status_code == 200
    assert session.get_answer(answer_id) == scoped


def test_duplicate_save_requests_reuse_one_artifact(exports):
    client, _, _, answer_id, _, output = exports
    url = f"/sessions/export-session/answers/{answer_id}/save"
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(url).json(), range(2)))
    assert [item["status"] for item in responses] == ["completed", "completed"]
    assert responses[0]["response"]["actions"] == responses[1]["response"]["actions"]
    assert len(list(output.glob("*.txt"))) == 1


def test_published_export_with_unavailable_readback_remains_unknown(exports, monkeypatch):
    client, _, session, answer_id, original, output = exports
    real_read = Path.read_bytes

    def unavailable_artifact_readback(path):
        if path.parent == output and path.suffix == ".txt":
            raise PermissionError("readback unavailable")
        return real_read(path)

    with monkeypatch.context() as patch:
        patch.setattr(Path, "read_bytes", unavailable_artifact_readback)
        result = client.post(f"/sessions/export-session/answers/{answer_id}/save").json()

    assert result["status"] == "failed"
    receipt = result["response"]["actions"][-1]
    assert receipt["status"] == "unknown"
    assert receipt["verification"] == "unverifiable"
    assert receipt["error_code"] == "artifact_unverifiable"
    assert len(list(output.glob("*.txt"))) == 1
    assert session.get_answer(answer_id) == original


def test_answer_snapshot_is_isolated_from_callers_and_other_sessions(exports):
    client, store, session, answer_id, original, _ = exports
    original.content.blocks[0].content[0].text = "mutated caller"
    first_read = session.get_answer(answer_id)
    first_read.content.blocks[0].content[0].text = "mutated reader"
    assert "mutated" not in export_answer_text(session.get_answer(answer_id))
    store.get_or_create("other-session")
    response = client.post(f"/sessions/other-session/answers/{answer_id}/save")
    assert response.status_code == 404


@pytest.mark.parametrize("slack", ["not_requested", "requested", "unresolved"])
@pytest.mark.parametrize("revalidated", [False, True])
def test_direct_retry_completes_only_the_matching_prepared_save(exports, slack, revalidated):
    client, _, session, answer_id, original, output = exports
    pending_response = original
    if revalidated:
        pending_response = original.model_copy(update={"semantic_validation": SemanticValidation(
            content_hash=original.content_hash, evidence_fingerprint=semantic_evidence_fingerprint(original.citations),
            threshold=0.8, checks=[SemanticCheck(name="request_fulfilled", probability=0.95)],
        )})
    operation = SaveOperation.for_text(
        export_answer_text(original, include_sources=True), operation_id="original-obligation",
        session_id=session.session_id, request_id="original-request", contract_revision=1,
        target_kind="compose", source_hash=original.content_hash, answer_hash=original.content_hash,
    )
    session.pending_action = PendingAction(
        contract=_contract(save="requested", slack=slack),
        response=pending_response, body_prepared=True, phase="awaiting_delivery", save_operation=operation,
    )
    result = client.post(f"/sessions/export-session/answers/{answer_id}/save").json()
    assert result["status"] == "completed"
    assert result["response"]["actions"][-1]["operation"] == operation.model_dump(mode="json")
    if slack == "not_requested":
        assert session.pending_action is None
    else:
        assert session.pending_action.completed_actions == ("save_text",)
        assert session.pending_action.save_receipt.verification == "verified"
        assert session.pending_action.contract.actions.slack_notify.intent == slack
    repeated = client.post(f"/sessions/export-session/answers/{answer_id}/save").json()
    assert repeated["status"] == "completed"
    assert repeated["response"]["actions"][-1]["operation"] == operation.model_dump(mode="json")
    assert len(list(output.glob("*.txt"))) == 1


def test_revalidated_snapshots_share_exports_and_the_current_expiry_generation(exports, monkeypatch):
    from src.infra import saved_artifacts

    client, _, session, _, _, output = exports
    original = checked_source()
    revised_report = original.semantic_validation.model_copy(update={
        "model": "revalidated-model", "checks": [SemanticCheck(name="request_fulfilled", probability=0.95)],
    })
    revised = original.model_copy(update={"semantic_validation": revised_report})
    original_id = session.register_answer(original)
    revised_id = session.register_answer(revised)
    assert original_id != revised_id
    assert session.get_answer(original_id) == original
    assert session.get_answer(revised_id) == revised
    original_url = f"/sessions/export-session/answers/{original_id}/save"
    revised_url = f"/sessions/export-session/answers/{revised_id}/save"

    first = client.post(original_url).json()["response"]["actions"][0]
    repeated = client.post(revised_url).json()["response"]["actions"][0]
    assert repeated["operation"] == first["operation"]
    assert repeated["artifact"] == first["artifact"]
    assert len(list(output.glob("*.txt"))) == 1

    old_manifest = output / (first["artifact"]["filename"] + ".json")
    manifest_bytes = old_manifest.read_bytes()
    monkeypatch.setattr(saved_artifacts, "time", lambda: first["artifact"]["expires_at"] + 1)
    renewed = client.post(revised_url).json()["response"]["actions"][0]
    assert renewed["operation"]["operation_id"] != first["operation"]["operation_id"]
    assert client.post(original_url).json()["response"]["actions"][0] == renewed
    assert old_manifest.read_bytes() == manifest_bytes
    assert len(list(output.glob("*.txt"))) == 2


@pytest.mark.parametrize("change", ["body", "source", "limitation"])
def test_distinct_export_content_keeps_separate_operations(exports, change):
    client, _, session, original_id, original, output = exports
    evidence = original.citations[0].evidence
    if change == "body":
        changed = finalize_answer(text_document("변경된 설명", basis="source", refs=[evidence.id]), [evidence])
    elif change == "source":
        evidence = _evidence(evidence.excerpt, source_type="official")
        changed = finalize_answer(text_document(evidence.excerpt, basis="excerpt", refs=[evidence.id]), [evidence])
    else:
        changed = original.model_copy(update={"issues": [ResponseIssue(code="scope", message="변경된 제한입니다.")]})
    changed_id = session.register_answer(changed)

    first = client.post(f"/sessions/export-session/answers/{original_id}/save").json()["response"]["actions"][0]
    second = client.post(f"/sessions/export-session/answers/{changed_id}/save").json()["response"]["actions"][0]

    assert first["operation"]["operation_id"] != second["operation"]["operation_id"]
    assert first["artifact"]["artifact_id"] != second["artifact"]["artifact_id"]
    assert len(list(output.glob("*.txt"))) == 2


def test_exporting_partial_answer_does_not_finish_its_incomplete_original_request(exports):
    client, _, session, answer_id, original, _ = exports
    pending = PendingAction(
        contract=_contract(save="requested"),
        response=original, body_prepared=False, phase="awaiting_body",
    )
    session.pending_action = pending
    result = client.post(f"/sessions/export-session/answers/{answer_id}/save").json()
    assert result["status"] == "completed"
    assert session.pending_action == pending


def test_natural_language_failure_direct_retry_and_natural_repeat_share_one_file(tmp_path, monkeypatch):
    from src.infra import saved_artifacts

    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: tmp_path)
    monkeypatch.setattr("src.app.web.routes.get_save_text_output_dir", lambda: tmp_path)
    manager = _manager(export_graph())
    original = checked_source()
    manager.session.previous_response = original
    real_link = saved_artifacts.os.link

    def unavailable_payload(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("temporary disk failure")
        return real_link(source, destination)

    with monkeypatch.context() as failing:
        failing.setattr(saved_artifacts.os, "link", unavailable_payload)
        failed = manager.run_agent_flow("방금 답변을 출처와 함께 txt 파일로 저장해줘")
    assert failed["status"] == "partial"
    assert failed["problem"]["code"] == "save_failed"
    assert failed["response"]["actions"][0]["status"] == "error"
    assert failed["answer_id"]
    assert manager.session.pending_action is not None
    store = InMemorySessionStore(settings=manager.settings, agent_factory=lambda: manager)
    store.get_or_create("flow-session")
    app = create_app()
    app.router.lifespan_context = lambda _app: nullcontext()
    app.state.answer_export_service = AnswerExportService(settings=manager.settings, session_store=store)
    with TestClient(app) as client:
        direct = client.post(f"/sessions/flow-session/answers/{failed['answer_id']}/save").json()
        assert direct["status"] == "completed"
        assert manager.session.pending_action is None
        repeated = manager.run_agent_flow("방금 답변을 출처와 함께 txt 파일로 저장해줘")
        assert repeated["status"] == "completed"
        receipts = [item["response"]["actions"][0] for item in (failed, direct, repeated)]
        assert receipts[0]["operation"] == receipts[1]["operation"] == receipts[2]["operation"]
    assert len(list(tmp_path.glob("*.txt"))) == 1


def test_expired_export_creates_one_new_generation_without_changing_old_artifact(exports, monkeypatch):
    from src.infra import saved_artifacts

    client, _, _, answer_id, _, output = exports
    url = f"/sessions/export-session/answers/{answer_id}/save"
    first = client.post(url).json()["response"]["actions"][0]
    old_manifest = output / (first["artifact"]["filename"] + ".json")
    manifest_bytes = old_manifest.read_bytes()
    monkeypatch.setattr(saved_artifacts, "time", lambda: first["artifact"]["expires_at"] + 1)

    renewed = client.post(url).json()

    assert renewed["status"] == "completed"
    next_receipt = renewed["response"]["actions"][0]
    assert next_receipt["operation"]["operation_id"] != first["operation"]["operation_id"]
    repeated = client.post(url).json()
    assert repeated["response"]["actions"][0]["operation"] == next_receipt["operation"]
    assert client.get(f"/download/{first['artifact']['filename']}").status_code == 410
    assert old_manifest.read_bytes() == manifest_bytes
    assert len(list(output.glob("*.txt"))) == 2


def test_expired_export_write_failure_retries_the_same_new_generation(exports, monkeypatch):
    from src.infra import saved_artifacts

    client, _, _, answer_id, _, output = exports
    url = f"/sessions/export-session/answers/{answer_id}/save"
    first = client.post(url).json()["response"]["actions"][0]
    monkeypatch.setattr(saved_artifacts, "time", lambda: first["artifact"]["expires_at"] + 1)
    real_link = saved_artifacts.os.link

    def unavailable_payload(source, destination):
        if Path(destination).suffix == ".txt":
            raise OSError("temporary disk failure")
        return real_link(source, destination)

    with monkeypatch.context() as failing:
        failing.setattr(saved_artifacts.os, "link", unavailable_payload)
        failed = client.post(url).json()
    assert failed["status"] == "failed"
    failed_receipt = failed["response"]["actions"][0]
    assert failed_receipt["operation"]["operation_id"] != first["operation"]["operation_id"]
    retried = client.post(url).json()
    assert retried["status"] == "completed"
    assert retried["response"]["actions"][0]["operation"] == failed_receipt["operation"]
    assert len(list(output.glob("*.txt"))) == 2


@pytest.mark.parametrize("damage", ["invalid_json", "changed_binding"])
def test_export_does_not_renew_damaged_or_conflicting_expired_metadata(exports, monkeypatch, damage):
    from src.infra import saved_artifacts

    client, _, _, answer_id, _, output = exports
    url = f"/sessions/export-session/answers/{answer_id}/save"
    first = client.post(url).json()["response"]["actions"][0]
    manifest_path = output / (first["artifact"]["filename"] + ".json")
    if damage == "invalid_json":
        manifest_path.write_text("broken", encoding="utf-8")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["operation"]["source_hash"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(saved_artifacts, "time", lambda: first["artifact"]["expires_at"] + 1)

    failed = client.post(url).json()

    assert failed["status"] == "failed"
    assert failed["response"]["actions"][0]["operation"] == first["operation"]
    assert len(list(output.glob("*.txt"))) == 1
