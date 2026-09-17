from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.app.web.routes import router
from src.core.save_contract import SaveOperation
from src.infra.saved_artifacts import manifest_path_for
from src.infra.tools.save_text import build_save_text_tool


@pytest.fixture
def downloads(tmp_path, monkeypatch):
    monkeypatch.setattr("src.app.web.routes.get_save_text_output_dir", lambda: tmp_path)
    app = FastAPI()
    app.include_router(router)
    with TestClient(app) as client:
        yield client, tmp_path


@pytest.fixture
def saved(downloads, monkeypatch):
    _, root = downloads
    monkeypatch.setattr("src.infra.tools.save_text.get_save_text_output_dir", lambda: root)
    text = "사용자가 요청한 답변\n출처와 제한도 보존합니다."
    operation = SaveOperation.for_text(
        text, operation_id="save-download", session_id="session-download", request_id="request-download",
        contract_revision=1, target_kind="compose", source_hash="a" * 64, answer_hash="b" * 64,
    )
    save = build_save_text_tool(ttl_seconds=60)
    result = save(text, operation=operation)
    return result, text, operation, save


def test_download_does_not_publish_a_file_without_a_committed_manifest(downloads):
    """An existing TXT alone cannot prove that a save operation completed."""
    client, root = downloads
    (root / "uncommitted.txt").write_bytes(b"partial or legacy data")

    response = client.get("/download/uncommitted.txt")

    assert response.status_code == 404
    assert response.json()["detail"]["code"] == "manifest_missing"


def test_download_does_not_expose_manifest_files(downloads):
    client, root = downloads
    (root / "response.txt.json").write_text('{"private": "metadata"}', encoding="utf-8")

    response = client.get("/download/response.txt.json")

    assert response.status_code == 404
    assert "private" not in response.text


def test_download_returns_the_actual_verified_bytes_and_operation_binding(downloads, saved):
    client, _ = downloads
    result, text, operation, _ = saved
    artifact = result["artifact"]

    response = client.get(f"/download/{artifact['filename']}")

    assert response.status_code == 200
    assert response.content == text.encode("utf-8-sig")
    assert response.headers["x-save-binding-sha256"] == operation.binding_sha256
    assert response.headers["x-artifact-id"] == artifact["artifact_id"]
    assert response.headers["etag"] == f'"{artifact["sha256"]}"'
    assert artifact["filename"] in response.headers["content-disposition"]


@pytest.mark.parametrize("damage,status,code", [
    ("delete_payload", 404, "artifact_missing"),
    ("delete_manifest", 404, "manifest_missing"),
    ("replace_payload", 409, "artifact_mismatch"),
    ("break_manifest", 409, "artifact_mismatch"),
])
def test_download_rejects_missing_or_inconsistent_artifacts(downloads, saved, damage, status, code):
    client, _ = downloads
    result, _, _, _ = saved
    path = Path(result["file_path"])
    if damage == "delete_payload":
        path.unlink()
    elif damage == "delete_manifest":
        manifest_path_for(path).unlink()
    elif damage == "replace_payload":
        path.write_bytes(b"different answer")
    else:
        manifest_path_for(path).write_text("not a manifest", encoding="utf-8")

    response = client.get(f"/download/{path.name}")

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code


def test_expired_file_cannot_be_downloaded_before_cleanup_runs(downloads, saved, monkeypatch):
    client, _ = downloads
    result, _, _, _ = saved
    monkeypatch.setattr("src.infra.saved_artifacts.time", lambda: result["artifact"]["expires_at"])

    response = client.get(f"/download/{result['artifact']['filename']}")

    assert response.status_code == 410
    assert response.json()["detail"]["code"] == "artifact_expired"
    assert Path(result["file_path"]).exists()


def test_download_read_failure_is_unverifiable_not_success(downloads, saved, monkeypatch):
    client, _ = downloads
    result, _, _, _ = saved
    saved_path = Path(result["file_path"])
    real_read_bytes = Path.read_bytes

    def unreadable_payload(path):
        if path == saved_path:
            raise OSError("storage unavailable")
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", unreadable_payload)
    response = client.get(f"/download/{saved_path.name}")

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "artifact_unverifiable"


def test_download_serves_verified_snapshot_even_if_cleanup_removes_the_path(downloads, saved, monkeypatch):
    client, _ = downloads
    result, text, _, _ = saved
    saved_path = Path(result["file_path"])
    real_read_bytes = Path.read_bytes

    def remove_after_read(path):
        payload = real_read_bytes(path)
        if path == saved_path:
            path.unlink()
        return payload

    monkeypatch.setattr(Path, "read_bytes", remove_after_read)
    response = client.get(f"/download/{saved_path.name}")

    assert response.status_code == 200
    assert response.content == text.encode("utf-8-sig")
    assert not saved_path.exists()


def test_same_operation_retry_retains_the_download_url(downloads, saved):
    client, _ = downloads
    result, text, operation, save = saved
    first = client.get(f"/download/{result['artifact']['filename']}")

    repeated = save(text, operation=operation)
    second = client.get(f"/download/{repeated['artifact']['filename']}")

    assert repeated["artifact"] == result["artifact"]
    assert first.status_code == second.status_code == 200
    assert first.content == second.content == text.encode("utf-8-sig")
