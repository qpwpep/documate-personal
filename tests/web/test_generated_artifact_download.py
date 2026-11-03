from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.app.web.routes import router
from src.core.save_contract import SaveOperation
from src.infra.saved_artifacts import save_artifact
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


def test_download_returns_the_actual_verified_bytes_and_operation_binding(downloads, saved):
    client, _ = downloads
    result, text, operation, _ = saved
    artifact = result.artifact

    response = client.get(f"/download/{artifact.filename}")

    assert response.status_code == 200
    assert response.content == text.encode("utf-8-sig")
    assert response.headers["x-save-binding-sha256"] == operation.binding_sha256
    assert response.headers["x-artifact-id"] == artifact.artifact_id
    assert response.headers["etag"] == f'"{artifact.sha256}"'
    assert unquote(response.headers["content-disposition"]) == (
        "attachment; filename*=utf-8''DocuMate - 사용자가 요청한 답변.txt"
    )


@pytest.mark.parametrize("text,download_name", [
    ("## pandas.merge 기본 사용법 [1]\n본문", "DocuMate - pandas.merge 기본 사용법.txt"),
    ('# 한글 / 경로: "답변"?\n본문', "DocuMate - 한글 경로 답변.txt"),
    ("```python\nprint('hello')\n```", "DocuMate 답변.txt"),
])
def test_download_uses_a_readable_portable_name_without_renaming_the_artifact(
    downloads, text, download_name,
):
    client, root = downloads
    operation = SaveOperation.for_text(
        text, operation_id="save-title", session_id="session-title", request_id="request-title",
        contract_revision=1, target_kind="compose", source_hash="a" * 64, answer_hash="b" * 64,
    )
    manifest, path = save_artifact(root, text, operation, ttl_seconds=60)

    response = client.get(f"/download/{manifest.artifact.filename}")

    assert response.status_code == 200
    assert unquote(response.headers["content-disposition"]) == (
        f"attachment; filename*=utf-8''{download_name}"
    )
    assert path.name == manifest.artifact.filename
    assert path.name.startswith("response_")
    assert path.read_bytes() == response.content == text.encode("utf-8-sig")


@pytest.mark.parametrize("damage,status,code", [
    ("delete_payload", 404, "artifact_missing"),
    ("replace_payload", 409, "artifact_mismatch"),
])
def test_download_rejects_missing_or_inconsistent_artifacts(downloads, saved, damage, status, code):
    client, _ = downloads
    result, _, _, _ = saved
    path = result.file_path
    if damage == "delete_payload":
        path.unlink()
    else:
        path.write_bytes(b"different answer")

    response = client.get(f"/download/{path.name}")

    assert response.status_code == status
    assert response.json()["detail"]["code"] == code
