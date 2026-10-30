"""Production UI assets must stay inside /app and leave the API usable."""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.app.web.app import create_app
from src.infra.settings import AppSettings


@pytest.fixture
def build_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    from src.app.web import static_ui

    path = tmp_path / "frontend" / "dist"
    monkeypatch.setattr(static_ui, "get_frontend_dist_path", lambda: path)
    return path


def test_production_build_serves_index_and_assets_without_catching_api_routes(build_path: Path) -> None:
    assets = build_path / "assets"
    assets.mkdir(parents=True)
    index = '<!doctype html><script type="module" src="/app/assets/main.js"></script>'
    (build_path / "index.html").write_text(index, encoding="utf-8")
    (assets / "main.js").write_text('document.title = "DocuMate";', encoding="utf-8")
    app = create_app()
    app.state.settings = AppSettings(_env_file=None, DOCLING_ENABLED=False)
    # This test exercises routing and files only; the agent lifespan is not started.
    client = TestClient(app)

    redirect = client.get("/app", follow_redirects=False)
    assert redirect.status_code == 307
    assert redirect.headers["location"] == "http://testserver/app/"
    response = client.get("/app/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.text == index
    asset = client.get("/app/assets/main.js")
    assert asset.status_code == 200
    assert asset.text == 'document.title = "DocuMate";'
    assert client.get("/").json() == {"message": "Hello World"}
    assert client.get("/client-config").json()["upload"]["accepted_suffixes"] == [".ipynb", ".py"]
    assert client.post("/agent/stream", json={}).status_code == 422
    assert client.get("/app/assets/missing.js").status_code == 404
    assert client.get("/app/unknown-page").status_code == 404
    assert client.get("/unknown-api").status_code == 404


@pytest.mark.parametrize("empty_directory", [False, True])
def test_missing_build_explains_setup_without_blocking_the_api(build_path: Path, empty_directory: bool) -> None:
    if empty_directory:
        build_path.mkdir(parents=True)
    app = create_app()
    client = TestClient(app)

    assert client.get("/app", follow_redirects=False).status_code == 307
    response = client.get("/app/")
    assert response.status_code == 503
    assert "npm --prefix frontend ci" in response.text
    assert "npm --prefix frontend run build" in response.text
    assert client.get("/").json() == {"message": "Hello World"}
    assert client.get("/app/assets/missing.js").status_code == 404
    assert client.get("/unknown-api").status_code == 404
