"""Browser limits come from the server's public configuration contract."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.web.routes import router
from src.core.conversation_memory import DEFAULT_QUERY_MAX_CHARS, DEFAULT_QUERY_MAX_UTF8_BYTES
from src.infra.settings import AppSettings


@pytest.mark.parametrize("docling_enabled,accepted_suffixes", [
    (False, [".ipynb", ".py"]),
    (True, [".bmp", ".docx", ".ipynb", ".jpeg", ".jpg", ".pdf", ".png", ".py", ".tif", ".tiff", ".webp"]),
])
def test_client_config_exposes_effective_limits_without_private_settings(
    docling_enabled: bool, accepted_suffixes: list[str],
) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = AppSettings(
        _env_file=None,
        DOCLING_ENABLED=docling_enabled,
        UPLOAD_MAX_FILES=3,
        UPLOAD_MAX_FILE_MIB=2,
        UPLOAD_MAX_TOTAL_MIB=6,
        OPENAI_API_KEY="private-openai-test-key",
        SLACK_BOT_TOKEN="xoxb-private-test-token",
    )

    with TestClient(app) as client:
        response = client.get("/client-config")

    assert response.status_code == 200
    assert response.json() == {
        "upload": {
            "accepted_suffixes": accepted_suffixes,
            "max_files": 3,
            "max_file_bytes": 2_097_152,
            "max_total_bytes": 6_291_456,
        },
        "query": {
            "max_chars": DEFAULT_QUERY_MAX_CHARS,
            "max_utf8_bytes": DEFAULT_QUERY_MAX_UTF8_BYTES,
        },
    }
    assert "private-openai-test-key" not in response.text
    assert "xoxb-private-test-token" not in response.text
