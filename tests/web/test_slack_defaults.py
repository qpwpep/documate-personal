"""The UI reads only the default destination selected by the sending server."""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.app.web.routes import router
from src.infra.settings import AppSettings


@pytest.mark.parametrize("user_id,email,expected_selector,expected_failure", [
    (None, None, None, None),
    (" U0123456789 ", None, {"kind": "user", "value": "U0123456789"}, None),
    (None, " recipient@example.com ", {"kind": "email", "value": "recipient@example.com"}, None),
    ("invalid-user", None, None, "configuration_error"),
    ("U0123456789", "recipient@example.com", None, "configuration_error"),
])
def test_default_destination_uses_server_configuration_without_exposing_credentials(
    user_id, email, expected_selector, expected_failure,
) -> None:
    app = FastAPI()
    app.include_router(router)
    app.state.settings = AppSettings(
        _env_file=None,
        SLACK_DEFAULT_USER_ID=user_id,
        SLACK_DEFAULT_DM_EMAIL=email,
        SLACK_BOT_TOKEN="xoxb-private-test-token",
    )

    with TestClient(app) as client:
        response = client.get("/slack/default-recipient")

    assert response.status_code == 200
    data = response.json()
    assert set(data) == {"selector", "failure"}
    assert data["selector"] == expected_selector
    assert (data["failure"]["code"] if data["failure"] else None) == expected_failure
    assert "xoxb-private-test-token" not in response.text
