"""Incomplete optional Slack settings do not reject a normal chat request."""
from __future__ import annotations

import json

import pytest

from tests.web.test_multi_upload_api import api, context, manifest


@pytest.mark.parametrize("recipient", [
    {"kind": "channel", "value": ""},
    {"kind": "user", "value": "not-a-user-id"},
    {"kind": "email", "value": "invalid-email"},
])
def test_normal_question_accepts_incomplete_slack_settings(api, recipient):
    response = api.client.post("/agent/stream", json={
        "query": "안녕하세요", "session_id": "session-a",
        "uploads": context(manifest(api)), "slack_recipient": recipient,
    })

    assert response.status_code == 200, response.text
    final_frame = next(frame for frame in response.text.split("\n\n")
                       if frame.startswith("event: final_response"))
    result = json.loads(next(line[6:] for line in final_frame.splitlines() if line.startswith("data: ")))
    assert result["status"] == "completed"
    assert result["response"]["actions"] == []
