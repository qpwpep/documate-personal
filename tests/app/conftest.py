"""Browser component boundary for AppTest, which does not execute JavaScript."""
from __future__ import annotations

import json

import pytest


class BrowserStorage:
    def __init__(self) -> None:
        self.snapshot: str | None = None

    def render(self, *, data: dict[str, object], **_kwargs: object) -> dict[str, object]:
        if data.get("restore"):
            return {"restore": {"snapshot": self.snapshot, "error": None}}
        snapshot = data.get("snapshot")
        if isinstance(snapshot, str):
            self.snapshot = snapshot
        return {}

    def type_draft(self, conversation_id: str, text: str) -> None:
        # Native typing reaches browser storage without submitting the composer
        # or triggering a Python rerun. Real JavaScript is checked in a browser.
        if self.snapshot is None:
            return
        snapshot = json.loads(self.snapshot)
        conversation = next(item for item in snapshot["conversations"] if item["id"] == conversation_id)
        conversation["draft"]["text"] = text
        self.snapshot = json.dumps(snapshot, ensure_ascii=False)


@pytest.fixture
def browser_storage(monkeypatch):
    from src.app.web import streamlit_browser_state

    storage = BrowserStorage()
    monkeypatch.setattr(streamlit_browser_state, "_bridge", storage.render)
    return storage
