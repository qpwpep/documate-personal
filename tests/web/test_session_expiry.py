"""Session access applies the same idle TTL before the next cleanup sweep."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from src.app.agent_manager import AgentFlowManager
from src.app.answer_exports import AnswerExportService
from src.app.web.routes import router
from src.app.web.session_store import InMemorySessionStore, SessionUnavailable
from src.app.web.upload_service import UploadService
from src.core.answer_schema import finalize_answer, text_document
from src.infra.settings import AppSettings
from tests.core.helpers import SequentialGraph


@dataclass
class MonotonicClock:
    now: float = 100

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> MonotonicClock:
    clock = MonotonicClock()
    # Keep HTTP/event-loop clocks real while controlling the store's idle time.
    monkeypatch.setattr("src.app.web.session_store.time", clock)
    return clock


@pytest.fixture
def store(clock: MonotonicClock) -> Iterator[InMemorySessionStore]:
    settings = AppSettings(
        _env_file=None, openai_api_key="test-key", tavily_api_key="test-key",
        session_ttl_seconds=10, session_cleanup_interval_seconds=60,
    )
    store = InMemorySessionStore(
        settings=settings,
        agent_factory=lambda: AgentFlowManager(settings=settings, graph=SequentialGraph()),
    )
    yield store
    store.close_all()


def test_idle_session_is_recreated_before_next_cleanup(
    store: InMemorySessionStore, clock: MonotonicClock,
) -> None:
    original = store.get_or_create("session-a")
    original.memory_summary = "Expired conversation"
    epoch = original.session.upload_epoch

    clock.now = 111
    replacement = store.get_or_create("session-a")

    assert replacement is not original
    assert replacement.session.upload_epoch != epoch
    assert replacement.memory_summary is None
    assert original.memory_summary is None


def test_existing_only_access_does_not_recreate_expired_session(
    store: InMemorySessionStore, clock: MonotonicClock,
) -> None:
    original = store.get_or_create("session-a")
    original.memory_summary = "Expired conversation"
    clock.now = 111

    with pytest.raises(SessionUnavailable) as failure:
        with store.locked_existing_session("session-a"):
            pytest.fail("An expired session must not be returned")

    assert failure.value.expired
    assert "session-a" not in store.active_session_ids()
    assert original.memory_summary is None


def test_live_access_renews_the_idle_deadline(
    store: InMemorySessionStore, clock: MonotonicClock,
) -> None:
    original = store.get_or_create("session-a")
    clock.now = 109
    with store.locked_existing_session("session-a") as entry:
        assert entry.agent is original

    clock.now = 118
    assert store.get_or_create("session-a") is original


def test_active_request_survives_ttl_and_renews_on_completion(
    store: InMemorySessionStore, clock: MonotonicClock,
) -> None:
    with store.locked_session("session-a") as (entry, _wait_ms):
        original = entry.agent
        original.memory_summary = "Active conversation"
        clock.now = 111
        assert store.get_or_create("session-a") is original
        clock.now = 171
        assert store.cleanup_expired(now=clock.now, ttl_seconds=10) == 0
        assert original.memory_summary == "Active conversation"

    clock.now = 181
    assert store.cleanup_expired(now=clock.now, ttl_seconds=10) == 0
    clock.now = 182
    assert store.cleanup_expired(now=clock.now, ttl_seconds=10) == 1
    assert original.memory_summary is None


def test_pinned_request_is_protected_before_acquiring_its_lock(
    store: InMemorySessionStore, clock: MonotonicClock,
) -> None:
    entry = store.get_or_create_entry("session-a", activate_request=True)
    assert not entry.request_lock.locked()

    clock.now = 111
    assert store.get_or_create("session-a") is entry.agent
    clock.now = 171
    assert store.cleanup_expired(now=clock.now, ttl_seconds=10) == 0
    with store.locked_existing_session("session-a") as current:
        assert current is entry


@pytest.mark.parametrize("first_access", ["manifest", "save"])
def test_manifest_refresh_cannot_revive_expired_answer_history(
    store: InMemorySessionStore, clock: MonotonicClock, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch, first_access: str,
) -> None:
    monkeypatch.setattr("src.infra.runtime_paths.get_project_root_path", lambda: tmp_path)
    app = FastAPI()
    app.state.upload_service = UploadService(settings=store.settings, session_store=store)
    app.state.answer_export_service = AnswerExportService(settings=store.settings, session_store=store)
    app.include_router(router)

    with TestClient(app) as client:
        manifest_url = "/sessions/session-a/uploads"
        initial = client.get(manifest_url)
        assert initial.status_code == 200
        original = store.get_or_create("session-a")
        answer_id = original.session.register_answer(finalize_answer(text_document("Original answer"), []))
        save_url = f"/sessions/session-a/answers/{answer_id}/save"
        clock.now = 111

        if first_access == "manifest":
            refreshed = client.get(manifest_url)
            saved = client.post(save_url)
        else:
            saved = client.post(save_url)
            refreshed = client.get(manifest_url)

    assert refreshed.status_code == 200
    assert refreshed.json()["epoch"] != initial.json()["epoch"]
    assert saved.status_code == (404 if first_access == "manifest" else 410)
    assert saved.json()["detail"]["code"] == (
        "answer_unavailable" if first_access == "manifest" else "session_expired"
    )
    assert original.session.get_answer(answer_id) is None
    assert not list(tmp_path.rglob("*.txt"))
