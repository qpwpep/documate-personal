from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from src.app.agent_manager import AgentFlowManager
from src.core.contracts import SessionMetadata
from src.infra.logging_utils import log_event
from src.runtime.progress import ProgressEmitter
from src.infra.settings import AppSettings


logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    agent: AgentFlowManager
    last_accessed_monotonic: float
    request_lock: Lock = field(default_factory=Lock)
    active_request_count: int = 0


class InMemorySessionStore:
    def __init__(
        self,
        settings: AppSettings,
        agent_factory: Callable[[], AgentFlowManager],
    ) -> None:
        self.settings = settings
        self._agent_factory = agent_factory
        self.active_agents: dict[str, SessionEntry] = {}
        self._lock = Lock()
        self._last_cleanup_monotonic = 0.0

    def _pop_session_entry(self, session_id: str) -> SessionEntry | None:
        entry = self.active_agents.pop(session_id, None)
        if entry is None:
            return None

        try:
            with entry.request_lock:
                entry.agent.close()
        except Exception as exc:
            log_event(
                logger,
                logging.WARNING,
                "session_close_error",
                session_id=session_id[:8],
                error=exc,
            )
        return entry

    def cleanup_expired(self, *, now: float, ttl_seconds: int) -> int:
        expired_session_ids = [
            sid
            for sid, entry in self.active_agents.items()
            if entry.active_request_count <= 0
            and not entry.request_lock.locked()
            and now - entry.last_accessed_monotonic > ttl_seconds
        ]
        for sid in expired_session_ids:
            self._pop_session_entry(sid)
        return len(expired_session_ids)

    def evict_lru_if_needed(self, max_active_sessions: int) -> int:
        evicted_count = 0
        while len(self.active_agents) > max_active_sessions:
            evictable_sessions = [
                item
                for item in self.active_agents.items()
                if item[1].active_request_count <= 0 and not item[1].request_lock.locked()
            ]
            if not evictable_sessions:
                break
            lru_session_id = min(
                evictable_sessions,
                key=lambda item: item[1].last_accessed_monotonic,
            )[0]
            self._pop_session_entry(lru_session_id)
            evicted_count += 1
        return evicted_count

    def maybe_run_cleanup(self, *, now: float) -> int:
        if now - self._last_cleanup_monotonic < self.settings.session_cleanup_interval_seconds:
            return 0

        expired_removed_count = self.cleanup_expired(
            now=now,
            ttl_seconds=self.settings.session_ttl_seconds,
        )
        self._last_cleanup_monotonic = now
        return expired_removed_count

    def get_or_create_entry(self, session_id: str) -> SessionEntry:
        now = time.monotonic()
        with self._lock:
            expired_removed_count = self.maybe_run_cleanup(now=now)
            existing_entry = self.active_agents.get(session_id)
            if existing_entry is not None:
                existing_entry.last_accessed_monotonic = now
                log_event(
                    logger,
                    logging.INFO,
                    "session_cache_event",
                    session_id=session_id[:8],
                    session_hit=True,
                    session_recreated=False,
                    expired_removed_count=expired_removed_count,
                    lru_evicted_count=0,
                    active_session_count=len(self.active_agents),
                )
                return existing_entry

            recreated_agent = self._agent_factory()
            self.active_agents[session_id] = SessionEntry(
                agent=recreated_agent,
                last_accessed_monotonic=now,
            )
            lru_evicted_count = self.evict_lru_if_needed(self.settings.max_active_sessions)
            log_event(
                logger,
                logging.INFO,
                "session_cache_event",
                session_id=session_id[:8],
                session_hit=False,
                session_recreated=True,
                expired_removed_count=expired_removed_count,
                lru_evicted_count=lru_evicted_count,
                active_session_count=len(self.active_agents),
            )
            return self.active_agents[session_id]

    def get_or_create(self, session_id: str) -> AgentFlowManager:
        return self.get_or_create_entry(session_id).agent

    def run_session_request(
        self,
        *,
        session_id: str,
        session_metadata: SessionMetadata,
        user_input: str,
        upload_file_path: str | None = None,
        progress_emitter: ProgressEmitter | None = None,
    ) -> tuple[AgentFlowManager, dict[str, Any], int]:
        entry = self.get_or_create_entry(session_id)
        with self._lock:
            active_entry = self.active_agents.get(session_id)
            if active_entry is not None:
                entry = active_entry
            entry.active_request_count += 1
            entry.last_accessed_monotonic = time.monotonic()

        lock_started = time.monotonic()
        try:
            with entry.request_lock:
                session_lock_wait_ms = int((time.monotonic() - lock_started) * 1000)
                agent_manager = entry.agent
                agent_manager.set_session_metadata(session_metadata)
                agent_answer = agent_manager.run_agent_flow(
                    user_input,
                    upload_file_path,
                    progress_emitter=progress_emitter,
                )
                return agent_manager, agent_answer, session_lock_wait_ms
        finally:
            finished_at = time.monotonic()
            with self._lock:
                active_entry = self.active_agents.get(session_id)
                if active_entry is entry:
                    active_entry.last_accessed_monotonic = finished_at
                    active_entry.active_request_count = max(0, active_entry.active_request_count - 1)

    def active_session_ids(self) -> set[str]:
        with self._lock:
            return set(self.active_agents.keys())

    def close_all(self) -> None:
        with self._lock:
            session_ids = list(self.active_agents.keys())
        for session_id in session_ids:
            self._pop_session_entry(session_id)
