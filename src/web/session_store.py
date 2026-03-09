from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock

from ..agent_manager import AgentFlowManager
from ..logging_utils import log_event
from ..settings import AppSettings


logger = logging.getLogger(__name__)


@dataclass
class SessionEntry:
    agent: AgentFlowManager
    last_accessed_monotonic: float
    created_monotonic: float


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
            if now - entry.last_accessed_monotonic > ttl_seconds
        ]
        for sid in expired_session_ids:
            self._pop_session_entry(sid)
        return len(expired_session_ids)

    def evict_lru_if_needed(self, max_active_sessions: int) -> int:
        evicted_count = 0
        while len(self.active_agents) > max_active_sessions:
            lru_session_id = min(
                self.active_agents.items(),
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

    def get_or_create(self, session_id: str) -> AgentFlowManager:
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
                return existing_entry.agent

            recreated_agent = self._agent_factory()
            self.active_agents[session_id] = SessionEntry(
                agent=recreated_agent,
                created_monotonic=now,
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
            return self.active_agents[session_id].agent

    def active_session_ids(self) -> set[str]:
        with self._lock:
            return set(self.active_agents.keys())

    def close_all(self) -> None:
        with self._lock:
            session_ids = list(self.active_agents.keys())
        for session_id in session_ids:
            self._pop_session_entry(session_id)
