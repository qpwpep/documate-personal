from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from ..agent_manager import AgentFlowManager
from ..logging_utils import configure_logging, log_event
from ..runtime_encoding import ensure_utf8_stdio
from ..settings import ConfigurationError, get_settings, validate_required_keys
from .cleanup import RuntimeCleaner
from .routes import router
from .session_store import InMemorySessionStore


ensure_utf8_stdio()
configure_logging()
logger = logging.getLogger("uvicorn")


def _warn_if_utf8_mode_disabled() -> None:
    if sys.flags.utf8_mode == 1:
        return

    log_event(
        logger,
        logging.WARNING,
        "utf8_mode_disabled",
        suggested_command=(
            "uv run python -X utf8 -m uvicorn src.web.app:app --host 0.0.0.0 --port 8000"
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = get_settings()
        validate_required_keys(settings, context="fastapi_startup")
    except ConfigurationError as exc:
        logger.error(str(exc))
        raise RuntimeError(str(exc)) from exc

    session_store = InMemorySessionStore(
        settings=settings,
        agent_factory=lambda: AgentFlowManager(settings=settings),
    )
    runtime_cleaner = RuntimeCleaner(settings=settings, session_store=session_store)
    app.state.settings = settings
    app.state.session_store = session_store
    app.state.runtime_cleaner = runtime_cleaner
    runtime_cleaner.run_once(force=True, current_session_id=None)
    yield
    session_store.close_all()


def create_app() -> FastAPI:
    _warn_if_utf8_mode_disabled()
    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        log_event(
            logger,
            logging.INFO,
            "incoming_request",
            request_id=request_id[:8],
            method=request.method,
            path=request.url.path,
        )
        response = await call_next(request)
        log_event(
            logger,
            logging.INFO,
            "request_finished",
            request_id=request_id[:8],
            status_code=response.status_code,
        )
        return response

    app.include_router(router)
    return app


app = create_app()
