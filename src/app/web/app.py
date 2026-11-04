from __future__ import annotations

import logging
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from pydantic.json_schema import models_json_schema

from src.app.agent_manager import AgentFlowManager
from src.app.answer_exports import AnswerExportService
from src.infra.logging_utils import configure_logging, log_event
from src.infra.runtime_encoding import ensure_utf8_stdio
from src.infra.settings import ConfigurationError, get_settings, validate_required_keys
from src.infra.llm import validate_output_schemas
from src.app.web.agent_request_service import AgentRequestService
from src.app.web.cleanup import RuntimeCleaner
from src.app.web.routes import router
from src.app.transport_contracts import AgentResponse, HTTPResponseSchema
from src.app.web.schemas import ClientConfig, StagedUpload
from src.core.contracts.outcome import AnswerExportResult
from src.core.slack_contract import SlackDefault
from src.core.uploads import UploadManifest
from src.app.web.session_store import InMemorySessionStore
from src.app.web.static_ui import mount_react_ui


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
            "uv run python -X utf8 -m uvicorn src.app.web.app:app --host 0.0.0.0 --port 8000"
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        settings = get_settings()
        validate_required_keys(settings, context="fastapi_startup")
        validate_output_schemas()
    except ConfigurationError as exc:
        logger.error(str(exc))
        raise RuntimeError(str(exc)) from exc

    log_event(
        logger,
        logging.INFO,
        "fastapi_runtime_settings",
        **settings.fastapi_runtime_log_fields(),
    )

    session_store = InMemorySessionStore(
        settings=settings,
        agent_factory=lambda: AgentFlowManager(settings=settings),
    )
    runtime_cleaner = RuntimeCleaner(settings=settings, session_store=session_store)
    agent_request_service = AgentRequestService(
        runtime_cleaner=runtime_cleaner,
        session_store=session_store,
    )
    app.state.settings = settings
    app.state.session_store = session_store
    app.state.runtime_cleaner = runtime_cleaner
    app.state.agent_request_service = agent_request_service
    app.state.answer_export_service = AnswerExportService(settings=settings, session_store=session_store)
    from src.app.web.upload_service import UploadService
    from src.infra.docling_runner import DoclingRunner
    converter = DoclingRunner(cache_max_bytes=settings.document_cache_bytes_per_store(),
                             cache_ttl_seconds=settings.document_cache_ttl_seconds)
    app.state.document_converter = converter
    app.state.upload_service = UploadService(settings=settings, session_store=session_store, converter=converter)
    runtime_cleaner.run_once(force=True, current_session_id=None)
    try:
        yield
    finally:
        converter.close()
        session_store.close_all()


def create_app() -> FastAPI:
    _warn_if_utf8_mode_disabled()
    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def add_execution_id(request: Request, call_next):
        execution_id = str(uuid.uuid4())
        request.state.execution_id = execution_id
        log_event(
            logger,
            logging.INFO,
            "incoming_request",
            execution_id=execution_id,
            method=request.method,
            path=request.url.path,
        )
        response = await call_next(request)
        response.headers["X-Execution-ID"] = execution_id
        log_event(
            logger,
            logging.INFO,
            "request_finished",
            execution_id=execution_id,
            status_code=response.status_code,
        )
        return response

    app.include_router(router)
    mount_react_ui(app)

    default_openapi = app.openapi

    def stream_openapi():
        if app.openapi_schema is None:
            schema = default_openapi()
            # HTTP outputs include model defaults, unlike model/LLM input schemas.
            # SSE also needs its final-response model registered explicitly.
            _, response_schemas = models_json_schema(
                [(model, "serialization") for model in (
                    AgentResponse, AnswerExportResult, SlackDefault,
                    UploadManifest, StagedUpload, ClientConfig,
                )],
                ref_template="#/components/schemas/{model}",
                schema_generator=HTTPResponseSchema,
            )
            components = schema.setdefault("components", {}).setdefault("schemas", {})
            components.update(response_schemas["$defs"])
        return app.openapi_schema

    app.openapi = stream_openapi
    return app


app = create_app()
