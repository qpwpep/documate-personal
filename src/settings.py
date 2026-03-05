from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    tavily_api_key: str | None = Field(default=None, alias="TAVILY_API_KEY")

    chat_model: str = Field(default="gpt-5-mini", alias="CHAT_MODEL")
    planner_model: str = Field(default="gpt-5-nano", alias="PLANNER_MODEL")
    summary_model: str = Field(default="gpt-5-mini", alias="SUMMARY_MODEL")

    verbose: bool = Field(default=True, alias="VERBOSE")
    fastapi_url: str = Field(default="http://localhost:8000", alias="FASTAPI_URL")
    session_ttl_seconds: int = Field(default=1800, alias="SESSION_TTL_SECONDS", ge=1)
    max_active_sessions: int = Field(default=200, alias="MAX_ACTIVE_SESSIONS", ge=1)
    session_cleanup_interval_seconds: int = Field(
        default=60,
        alias="SESSION_CLEANUP_INTERVAL_SECONDS",
        ge=1,
    )
    generated_file_ttl_seconds: int = Field(
        default=86400,
        alias="GENERATED_FILE_TTL_SECONDS",
        ge=1,
    )
    file_cleanup_interval_seconds: int = Field(
        default=60,
        alias="FILE_CLEANUP_INTERVAL_SECONDS",
        ge=1,
    )

    slack_bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_default_dm_email: str | None = Field(default=None, alias="SLACK_DEFAULT_DM_EMAIL")
    slack_default_user_id: str | None = Field(default=None, alias="SLACK_DEFAULT_USER_ID")


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def validate_required_keys(settings: AppSettings, context: str) -> None:
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.tavily_api_key:
        missing.append("TAVILY_API_KEY")

    if missing:
        missing_keys = ", ".join(missing)
        raise ConfigurationError(f"[{context}] Missing required environment variables: {missing_keys}")
