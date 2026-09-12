from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import dotenv_values
from pydantic import BaseModel, Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

from src.core.conversation_memory import ConversationMemoryPolicy
from src.infra.runtime_paths import get_benchmark_config_path, get_env_file_path


DEFAULT_BENCHMARK_CONFIG_PATH = get_benchmark_config_path()
ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
# Exact names verified in the GPT-5.6 model docs; other names defer to the provider.
_PLANNER_MODEL_REASONING_EFFORTS: dict[str, tuple[ReasoningEffort, ...]] = dict.fromkeys(
    ("gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol", "gpt-5.6"),
    ("none", "low", "medium", "high", "xhigh", "max"),
)
EnvExampleGroup = Literal[
    "required_secrets", "models", "planning_summary", "synthesis", "search",
    "application_settings", "sessions", "files", "memory", "slack",
    "benchmark", "benchmark_slack",
]


@dataclass(frozen=True)
class EnvVarSpec:
    env_name: str
    field_name: str | None
    default: str | int | float | bool | None
    description: str
    example: str | int | float | bool | None = None
    section: Literal["app", "benchmark"] = "app"
    config_runtime_key: str | None = None
    sync_notes: tuple[str, ...] = ()
    example_group: EnvExampleGroup = "application_settings"


APP_ENV_SPECS = (
    EnvVarSpec(
        "OPENAI_API_KEY",
        "openai_api_key",
        None,
        "OpenAI 호출과 임베딩 생성에 필요",
        example_group="required_secrets",
    ),
    EnvVarSpec(
        "TAVILY_API_KEY",
        "tavily_api_key",
        None,
        "공식 문서 검색에 필요",
        example_group="required_secrets",
    ),
    EnvVarSpec("CHAT_MODEL", "chat_model", "gpt-5.6-luna", "synthesis 모델 기본값", example="gpt-5.6-luna", example_group="models"),
    EnvVarSpec("PLANNER_MODEL", "planner_model", "gpt-5.6-luna", "planner 모델 기본값", example="gpt-5.6-luna", example_group="models"),
    EnvVarSpec("SUMMARY_MODEL", "summary_model", "gpt-5.6-luna", "session summary 모델 기본값", example="gpt-5.6-luna", example_group="models"),
    EnvVarSpec("SUMMARY_MAX_TOKENS", "summary_max_tokens", 1024, "요약 LLM 생성 토큰 상한; 저장 요약 예산과 독립", example=1024, example_group="planning_summary"),
    EnvVarSpec("PLANNER_MAX_TOKENS", "planner_max_tokens", 1920, "planner structured output 최대 토큰", example=1920, example_group="planning_summary"),
    EnvVarSpec(
        "PLANNER_REASONING_EFFORT",
        "planner_reasoning_effort",
        None,
        "planner reasoning effort override (none/minimal/low/medium/high/xhigh/max, 빈 값이면 모델 기본값, none은 명시 override)",
        example="high",
        sync_notes=tuple(
            f"{model}: {', '.join(efforts)}"
            for model, efforts in _PLANNER_MODEL_REASONING_EFFORTS.items()
        ),
        example_group="planning_summary",
    ),
    EnvVarSpec("DOCS_SEARCH_TIMEOUT_SECONDS", "docs_search_timeout_seconds", 5, "Tavily 요청별 timeout", example=5, example_group="search"),
    EnvVarSpec("SYNTHESIS_TIMEOUT_SECONDS", "synthesis_timeout_seconds", 20, "synthesis provider 요청 timeout", example=20, example_group="synthesis"),
    EnvVarSpec("SYNTHESIS_USE_RESPONSES_API", "synthesis_use_responses_api", False, "synthesis Responses API 사용 여부", example=False, example_group="synthesis"),
    EnvVarSpec("SYNTHESIS_MAX_RETRIES", "synthesis_max_retries", 0, "synthesis provider SDK 재시도 횟수", example=0, example_group="synthesis"),
    EnvVarSpec("SYNTHESIS_MAX_TOKENS", "synthesis_max_tokens", 4096, "일반 synthesis 생성 토큰 상한", example=4096, example_group="synthesis"),
    EnvVarSpec("SYNTHESIS_COMPACT_MAX_TOKENS", "synthesis_compact_max_tokens", 960, "timeout 복구용 synthesis 생성 토큰 상한; 일반 상한과 독립", example=960, example_group="synthesis"),
    EnvVarSpec(
        "SYNTHESIS_PROMPT_SNIPPET_CHARS",
        "synthesis_prompt_snippet_chars",
        1800,
        "evidence snippet 길이 제한",
        example=1800,
        example_group="synthesis",
    ),
    EnvVarSpec(
        "SYNTHESIS_COMPACT_PROMPT_SNIPPET_CHARS",
        "synthesis_compact_prompt_snippet_chars",
        900,
        "timeout 복구용 evidence snippet 길이 제한; 일반 설정보다 확대하지 않음",
        example=900,
        example_group="synthesis",
    ),
    EnvVarSpec(
        "SYNTHESIS_REASONING_EFFORT",
        "synthesis_reasoning_effort",
        None,
        "synthesis reasoning effort override (none/minimal/low/medium/high/xhigh/max, 빈 값이면 모델 기본값, none은 명시 override)",
        example="low",
        sync_notes=(
            "gpt-5.6-luna: none, low, medium, high, xhigh, max",
            "gpt-5-nano: minimal, low, medium, high",
        ),
        example_group="synthesis",
    ),
    EnvVarSpec("VERBOSE", "verbose", True, "에이전트 런타임 상세 로그 출력", example=True),
    EnvVarSpec("FASTAPI_URL", "fastapi_url", "http://127.0.0.1:8000", "Streamlit이 호출하는 API 주소", example="http://127.0.0.1:8000"),
    EnvVarSpec("SESSION_TTL_SECONDS", "session_ttl_seconds", 1800, "세션 TTL", example=1800, example_group="sessions"),
    EnvVarSpec("MAX_ACTIVE_SESSIONS", "max_active_sessions", 200, "최대 활성 세션 수", example=200, example_group="sessions"),
    EnvVarSpec("UPLOAD_MAX_FILES", "upload_max_files", 10, "세션별 활성 업로드 파일 개수", example=10, example_group="sessions"),
    EnvVarSpec("UPLOAD_MAX_FILE_MIB", "upload_max_file_mib", 10, "업로드 파일당 크기 상한 (MiB)", example=10, example_group="sessions"),
    EnvVarSpec("UPLOAD_MAX_TOTAL_MIB", "upload_max_total_mib", 50, "세션별 활성 업로드 합계 상한 (MiB)", example=50, example_group="sessions"),
    EnvVarSpec(
        "SESSION_CLEANUP_INTERVAL_SECONDS",
        "session_cleanup_interval_seconds",
        60,
        "세션 정리 주기",
        example=60,
        example_group="sessions",
    ),
    EnvVarSpec(
        "GENERATED_FILE_TTL_SECONDS",
        "generated_file_ttl_seconds",
        86400,
        "`save_text` 결과 파일 TTL",
        example=86400,
        example_group="files",
    ),
    EnvVarSpec(
        "FILE_CLEANUP_INTERVAL_SECONDS",
        "file_cleanup_interval_seconds",
        60,
        "업로드/생성 파일 정리 주기",
        example=60,
        example_group="files",
    ),
    EnvVarSpec(
        "MEMORY_HIGH_WATER_TURNS",
        "memory_high_water_turns",
        8,
        "대화 compaction을 시작하는 Human turn high watermark",
        example=8,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_LOW_WATER_TURNS",
        "memory_low_water_turns",
        6,
        "compaction 후 Human turn low watermark",
        example=6,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_HIGH_WATER_TOKENS",
        "memory_high_water_tokens",
        32000,
        "대화 메모리 추정 token high watermark",
        example=32000,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_LOW_WATER_TOKENS",
        "memory_low_water_tokens",
        16000,
        "compaction 후 추정 token low watermark",
        example=16000,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_HIGH_WATER_BYTES",
        "memory_high_water_bytes",
        98304,
        "대화 메모리 UTF-8 직렬화 byte high watermark",
        example=98304,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_LOW_WATER_BYTES",
        "memory_low_water_bytes",
        49152,
        "compaction 후 UTF-8 직렬화 byte low watermark",
        example=49152,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_HIGH_WATER_MESSAGES",
        "memory_high_water_messages",
        18,
        "대화 메시지 수 high watermark",
        example=18,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_LOW_WATER_MESSAGES",
        "memory_low_water_messages",
        14,
        "compaction 후 메시지 수 low watermark",
        example=14,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_SUMMARY_MAX_TOKENS",
        "memory_summary_max_tokens",
        256,
        "저장할 rolling summary의 추정 token 상한; LLM 생성 예산과 독립",
        example=256,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_SUMMARY_MAX_BYTES",
        "memory_summary_max_bytes",
        4096,
        "rolling summary UTF-8 byte 상한",
        example=4096,
        example_group="memory",
    ),
    EnvVarSpec(
        "MEMORY_HARD_MAX_BYTES",
        "memory_hard_max_bytes",
        131072,
        "summary와 최근 메시지를 합친 durable snapshot 절대 byte 상한",
        example=131072,
        example_group="memory",
    ),
    EnvVarSpec(
        "SLACK_BOT_TOKEN",
        "slack_bot_token",
        None,
        "Slack 전송용 토큰",
        example_group="slack",
    ),
    EnvVarSpec(
        "SLACK_DEFAULT_DM_EMAIL",
        "slack_default_dm_email",
        None,
        "기본 DM 대상 이메일",
        example_group="slack",
    ),
    EnvVarSpec(
        "SLACK_DEFAULT_USER_ID",
        "slack_default_user_id",
        None,
        "기본 DM 대상 사용자",
        example_group="slack",
    ),
)

BENCHMARK_ENV_SPECS = (
    EnvVarSpec(
        "JUDGE_MODEL",
        None,
        "gpt-5.6-luna",
        "benchmark judge 모델 override",
        example="gpt-5.6-luna",
        section="benchmark",
        config_runtime_key="judge_model",
        example_group="benchmark",
    ),
    EnvVarSpec(
        "BENCHMARK_ENDPOINT",
        None,
        "http://127.0.0.1:8000",
        "benchmark 대상 FastAPI 주소 override",
        example="http://127.0.0.1:8000",
        section="benchmark",
        example_group="benchmark",
    ),
    EnvVarSpec(
        "BENCHMARK_JUDGE_ENABLED",
        None,
        True,
        "judge 사용 여부 override",
        example=True,
        section="benchmark",
        config_runtime_key="judge_enabled",
        example_group="benchmark",
    ),
    EnvVarSpec(
        "BENCHMARK_SLACK_ENABLED",
        None,
        False,
        "benchmark live Slack 전송 opt-in",
        example=False,
        section="benchmark",
        example_group="benchmark_slack",
    ),
    EnvVarSpec(
        "BENCHMARK_SLACK_CHANNEL_ID",
        None,
        None,
        "benchmark channel case 전송용 Slack channel id",
        example="C0123456789",
        section="benchmark",
        example_group="benchmark_slack",
    ),
    EnvVarSpec(
        "BENCHMARK_SLACK_USER_ID",
        None,
        None,
        "benchmark DM case 전송용 Slack user id",
        example="U0123456789",
        section="benchmark",
        example_group="benchmark_slack",
    ),
    EnvVarSpec(
        "BENCHMARK_SLACK_EMAIL",
        None,
        None,
        "benchmark DM case 전송용 Slack email",
        example="bench@example.com",
        section="benchmark",
        example_group="benchmark_slack",
    ),
)

APP_ENV_SPEC_BY_NAME = {spec.env_name: spec for spec in APP_ENV_SPECS}
BENCHMARK_ENV_SPEC_BY_NAME = {spec.env_name: spec for spec in BENCHMARK_ENV_SPECS}


class ConfigurationError(RuntimeError):
    """Raised when required runtime configuration is missing."""


class BenchmarkCLIEnvSettings(BaseModel):
    endpoint: str
    judge_model: str
    judge_enabled: bool
    live_slack_enabled: bool
    live_slack_channel_id: str | None = None
    live_slack_user_id: str | None = None
    live_slack_email: str | None = None


def _app_default(env_name: str) -> str | int | float | bool | None:
    return APP_ENV_SPEC_BY_NAME[env_name].default


def _normalize_env_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _str_to_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def _resolve_benchmark_env_value(
    env_name: str,
    *,
    dotenv_payload: dict[str, str | None],
    os_environ: dict[str, str],
    defaults: dict[str, str | int | float | bool | None],
) -> str | int | float | bool | None:
    if env_name in dotenv_payload:
        return dotenv_payload.get(env_name)
    if env_name in os_environ:
        return os_environ.get(env_name)
    return defaults.get(env_name)


def load_benchmark_env_defaults(config_path: Path = DEFAULT_BENCHMARK_CONFIG_PATH) -> dict[str, str | int | float | bool | None]:
    runtime: dict[str, str | int | float | bool | None] = {}
    if config_path.exists():
        payload = tomllib.loads(config_path.read_text(encoding="utf-8"))
        runtime = payload.get("runtime", {})

    defaults: dict[str, str | int | float | bool | None] = {}
    for spec in BENCHMARK_ENV_SPECS:
        if spec.config_runtime_key:
            defaults[spec.env_name] = runtime.get(spec.config_runtime_key, spec.default)
        else:
            defaults[spec.env_name] = spec.default
    return defaults


def load_benchmark_cli_env_settings(
    config_path: Path = DEFAULT_BENCHMARK_CONFIG_PATH,
    *,
    env_path: Path | None = None,
) -> BenchmarkCLIEnvSettings:
    defaults = load_benchmark_env_defaults(config_path)
    resolved_env_path = env_path or get_env_file_path()
    dotenv_payload = (
        {
            str(key): (_normalize_env_text(value) if value is not None else None)
            for key, value in dotenv_values(resolved_env_path).items()
        }
        if resolved_env_path.exists()
        else {}
    )

    endpoint = _normalize_env_text(
        _resolve_benchmark_env_value(
            "BENCHMARK_ENDPOINT",
            dotenv_payload=dotenv_payload,
            os_environ=os.environ,
            defaults=defaults,
        )
    ) or str(defaults["BENCHMARK_ENDPOINT"])
    judge_model = _normalize_env_text(
        _resolve_benchmark_env_value(
            "JUDGE_MODEL",
            dotenv_payload=dotenv_payload,
            os_environ=os.environ,
            defaults=defaults,
        )
    ) or str(defaults["JUDGE_MODEL"])
    judge_enabled = _str_to_bool(
        _normalize_env_text(
            _resolve_benchmark_env_value(
                "BENCHMARK_JUDGE_ENABLED",
                dotenv_payload=dotenv_payload,
                os_environ=os.environ,
                defaults=defaults,
            )
        ),
        bool(defaults["BENCHMARK_JUDGE_ENABLED"]),
    )
    live_slack_enabled = _str_to_bool(
        _normalize_env_text(
            _resolve_benchmark_env_value(
                "BENCHMARK_SLACK_ENABLED",
                dotenv_payload=dotenv_payload,
                os_environ=os.environ,
                defaults=defaults,
            )
        ),
        bool(defaults["BENCHMARK_SLACK_ENABLED"]),
    )

    return BenchmarkCLIEnvSettings(
        endpoint=endpoint,
        judge_model=judge_model,
        judge_enabled=judge_enabled,
        live_slack_enabled=live_slack_enabled,
        live_slack_channel_id=_normalize_env_text(
            _resolve_benchmark_env_value(
                "BENCHMARK_SLACK_CHANNEL_ID",
                dotenv_payload=dotenv_payload,
                os_environ=os.environ,
                defaults=defaults,
            )
        ),
        live_slack_user_id=_normalize_env_text(
            _resolve_benchmark_env_value(
                "BENCHMARK_SLACK_USER_ID",
                dotenv_payload=dotenv_payload,
                os_environ=os.environ,
                defaults=defaults,
            )
        ),
        live_slack_email=_normalize_env_text(
            _resolve_benchmark_env_value(
                "BENCHMARK_SLACK_EMAIL",
                dotenv_payload=dotenv_payload,
                os_environ=os.environ,
                defaults=defaults,
            )
        ),
    )


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(get_env_file_path()),
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )

    openai_api_key: str | None = Field(default=_app_default("OPENAI_API_KEY"), alias="OPENAI_API_KEY")
    tavily_api_key: str | None = Field(default=_app_default("TAVILY_API_KEY"), alias="TAVILY_API_KEY")

    chat_model: str = Field(default=_app_default("CHAT_MODEL"), alias="CHAT_MODEL")
    planner_model: str = Field(default=_app_default("PLANNER_MODEL"), alias="PLANNER_MODEL")
    summary_model: str = Field(default=_app_default("SUMMARY_MODEL"), alias="SUMMARY_MODEL")
    summary_max_tokens: int = Field(default=_app_default("SUMMARY_MAX_TOKENS"), alias="SUMMARY_MAX_TOKENS", ge=1)
    planner_max_tokens: int = Field(default=_app_default("PLANNER_MAX_TOKENS"), alias="PLANNER_MAX_TOKENS", ge=1)
    planner_reasoning_effort: ReasoningEffort | None = Field(
        default=_app_default("PLANNER_REASONING_EFFORT"),
        alias="PLANNER_REASONING_EFFORT",
    )
    docs_search_timeout_seconds: int = Field(
        default=_app_default("DOCS_SEARCH_TIMEOUT_SECONDS"),
        alias="DOCS_SEARCH_TIMEOUT_SECONDS",
        ge=1,
    )
    synthesis_timeout_seconds: int = Field(
        default=_app_default("SYNTHESIS_TIMEOUT_SECONDS"),
        alias="SYNTHESIS_TIMEOUT_SECONDS",
        ge=1,
    )
    synthesis_use_responses_api: bool = Field(
        default=_app_default("SYNTHESIS_USE_RESPONSES_API"),
        alias="SYNTHESIS_USE_RESPONSES_API",
    )
    synthesis_max_retries: int = Field(
        default=_app_default("SYNTHESIS_MAX_RETRIES"),
        alias="SYNTHESIS_MAX_RETRIES",
        ge=0,
    )
    synthesis_max_tokens: int = Field(
        default=_app_default("SYNTHESIS_MAX_TOKENS"),
        alias="SYNTHESIS_MAX_TOKENS",
        ge=1,
    )
    synthesis_compact_max_tokens: int = Field(
        default=_app_default("SYNTHESIS_COMPACT_MAX_TOKENS"),
        alias="SYNTHESIS_COMPACT_MAX_TOKENS",
        ge=1,
    )
    synthesis_prompt_snippet_chars: int = Field(
        default=_app_default("SYNTHESIS_PROMPT_SNIPPET_CHARS"),
        alias="SYNTHESIS_PROMPT_SNIPPET_CHARS",
        ge=80,
    )
    synthesis_compact_prompt_snippet_chars: int = Field(
        default=_app_default("SYNTHESIS_COMPACT_PROMPT_SNIPPET_CHARS"),
        alias="SYNTHESIS_COMPACT_PROMPT_SNIPPET_CHARS",
        ge=80,
    )
    synthesis_reasoning_effort: ReasoningEffort | None = Field(
        default=_app_default("SYNTHESIS_REASONING_EFFORT"),
        alias="SYNTHESIS_REASONING_EFFORT",
    )
    verbose: bool = Field(default=_app_default("VERBOSE"), alias="VERBOSE")
    fastapi_url: str = Field(default=_app_default("FASTAPI_URL"), alias="FASTAPI_URL")
    session_ttl_seconds: int = Field(default=_app_default("SESSION_TTL_SECONDS"), alias="SESSION_TTL_SECONDS", ge=1)
    max_active_sessions: int = Field(default=_app_default("MAX_ACTIVE_SESSIONS"), alias="MAX_ACTIVE_SESSIONS", ge=1)
    upload_max_files: int = Field(default=_app_default("UPLOAD_MAX_FILES"), alias="UPLOAD_MAX_FILES", ge=1, le=100)
    upload_max_file_mib: int = Field(default=_app_default("UPLOAD_MAX_FILE_MIB"), alias="UPLOAD_MAX_FILE_MIB", ge=1)
    upload_max_total_mib: int = Field(default=_app_default("UPLOAD_MAX_TOTAL_MIB"), alias="UPLOAD_MAX_TOTAL_MIB", ge=1)
    session_cleanup_interval_seconds: int = Field(
        default=_app_default("SESSION_CLEANUP_INTERVAL_SECONDS"),
        alias="SESSION_CLEANUP_INTERVAL_SECONDS",
        ge=1,
    )
    generated_file_ttl_seconds: int = Field(
        default=_app_default("GENERATED_FILE_TTL_SECONDS"),
        alias="GENERATED_FILE_TTL_SECONDS",
        ge=1,
    )
    file_cleanup_interval_seconds: int = Field(
        default=_app_default("FILE_CLEANUP_INTERVAL_SECONDS"),
        alias="FILE_CLEANUP_INTERVAL_SECONDS",
        ge=1,
    )
    memory_high_water_turns: int = Field(
        default=_app_default("MEMORY_HIGH_WATER_TURNS"),
        alias="MEMORY_HIGH_WATER_TURNS",
        ge=1,
    )
    memory_low_water_turns: int = Field(
        default=_app_default("MEMORY_LOW_WATER_TURNS"),
        alias="MEMORY_LOW_WATER_TURNS",
        ge=1,
    )
    memory_high_water_tokens: int = Field(
        default=_app_default("MEMORY_HIGH_WATER_TOKENS"),
        alias="MEMORY_HIGH_WATER_TOKENS",
        ge=1,
    )
    memory_low_water_tokens: int = Field(
        default=_app_default("MEMORY_LOW_WATER_TOKENS"),
        alias="MEMORY_LOW_WATER_TOKENS",
        ge=1,
    )
    memory_high_water_bytes: int = Field(
        default=_app_default("MEMORY_HIGH_WATER_BYTES"),
        alias="MEMORY_HIGH_WATER_BYTES",
        ge=1,
    )
    memory_low_water_bytes: int = Field(
        default=_app_default("MEMORY_LOW_WATER_BYTES"),
        alias="MEMORY_LOW_WATER_BYTES",
        ge=1,
    )
    memory_high_water_messages: int = Field(
        default=_app_default("MEMORY_HIGH_WATER_MESSAGES"),
        alias="MEMORY_HIGH_WATER_MESSAGES",
        ge=1,
    )
    memory_low_water_messages: int = Field(
        default=_app_default("MEMORY_LOW_WATER_MESSAGES"),
        alias="MEMORY_LOW_WATER_MESSAGES",
        ge=1,
    )
    memory_summary_max_tokens: int = Field(
        default=_app_default("MEMORY_SUMMARY_MAX_TOKENS"),
        alias="MEMORY_SUMMARY_MAX_TOKENS",
        ge=1,
    )
    memory_summary_max_bytes: int = Field(
        default=_app_default("MEMORY_SUMMARY_MAX_BYTES"),
        alias="MEMORY_SUMMARY_MAX_BYTES",
        ge=1,
    )
    memory_hard_max_bytes: int = Field(
        default=_app_default("MEMORY_HARD_MAX_BYTES"),
        alias="MEMORY_HARD_MAX_BYTES",
        ge=1,
    )

    slack_bot_token: str | None = Field(default=_app_default("SLACK_BOT_TOKEN"), alias="SLACK_BOT_TOKEN")
    slack_default_dm_email: str | None = Field(default=_app_default("SLACK_DEFAULT_DM_EMAIL"), alias="SLACK_DEFAULT_DM_EMAIL")
    slack_default_user_id: str | None = Field(default=_app_default("SLACK_DEFAULT_USER_ID"), alias="SLACK_DEFAULT_USER_ID")

    def fastapi_runtime_log_fields(self) -> dict[str, str | int]:
        return {
            "chat_model": self.chat_model,
            "planner_model": self.planner_model,
            "summary_model": self.summary_model,
            "planner_max_tokens": self.planner_max_tokens,
            "planner_reasoning_effort": self.planner_reasoning_effort or "model_default",
            "summary_max_tokens": self.summary_max_tokens,
            "docs_search_timeout_seconds": self.docs_search_timeout_seconds,
            "synthesis_timeout_seconds": self.synthesis_timeout_seconds,
            "synthesis_use_responses_api": str(self.synthesis_use_responses_api).lower(),
            "synthesis_max_retries": self.synthesis_max_retries,
            "synthesis_max_tokens": self.synthesis_max_tokens,
            "synthesis_compact_max_tokens": self.synthesis_compact_max_tokens,
            "synthesis_prompt_snippet_chars": self.synthesis_prompt_snippet_chars,
            "synthesis_compact_prompt_snippet_chars": self.synthesis_compact_prompt_snippet_chars,
            "synthesis_reasoning_effort": self.synthesis_reasoning_effort or "model_default",
            "memory_high_water_turns": self.memory_high_water_turns,
            "memory_low_water_turns": self.memory_low_water_turns,
            "memory_high_water_tokens": self.memory_high_water_tokens,
            "memory_low_water_tokens": self.memory_low_water_tokens,
            "memory_high_water_bytes": self.memory_high_water_bytes,
            "memory_low_water_bytes": self.memory_low_water_bytes,
            "memory_high_water_messages": self.memory_high_water_messages,
            "memory_low_water_messages": self.memory_low_water_messages,
            "memory_summary_max_tokens": self.memory_summary_max_tokens,
            "memory_summary_max_bytes": self.memory_summary_max_bytes,
            "memory_hard_max_bytes": self.memory_hard_max_bytes,
        }

    def conversation_memory_policy(self) -> ConversationMemoryPolicy:
        return ConversationMemoryPolicy(
            high_water_turns=self.memory_high_water_turns,
            low_water_turns=self.memory_low_water_turns,
            high_water_tokens=self.memory_high_water_tokens,
            low_water_tokens=self.memory_low_water_tokens,
            high_water_bytes=self.memory_high_water_bytes,
            low_water_bytes=self.memory_low_water_bytes,
            high_water_messages=self.memory_high_water_messages,
            low_water_messages=self.memory_low_water_messages,
            summary_max_tokens=self.memory_summary_max_tokens,
            summary_max_bytes=self.memory_summary_max_bytes,
            hard_max_bytes=self.memory_hard_max_bytes,
        )

    @model_validator(mode="after")
    def _validate_conversation_memory_policy(self) -> "AppSettings":
        self.conversation_memory_policy()
        return self

    @field_validator("planner_reasoning_effort", "synthesis_reasoning_effort", mode="before")
    @classmethod
    def _normalize_reasoning_effort(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"", "default", "model_default"}:
                return None
            return normalized
        return value

    @field_validator("planner_reasoning_effort")
    @classmethod
    def _validate_planner_reasoning_effort(cls, value: ReasoningEffort | None, info: ValidationInfo) -> ReasoningEffort | None:
        model = info.data.get("planner_model")
        supported = _PLANNER_MODEL_REASONING_EFFORTS.get(model)
        if value is not None and supported is not None and value not in supported:
            raise ValueError(
                f"PLANNER_REASONING_EFFORT={value!r} is not supported "
                f"by PLANNER_MODEL={model!r}; supported values: {', '.join(supported)}. "
                "Leave PLANNER_REASONING_EFFORT blank to use the model default."
            )
        return value


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


__all__ = [
    "APP_ENV_SPECS",
    "APP_ENV_SPEC_BY_NAME",
    "AppSettings",
    "BenchmarkCLIEnvSettings",
    "BENCHMARK_ENV_SPECS",
    "BENCHMARK_ENV_SPEC_BY_NAME",
    "ConfigurationError",
    "DEFAULT_BENCHMARK_CONFIG_PATH",
    "EnvVarSpec",
    "get_settings",
    "load_benchmark_cli_env_settings",
    "load_benchmark_env_defaults",
    "validate_required_keys",
]
