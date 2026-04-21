from __future__ import annotations

from pathlib import Path


_CURRENT_FILE_PATH = Path(__file__).resolve()


def get_project_root_path() -> Path:
    return _CURRENT_FILE_PATH.parent.parent.parent


def get_env_file_path() -> Path:
    return get_project_root_path() / ".env"


def get_env_example_path() -> Path:
    return get_project_root_path() / ".env.example"


def get_readme_path() -> Path:
    return get_project_root_path() / "README.md"


def get_data_dir() -> Path:
    return get_project_root_path() / "data"


def get_uploads_dir() -> Path:
    return get_project_root_path() / "uploads"


def get_upload_session_dir(session_id: str) -> Path:
    return get_uploads_dir() / session_id


def get_output_dir() -> Path:
    return get_project_root_path() / "output"


def get_save_text_output_dir() -> Path:
    return get_output_dir() / "save_text"


def get_runtime_output_dir() -> Path:
    return get_output_dir() / "runtime"


def get_benchmark_output_dir() -> Path:
    return get_output_dir() / "benchmarks"


def get_service_state_path() -> Path:
    return get_runtime_output_dir() / "web_services_state.json"


def get_runtime_log_path(filename: str) -> Path:
    return get_runtime_output_dir() / filename


def get_docs_dir() -> Path:
    return get_project_root_path() / "docs"


def get_benchmark_history_svg_path() -> Path:
    return get_docs_dir() / "assets" / "benchmark_history.svg"


def get_benchmark_data_dir() -> Path:
    return get_data_dir() / "benchmarks"


def get_benchmark_fixtures_dir() -> Path:
    return get_benchmark_data_dir() / "fixtures"


def get_benchmark_config_path() -> Path:
    return get_benchmark_data_dir() / "config.toml"


def get_generated_cases_fixture_path() -> Path:
    return get_benchmark_fixtures_dir() / "cases.generated.jsonl"


def get_regression_seed_cases_path() -> Path:
    return get_benchmark_fixtures_dir() / "cases.regression.seed.jsonl"


def get_local_rag_index_dir() -> Path:
    return get_data_dir() / "index"
