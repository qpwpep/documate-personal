from __future__ import annotations

from pathlib import Path


_CURRENT_FILE_PATH = Path(__file__).resolve()


def get_project_root_path() -> Path:
    return _CURRENT_FILE_PATH.parent.parent


def get_output_dir() -> Path:
    return get_project_root_path() / "output"


def get_save_text_output_dir() -> Path:
    return get_output_dir() / "save_text"


def get_runtime_output_dir() -> Path:
    return get_output_dir() / "runtime"


def get_service_state_path() -> Path:
    return get_runtime_output_dir() / "web_services_state.json"


def get_runtime_log_path(filename: str) -> Path:
    return get_runtime_output_dir() / filename
