"""Serve the built React UI without changing the API routing contract."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles


REACT_BUILD_INSTRUCTIONS = (
    "React UI 빌드가 없습니다. 저장소 루트에서 다음 명령을 실행한 뒤 서버를 다시 시작하세요.\n\n"
    "npm --prefix frontend ci\n"
    "npm --prefix frontend run build\n"
)


def get_frontend_dist_path() -> Path:
    # UI assets belong to the source checkout, independently of runtime data paths.
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def mount_react_ui(app: FastAPI) -> None:
    directory = get_frontend_dist_path()
    if (directory / "index.html").is_file():
        app.mount("/app", StaticFiles(directory=directory, html=True), name="react-ui")
        return

    @app.get("/app/", include_in_schema=False)
    def missing_build() -> PlainTextResponse:
        return PlainTextResponse(REACT_BUILD_INSTRUCTIONS, status_code=503)
