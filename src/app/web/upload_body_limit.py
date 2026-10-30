"""Bound multipart bytes before Starlette spools browser uploads to disk."""
from __future__ import annotations

from collections.abc import Awaitable, Callable

from fastapi import HTTPException, Request, Response
from fastapi.routing import APIRoute
import starlette.exceptions
from starlette.formparsers import MultiPartException
from starlette.types import Message


class BoundedUploadRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Awaitable[Response]]:
        handler = super().get_route_handler()

        async def bounded(request: Request) -> Response:
            file_mib = request.app.state.settings.upload_max_file_mib
            # A single file also needs a boundary, filename and part headers.
            body_limit = file_mib * 1024 * 1024 + 64 * 1024
            declared_length = request.headers.get("content-length")
            if declared_length is not None:
                try:
                    declared_bytes = int(declared_length)
                except ValueError as exc:
                    raise HTTPException(status_code=400, detail="잘못된 Content-Length입니다.") from exc
                if declared_bytes > body_limit:
                    raise _too_large(file_mib)

            received = 0
            exceeded = False

            async def receive() -> Message:
                nonlocal received, exceeded
                message = await request.receive()
                if message["type"] == "http.request":
                    received += len(message.get("body", b""))
                    if received > body_limit:
                        exceeded = True
                        # Multipart parser error handling closes partially
                        # spooled files before Request translates this to 400.
                        raise MultiPartException("Upload body exceeds its limit")
                return message

            try:
                return await handler(Request(request.scope, receive=receive))
            except starlette.exceptions.HTTPException:
                if exceeded:
                    raise _too_large(file_mib) from None
                raise

        return bounded


def _too_large(file_mib: int) -> HTTPException:
    return HTTPException(status_code=413, detail={
        "code": "UPLOAD_FILE_TOO_LARGE",
        "message": f"파일당 {file_mib} MiB까지 첨부할 수 있습니다.",
        "files": [],
    })
