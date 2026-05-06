from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal
from urllib.parse import urlparse, urlunparse

import requests

from src.infra.tools.docs_search.policy import doc_url_filter_reason


DocUrlValidationReason = Literal["http_error", "redirect_policy", "request_failed"]


@dataclass(frozen=True, slots=True)
class DocUrlValidationResult:
    ok: bool
    final_url: str
    status_code: int | None = None
    reason: DocUrlValidationReason | None = None


_REQUEST_HEADERS = {
    "User-Agent": "DocuMate/0.2 (+https://github.com/documate)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


@lru_cache(maxsize=512)
def validate_doc_url(url: str, *, timeout_seconds: float = 2.0) -> DocUrlValidationResult:
    candidate = str(url or "").strip()
    if not candidate:
        return DocUrlValidationResult(ok=False, final_url="", reason="request_failed")

    response = None
    try:
        response = requests.head(
            candidate,
            allow_redirects=True,
            headers=_REQUEST_HEADERS,
            timeout=timeout_seconds,
        )
        if response.status_code != 200:
            response.close()
            response = requests.get(
                candidate,
                allow_redirects=True,
                headers=_REQUEST_HEADERS,
                stream=True,
                timeout=timeout_seconds,
            )
        final_url = _preserve_fragment(candidate, response.url)
        status_code = int(response.status_code)
    except requests.RequestException:
        return DocUrlValidationResult(
            ok=False,
            final_url=candidate,
            reason="request_failed",
        )
    finally:
        if response is not None:
            response.close()

    if status_code != 200:
        return DocUrlValidationResult(
            ok=False,
            final_url=final_url,
            status_code=status_code,
            reason="http_error",
        )

    if doc_url_filter_reason(final_url) is not None:
        return DocUrlValidationResult(
            ok=False,
            final_url=final_url,
            status_code=status_code,
            reason="redirect_policy",
        )

    return DocUrlValidationResult(
        ok=True,
        final_url=final_url,
        status_code=status_code,
    )


def _preserve_fragment(original_url: str, final_url: str) -> str:
    original = urlparse(str(original_url or ""))
    final = urlparse(str(final_url or ""))
    if original.fragment and not final.fragment:
        return urlunparse(final._replace(fragment=original.fragment))
    return str(final_url or "")
