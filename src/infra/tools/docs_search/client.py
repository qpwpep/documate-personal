from __future__ import annotations

from typing import Any, Literal

import requests

from src.core.contracts.debug import ErrorCode
from src.infra.llm_boundary import bounded_request_timeout, check_request_deadline

TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"


class TavilySearchError(RuntimeError):
    """Keep provider status without exposing its untrusted response body."""

    def __init__(self, status_code: int | None = None) -> None:
        self.status_code = status_code
        self.error_code: ErrorCode = "RETRIEVAL_DOCS_FAILED"
        if status_code is None or status_code in {401, 403}:
            self.error_code = "provider_configuration"
        elif status_code == 429:
            self.error_code = "provider_rate_limited"
        message = "Tavily API key is not configured" if status_code is None else f"Tavily search failed (HTTP {status_code})"
        super().__init__(message)


def request_tavily_search(
    *,
    query: str,
    tavily_api_key: str | None,
    include_domains: list[str],
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"],
    timeout_seconds: int,
    max_results: int = 3,
    include_raw_content: Literal[False, "markdown", "text"] = False,
) -> dict[str, Any]:
    if not tavily_api_key:
        raise TavilySearchError()

    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json",
        "X-Client-Source": "documate",
    }
    payload: dict[str, Any] = {
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "topic": "general",
        "include_domains": include_domains,
    }
    if include_raw_content:
        payload["include_raw_content"] = include_raw_content

    timeout = bounded_request_timeout(timeout_seconds, stage="retrieval")
    try:
        response = requests.post(
            TAVILY_SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except requests.Timeout as exc:
        check_request_deadline("retrieval")
        raise TimeoutError(f"Tavily search timed out after {timeout}s") from exc
    except requests.RequestException as exc:
        check_request_deadline("retrieval")
        raise RuntimeError(f"Tavily request failed ({type(exc).__name__})") from exc

    check_request_deadline("retrieval")
    if response.status_code != 200:
        raise TavilySearchError(response.status_code)

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("invalid JSON response from Tavily") from exc

    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise RuntimeError("missing or invalid Tavily results payload")
    return body
