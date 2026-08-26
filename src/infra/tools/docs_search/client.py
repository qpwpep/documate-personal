from __future__ import annotations

from typing import Any, Literal

import requests

TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"


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
        raise RuntimeError("TAVILY_API_KEY is not configured")

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

    try:
        response = requests.post(
            TAVILY_SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )
    except requests.Timeout as exc:
        raise TimeoutError(f"Tavily search timed out after {timeout_seconds}s") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"Tavily request failed ({exc})") from exc

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError("invalid JSON response from Tavily") from exc

    if response.status_code != 200:
        detail = body.get("detail") if isinstance(body, dict) else None
        error_message = detail.get("error") if isinstance(detail, dict) else None
        if not error_message:
            error_message = f"HTTP {response.status_code}"
        raise RuntimeError(str(error_message))

    if not isinstance(body, dict) or not isinstance(body.get("results"), list):
        raise RuntimeError("missing or invalid Tavily results payload")
    return body
