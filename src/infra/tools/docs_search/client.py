from __future__ import annotations

from typing import Any, Literal

import requests

from src.infra.tail_latency import invoke_with_optional_hedge


TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"


def request_tavily_search(
    *,
    query: str,
    tavily_api_key: str | None,
    include_domains: list[str],
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"],
    timeout_seconds: int,
    hedge_delay_seconds: float = 0.0,
    hedge_max_attempts: int = 2,
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

    def _post_search() -> requests.Response:
        return requests.post(
            TAVILY_SEARCH_API_URL,
            json=payload,
            headers=headers,
            timeout=timeout_seconds,
        )

    try:
        hedge_result = invoke_with_optional_hedge(
            _post_search,
            hedge_delay_seconds=hedge_delay_seconds,
            max_attempts=hedge_max_attempts,
            is_success=lambda item: int(item.status_code) == 200,
            overall_timeout_seconds=timeout_seconds,
        )
        response = hedge_result.value
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

    if not isinstance(body, dict):
        raise RuntimeError("unexpected response type from Tavily")
    if hedge_result.hedge_started or hedge_result.hedge_dropped:
        body["_tail_hedge"] = {
            "hedge_started": hedge_result.hedge_started,
            "hedge_dropped": hedge_result.hedge_dropped,
            "hedge_winner": hedge_result.winner,
            "hedge_attempts_started": hedge_result.hedges_started,
            "hedge_attempts_dropped": hedge_result.hedges_dropped,
        }
    return body
