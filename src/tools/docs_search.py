from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..domain_docs import DEFAULT_DOCS
from ..settings import AppSettings
from ._common import build_evidence_item, build_retrieval_payload, dedupe_evidence_dicts


TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"


class TavilyArgs(BaseModel):
    query: str = Field(description="Search query for official documentation.")
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = Field(
        default="basic",
        description="Search depth for Tavily.",
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="Optional domain whitelist for this query.",
    )


def normalize_include_domains(raw_values: list[str]) -> list[str]:
    normalized: list[str] = []
    for value in raw_values:
        candidate = value.strip()
        if not candidate:
            continue
        parsed = urlparse(candidate if "://" in candidate else f"https://{candidate}")
        domain = (parsed.netloc or parsed.path).strip().lower()
        if domain.startswith("www."):
            domain = domain[4:]
        if domain and domain not in normalized:
            normalized.append(domain)
    return normalized


def request_tavily_search(
    *,
    query: str,
    tavily_api_key: str | None,
    include_domains: list[str],
    search_depth: Literal["basic", "advanced", "fast", "ultra-fast"],
    timeout_seconds: int,
    max_results: int = 3,
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

    if not isinstance(body, dict):
        raise RuntimeError("unexpected response type from Tavily")
    return body


def build_docs_search_tool(settings: AppSettings) -> Any:
    default_domains = normalize_include_domains(list(DEFAULT_DOCS.values()))

    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        domains = normalize_include_domains(include_domains or default_domains)
        try:
            raw_results = request_tavily_search(
                query=query,
                tavily_api_key=settings.tavily_api_key,
                include_domains=domains,
                search_depth=search_depth,
                timeout_seconds=settings.docs_search_timeout_seconds,
            )
        except Exception as exc:
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=query,
                status="error",
                message=f"invoke failed ({exc})",
            )

        if not isinstance(raw_results, dict):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=query,
                status="error",
                message="unexpected response type from Tavily",
            )
        results = raw_results.get("results")
        if not isinstance(results, list):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=query,
                status="error",
                message="missing or invalid Tavily results payload",
            )

        evidence_items = []
        for result in results:
            if not isinstance(result, dict):
                continue
            evidence_item = build_evidence_item(
                kind="official",
                tool="tavily_search",
                url_or_path=str(result.get("url") or "").strip(),
                title=result.get("title"),
                snippet=result.get("content"),
                score=result.get("score"),
                metadata={},
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)

        evidence = dedupe_evidence_dicts(evidence_items)
        return build_retrieval_payload(
            tool="tavily_search",
            route="docs",
            query=query,
            evidence=evidence,
            status="success" if evidence else "no_result",
            message="" if evidence else "no official documentation evidence found",
        )

    return StructuredTool.from_function(
        name="tavily_search",
        description=(
            "Search official documentation on the web and return structured evidence items. "
            "Use this for current or official references."
        ),
        func=tavily_search,
        args_schema=TavilyArgs,
    )
