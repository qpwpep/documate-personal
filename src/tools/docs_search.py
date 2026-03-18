from __future__ import annotations

from typing import Any, Literal
from urllib.parse import urlparse

import requests
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from ..answer_schema import normalize_confidence
from ..settings import AppSettings
from ._common import build_evidence_item, build_retrieval_payload, dedupe_evidence_dicts


TAVILY_SEARCH_API_URL = "https://api.tavily.com/search"
_DOCS_QUERY_HINTS: tuple[tuple[tuple[str, ...], str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("train_test_split",),
        "scikit-learn",
        ("scikit-learn.org",),
        ("train_test_split sklearn.model_selection",),
    ),
    (
        ("standardscaler",),
        "scikit-learn",
        ("scikit-learn.org",),
        ("StandardScaler sklearn.preprocessing",),
    ),
    (
        ("logisticregression",),
        "scikit-learn",
        ("scikit-learn.org",),
        ("LogisticRegression sklearn.linear_model",),
    ),
    (
        ("pipeline",),
        "scikit-learn",
        ("scikit-learn.org",),
        ("Pipeline sklearn.pipeline",),
    ),
    (("response_model",), "fastapi", ("fastapi.tiangolo.com",), ("response_model fastapi",)),
    (
        ("merge",),
        "pandas",
        ("pandas.pydata.org",),
        ("pandas merge user guide", "pandas merging user guide"),
    ),
    (
        ("groupby",),
        "pandas",
        ("pandas.pydata.org",),
        ("pandas groupby user guide",),
    ),
    (
        ("concat",),
        "pandas",
        ("pandas.pydata.org",),
        ("pandas concat user guide",),
    ),
    (("broadcasting",), "numpy", ("numpy.org",), ("broadcasting numpy",)),
    (("dataloader", "dataset"), "pytorch", ("docs.pytorch.org",), ("DataLoader torch.utils.data",)),
)
_ALLOWED_DOC_PATH_PREFIXES: dict[str, tuple[str, ...]] = {
    "docs.python.org": ("/3/",),
    "git-scm.com": ("/docs/",),
    "python.langchain.com": ("/docs/",),
    "matplotlib.org": ("/stable/",),
    "numpy.org": ("/doc/stable/",),
    "pandas.pydata.org": ("/docs/",),
    "docs.pytorch.org": ("/docs/stable/",),
    "huggingface.co": ("/docs/",),
    "fastapi.tiangolo.com": ("/",),
    "crummy.com": ("/software/BeautifulSoup/bs4/doc/",),
    "docs.streamlit.io": ("/",),
    "gradio.app": ("/docs/",),
    "scikit-learn.org": ("/stable/",),
    "docs.pydantic.dev": ("/latest/",),
}
_ERROR_PAGE_MARKERS: tuple[str, ...] = (
    "404",
    "page not found",
    "github pages",
    "does not exist",
    "not found",
    "requested file",
)


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


def _normalize_path_prefix(path: str) -> str:
    normalized = "/" + str(path or "").strip().lstrip("/")
    while "//" in normalized:
        normalized = normalized.replace("//", "/")
    if normalized == "/":
        return normalized
    return normalized if normalized.endswith("/") else normalized + "/"


def is_allowed_doc_url(url: str) -> bool:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme.lower() != "https" or not parsed.netloc:
        return False
    domain = parsed.netloc.strip().lower()
    if domain.startswith("www."):
        domain = domain[4:]
    allowed_prefixes = _ALLOWED_DOC_PATH_PREFIXES.get(domain)
    if not allowed_prefixes:
        return False
    normalized_path = _normalize_path_prefix(parsed.path or "/")
    return any(normalized_path.startswith(prefix) for prefix in allowed_prefixes)


def is_valid_doc_result(*, url: str, title: Any, snippet: Any) -> bool:
    if not is_allowed_doc_url(url):
        return False

    combined = " ".join(
        part.strip().lower()
        for part in [str(url or ""), str(title or ""), str(snippet or "")]
        if str(part or "").strip()
    )
    if not combined:
        return False

    return not any(marker in combined for marker in _ERROR_PAGE_MARKERS)


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


def infer_docs_query_hint(query: str) -> tuple[str, list[str], list[str]] | None:
    lowered = str(query or "").lower()
    for identifiers, library_name, domains, fallback_queries in _DOCS_QUERY_HINTS:
        if any(identifier in lowered for identifier in identifiers):
            return library_name, list(domains), list(fallback_queries)
    return None


def build_docs_search_tool(settings: AppSettings) -> Any:
    default_domains = list(_ALLOWED_DOC_PATH_PREFIXES.keys())

    def tavily_search(
        query: str,
        search_depth: Literal["basic", "advanced", "fast", "ultra-fast"] = "basic",
        include_domains: list[str] | None = None,
    ) -> dict[str, Any]:
        domains = normalize_include_domains(include_domains or default_domains)
        effective_query = str(query or "").strip()
        fallback_queries: list[str] = []
        if include_domains is None:
            query_hint = infer_docs_query_hint(effective_query)
            if query_hint is not None:
                library_name, hinted_domains, fallback_queries = query_hint
                domains = normalize_include_domains(hinted_domains)
                if library_name.lower() not in effective_query.lower():
                    effective_query = f"{effective_query} {library_name}".strip()
        try:
            raw_results = request_tavily_search(
                query=effective_query,
                tavily_api_key=settings.tavily_api_key,
                include_domains=domains,
                search_depth=search_depth,
                timeout_seconds=settings.docs_search_timeout_seconds,
            )
        except Exception as exc:
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message=f"invoke failed ({exc})",
            )

        if not isinstance(raw_results, dict):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="unexpected response type from Tavily",
            )
        results = raw_results.get("results")
        if not isinstance(results, list):
            return build_retrieval_payload(
                tool="tavily_search",
                route="docs",
                query=effective_query,
                status="error",
                message="missing or invalid Tavily results payload",
            )

        evidence_items = []
        for result in results:
            if not isinstance(result, dict):
                continue
            url = str(result.get("url") or "").strip()
            if not is_valid_doc_result(
                url=url,
                title=result.get("title"),
                snippet=result.get("content"),
            ):
                continue
            evidence_item = build_evidence_item(
                kind="official",
                tool="tavily_search",
                url_or_path=url,
                title=result.get("title"),
                snippet=result.get("content"),
                score=normalize_confidence(result.get("score"), clamp=True),
                metadata={},
            )
            if evidence_item is not None:
                evidence_items.append(evidence_item)

        for fallback_query in fallback_queries:
            if evidence_items:
                break
            try:
                fallback_results = request_tavily_search(
                    query=fallback_query,
                    tavily_api_key=settings.tavily_api_key,
                    include_domains=domains,
                    search_depth=search_depth,
                    timeout_seconds=settings.docs_search_timeout_seconds,
                )
            except Exception:
                continue
            fallback_items = fallback_results.get("results") if isinstance(fallback_results, dict) else None
            if not isinstance(fallback_items, list):
                continue
            for result in fallback_items:
                if not isinstance(result, dict):
                    continue
                url = str(result.get("url") or "").strip()
                if not is_valid_doc_result(
                    url=url,
                    title=result.get("title"),
                    snippet=result.get("content"),
                ):
                    continue
                evidence_item = build_evidence_item(
                    kind="official",
                    tool="tavily_search",
                    url_or_path=url,
                    title=result.get("title"),
                    snippet=result.get("content"),
                    score=normalize_confidence(result.get("score"), clamp=True),
                    metadata={},
                )
                if evidence_item is not None:
                    evidence_items.append(evidence_item)

        evidence = dedupe_evidence_dicts(evidence_items)
        return build_retrieval_payload(
            tool="tavily_search",
            route="docs",
            query=effective_query,
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
