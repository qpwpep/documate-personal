from __future__ import annotations

from functools import lru_cache
import re

from src.core.contracts.debug import RetryState
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.core.rules import get_rules_config


def _planner_rules():
    return get_rules_config().planner


@lru_cache(maxsize=1)
def _action_clause_pattern():
    return re.compile(_planner_rules().action_clause_pattern)


@lru_cache(maxsize=1)
def _docs_clause_pattern():
    return re.compile(_planner_rules().docs_clause_pattern)


@lru_cache(maxsize=1)
def _compare_clause_pattern():
    return re.compile(_planner_rules().compare_clause_pattern)


@lru_cache(maxsize=1)
def _docs_identifier_pattern():
    return re.compile(_planner_rules().docs_identifier_pattern)


@lru_cache(maxsize=1)
def _upload_marker_pattern():
    return re.compile(
        r"(?i)(upload(?:ed)?|current file|current notebook|this file|this notebook|\.ipynb|\.py|업로드|현재 파일|이 파일|이 노트북)"
    )


def _normalize_query_text(text: str) -> str:
    collapsed = " ".join(str(text or "").replace("\r", "\n").split())
    return collapsed.strip(" ,.;:-")


def _strip_auxiliary_clauses(text: str) -> str:
    normalized = str(text or "")
    lowered = normalized.lower()
    cut_index = len(normalized)
    for marker in _planner_rules().auxiliary_markers:
        index = lowered.find(marker.lower())
        if index >= 0:
            cut_index = min(cut_index, index)
    normalized = normalized[:cut_index]
    if "\n" in normalized:
        normalized = normalized.split("\n", 1)[0]
    return _normalize_query_text(normalized)


def _strip_action_clauses(text: str) -> str:
    parts = re.split(r"(?<=[?.!,])|\band\b|그리고|또는", str(text or ""), flags=re.I)
    kept = [part.strip() for part in parts if part.strip() and not _action_clause_pattern().search(part)]
    return _normalize_query_text(" ".join(kept or [str(text or "")]))


def _compact_docs_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    compact = re.sub(_compare_clause_pattern(), " ", compact)
    upload_match = re.search(
        r"(?i)(upload(?:ed)?|current file|current notebook|this file|this notebook|\.ipynb|\.py|업로드|현재 파일|이 파일|이 노트북)",
        compact,
    )
    if upload_match:
        compact = compact[: upload_match.start()]
    for phrase in _planner_rules().trailing_docs_stop_phrases:
        compact = re.sub(re.escape(phrase), " ", compact, flags=re.I)
    compact = re.sub(r"(?i)\b(official docs?|official documentation)\b", " ", compact)
    compact = re.sub(
        r"(?i)\b(explain|describe|summarize|show|find|tell)\b|설명|요약|정리|찾아|보여",
        " ",
        compact,
    )
    compact = re.sub(r"(?i)\b(and|it with the|with the)\b", " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    compact = _normalize_query_text(compact)

    identifier_tokens = _docs_identifier_pattern().findall(compact)
    if len(identifier_tokens) >= 2:
        deduped: list[str] = []
        stopwords = set(_planner_rules().docs_identifier_stopwords)
        for token in identifier_tokens:
            if token.lower() in stopwords:
                continue
            if token not in deduped:
                deduped.append(token)
        return " ".join(deduped[:4])

    return compact


def _compact_upload_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    upload_match = _upload_marker_pattern().search(compact)
    prefix = compact[: upload_match.start()] if upload_match else compact
    if upload_match:
        compact = compact[upload_match.start() :]
    compact = re.sub(_docs_clause_pattern(), " ", compact)
    compact = re.sub(_compare_clause_pattern(), " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    compact = _normalize_query_text(compact)

    preserved_identifiers = _extract_upload_identifiers(prefix)
    if preserved_identifiers:
        compact_identifiers = {
            token.lower()
            for token in _docs_identifier_pattern().findall(compact)
        }
        missing_identifiers = [
            token for token in preserved_identifiers if token.lower() not in compact_identifiers
        ]
        if missing_identifiers:
            compact = _normalize_query_text(" ".join([*missing_identifiers[:4], compact]))

    return compact


def _extract_upload_identifiers(text: str) -> list[str]:
    stopwords = {item.lower() for item in _planner_rules().docs_identifier_stopwords}
    stopwords.update(
        {
            "compare",
            "comparison",
            "versus",
            "vs",
            "upload",
            "uploaded",
            "current",
            "this",
            "file",
            "notebook",
            "find",
            "show",
            "tell",
            "explain",
            "describe",
            "usage",
            "used",
            "example",
            "examples",
        }
    )
    identifiers: list[str] = []
    for token in _docs_identifier_pattern().findall(text):
        normalized = token.strip()
        if not normalized or normalized.lower() in stopwords:
            continue
        if normalized not in identifiers:
            identifiers.append(normalized)
    return identifiers


def _compact_local_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    compact = re.sub(_docs_clause_pattern(), " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    return _normalize_query_text(compact)


def sanitize_retrieval_query(
    *,
    route: str,
    query: str,
    retry_context: RetryState | None = None,
) -> str:
    base_query = _normalize_query_text(query)
    if route == "docs":
        sanitized = _compact_docs_query(base_query)
        retry_reason = str(retry_context.retry_reason or "") if retry_context is not None else ""
        if retry_reason == "no_evidence":
            sanitized = re.sub(
                r"(?i)\b(why|how|explain|describe|summarize)\b|설명|요약|이유|주의점",
                " ",
                sanitized,
            )
            sanitized = _normalize_query_text(re.sub(r"\s+", " ", sanitized))
    elif route == "upload":
        sanitized = _compact_upload_query(base_query)
    else:
        sanitized = _compact_local_query(base_query)
    return sanitized or base_query


def sanitize_planner_output_queries(
    planner_output: PlannerOutput,
    *,
    user_input: str,
    retry_context: RetryState | None = None,
) -> PlannerOutput:
    if not planner_output.use_retrieval or not planner_output.tasks:
        return planner_output
    sanitized_tasks = [
        RetrievalTask(
            route=task.route,
            query=sanitize_retrieval_query(
                route=task.route,
                query=task.query or user_input,
                retry_context=retry_context,
            ),
            k=task.k,
        )
        for task in planner_output.tasks
    ]
    return PlannerOutput(use_retrieval=True, tasks=sanitized_tasks)
