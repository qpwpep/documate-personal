from __future__ import annotations

from functools import lru_cache
import re

from ...contracts.debug import RetryState
from ...planner_schema import PlannerOutput, RetrievalTask
from ...rules import get_rules_config


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
    parts = re.split(r"(?<=[?.!,])|\band\b|\uadf8\ub9ac\uace0|\ub610\ub294", str(text or ""), flags=re.I)
    kept = [part.strip() for part in parts if part.strip() and not _action_clause_pattern().search(part)]
    return _normalize_query_text(" ".join(kept or [str(text or "")]))


def _compact_docs_query(text: str) -> str:
    compact = _strip_auxiliary_clauses(_strip_action_clauses(text))
    compact = re.sub(_compare_clause_pattern(), " ", compact)
    upload_match = re.search(
        r"(?i)(upload(?:ed)?|current file|current notebook|this file|this notebook|\.ipynb|\.py|\uc5c5\ub85c\ub4dc|\ud604\uc7ac \ud30c\uc77c|\uc774 \ud30c\uc77c|\uc774 \ub178\ud2b8\ubd81)",
        compact,
    )
    if upload_match:
        compact = compact[: upload_match.start()]
    for phrase in _planner_rules().trailing_docs_stop_phrases:
        compact = re.sub(re.escape(phrase), " ", compact, flags=re.I)
    compact = re.sub(r"(?i)\b(official docs?|official documentation)\b", " ", compact)
    compact = re.sub(
        r"(?i)\b(explain|describe|summarize|show|find|tell)\b|\uc124\uba85|\uc694\uc57d|\uc815\ub9ac|\ucc3e\uc544|\ubcf4\uc5ec",
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
    upload_match = re.search(
        r"(?i)(upload(?:ed)?|current file|current notebook|this file|this notebook|\.ipynb|\.py|\uc5c5\ub85c\ub4dc|\ud604\uc7ac \ud30c\uc77c|\uc774 \ud30c\uc77c|\uc774 \ub178\ud2b8\ubd81)",
        compact,
    )
    if upload_match:
        compact = compact[upload_match.start() :]
    compact = re.sub(_docs_clause_pattern(), " ", compact)
    compact = re.sub(_compare_clause_pattern(), " ", compact)
    compact = re.sub(r"\s+", " ", compact)
    return _normalize_query_text(compact)


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
                r"(?i)\b(why|how|explain|describe|summarize)\b|\uc124\uba85|\uc694\uc57d|\uc774\uc720|\uc8fc\uc758\uc810",
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
