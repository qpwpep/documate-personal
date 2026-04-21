from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.answer_schema import clean_grounded_text
from src.core.contracts import GraphState
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.request_contracts import infer_answer_contract, render_answer_contract_prompt
from src.core.prompts import SYS_POLICY
from src.runtime.nodes.retrieval import format_evidence_for_prompt
from src.runtime.nodes.session import keep_recent_messages


SYNTHESIS_CONTRACT = (
    "[Synthesis Contract]\n"
    "- Return structured output only.\n"
    "- claims must be sentence-level.\n"
    "- Each claim must cite one or more exact evidence source_id values from Retrieved Evidence.\n"
    "- Do not invent evidence ids.\n"
    "- If the evidence is insufficient, return claims=[].\n"
    "- Do not embed citation numbers like [1] in claim text; the renderer adds them.\n"
    "- Do not list raw links or search results instead of synthesizing.\n"
    "- Restate the answer in the user's language.\n"
    "- Prioritize official documentation before secondary detail.\n"
    "- Ignore markdown formatting, breadcrumbs, navigation labels, and table-of-contents text that may appear in docs snippets.\n"
    "- For docs and hybrid evidence, avoid extraction-only responses unless the user explicitly asked for extraction."
)
PLAIN_SUMMARY_ATTACH_CONTRACT = (
    "[Plain Summary Attach Fallback]\n"
    "- Use only Retrieved Evidence.\n"
    "- Return plain text only.\n"
    "- Return at most 4 short lines.\n"
    "- Keep the answer grounded in the same order as Retrieved Evidence.\n"
    "- Do not include citations, bullets, numbering, JSON, or markdown."
)
_LEADING_BULLET_PATTERN = re.compile(r"^\s*(?:[-*]|\d+[.)])\s*")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_query_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split()).strip(" ,.;:-")


def _truncate_prompt_text(text: Any, *, limit: int) -> str:
    normalized = str(text or "").strip()
    if limit <= 0 or len(normalized) <= limit:
        return normalized
    bridge = " ... "
    if limit <= len(bridge) + 2:
        return normalized[: max(0, limit - 3)].rstrip() + "..."
    available = limit - len(bridge)
    head_chars = max(1, available // 2)
    tail_chars = max(1, available - head_chars)
    head = normalized[:head_chars].rstrip()
    tail = normalized[-tail_chars:].lstrip()
    if head and tail:
        return f"{head}{bridge}{tail}"
    separator = " ... "
    if limit <= len(separator) + 8:
        return normalized[:limit].rstrip()
    head = (limit - len(separator)) // 2
    tail = limit - len(separator) - head
    return normalized[:head].rstrip() + separator + normalized[-tail:].lstrip()


def _prepare_evidence_for_prompt(
    items: list[dict[str, Any]],
    *,
    snippet_char_limit: int,
    evidence_char_budget: int | None = None,
) -> list[dict[str, Any]]:
    prepared_items: list[dict[str, Any]] = []
    remaining_budget = evidence_char_budget
    for item in items:
        if not isinstance(item, dict):
            continue
        prompt_item = dict(item)
        kind = str(prompt_item.get("kind") or "").strip().lower()
        if kind == "official":
            cleaned_title = clean_grounded_text(str(prompt_item.get("title") or ""))
            cleaned_snippet = clean_grounded_text(str(prompt_item.get("snippet") or ""))
            prompt_item["title"] = cleaned_title
            prompt_item["snippet"] = cleaned_snippet
        if kind != "local":
            effective_limit = snippet_char_limit
            if remaining_budget is not None:
                if remaining_budget <= 0:
                    effective_limit = 0
                else:
                    effective_limit = min(effective_limit, remaining_budget)
            prompt_item["snippet"] = _truncate_prompt_text(
                prompt_item.get("snippet"),
                limit=effective_limit,
            )
        if remaining_budget is not None:
            remaining_budget -= len(str(prompt_item.get("snippet") or ""))
        prepared_items.append(prompt_item)
    return prepared_items


def _build_synthesis_instruction_block(
    *,
    action_rules: list[str],
    has_hybrid_evidence: bool,
    attempt: int,
) -> str:
    lines = [
        "[Synthesis Instructions]",
        "- Return structured output only.",
        "- claims must be sentence-level.",
        "- Each claim must cite one or more exact evidence source_id values from Retrieved Evidence.",
        "- Do not invent evidence ids.",
        "- If the evidence is insufficient, return claims=[].",
        "- Do not embed citation numbers like [1] in claim text; the renderer adds them.",
        "- Do not list raw links or search results instead of synthesizing.",
        "- Restate the answer in the user's language.",
        "- Prioritize official documentation before secondary detail.",
        "- Ignore markdown formatting, breadcrumbs, navigation labels, and table-of-contents text that may appear in docs snippets.",
        "- Do not answer with only a file path, a tool name, or an action acknowledgment.",
        "- For upload/local code evidence, explain the relevant snippet and extract the requested parameters or options instead of pasting raw code only.",
    ]
    if has_hybrid_evidence:
        lines.extend(
            [
                "[Hybrid Synthesis]",
                "- For docs plus uploaded/local evidence, explain the official takeaway first.",
                "- Then add an explicit comparison against the uploaded/local evidence.",
                "- Keep the official explanation and the comparison as distinct claim groups.",
                "- The official claim group must cite only official docs source_id values.",
                "- The uploaded/local claim group must cite only uploaded/local source_id values.",
                "- Mention the concrete uploaded/local code detail, configuration, or parameter that supports the comparison.",
                "- Do not collapse the whole answer into only docs or only uploaded/local evidence when both routes are present.",
                "- If one route is too generic or weak, say that evidence is limited instead of inventing a stronger claim.",
            ]
        )
    else:
        lines.append(
            "- Avoid extraction-only responses unless the user explicitly asked for extraction."
        )
    if action_rules:
        lines.append("- Action requests:")
        lines.extend(f"  - {rule}" for rule in action_rules)
        lines.append("  - Do not merely say that you will save or share the answer; output the exact body now.")
        lines.append(
            "- If you are saving or sharing in this turn, return the actual message body to save/share now, not a sentence about performing the action."
        )
        lines.append(
            "- Do not answer with a checklist about the action itself unless the user explicitly asked that checklist to be the message body."
        )
    if attempt > 1:
        lines.append(
            "Retry after evidence validation failed. Stay grounded in retrieved evidence, keep only supported claims, and satisfy the requested answer structure."
        )
    return "\n".join(lines)


def build_synthesis_messages(
    *,
    state: GraphState,
    action_rules: list[str],
    deduped_evidence: list[dict[str, Any]],
    attempt: int,
    max_turns: int,
    snippet_char_limit: int = 400,
    evidence_char_budget: int | None = None,
) -> tuple[list[BaseMessage], int, int]:
    runtime = get_runtime_state(state)
    prompt_evidence = _prepare_evidence_for_prompt(
        deduped_evidence,
        snippet_char_limit=snippet_char_limit,
        evidence_char_budget=evidence_char_budget,
    )
    evidence_kinds = {
        str(item.get("kind") or "").strip().lower()
        for item in prompt_evidence
        if isinstance(item, dict)
    }
    has_hybrid_evidence = "official" in evidence_kinds and "local" in evidence_kinds
    answer_contract = infer_answer_contract(
        runtime.user_input,
        ["docs", "upload"] if has_hybrid_evidence else [],
    )
    history_messages = [
        message for message in state.get("messages", []) if not isinstance(message, ToolMessage)
    ]
    history_before = len(history_messages)
    trimmed_history = keep_recent_messages(history_messages, max_turns=max_turns)

    model_messages: list[BaseMessage] = [SystemMessage(content=SYS_POLICY)]
    if runtime.memory_summary:
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{runtime.memory_summary}"))
    model_messages.extend(trimmed_history)
    model_messages.append(
        SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(prompt_evidence, max_snippet_chars=snippet_char_limit)}")
    )
    model_messages.append(SystemMessage(content=render_answer_contract_prompt(answer_contract)))
    model_messages.append(
        SystemMessage(
            content=_build_synthesis_instruction_block(
                action_rules=action_rules,
                has_hybrid_evidence=has_hybrid_evidence,
                attempt=attempt,
            )
        )
    )

    return model_messages, history_before, len(trimmed_history)


def build_plain_summary_attach_messages(
    *,
    user_input: str,
    deduped_evidence: list[dict[str, Any]],
    snippet_char_limit: int = 400,
    evidence_char_budget: int | None = None,
) -> list[BaseMessage]:
    compact_evidence = _prepare_evidence_for_prompt(
        deduped_evidence[:2],
        snippet_char_limit=snippet_char_limit,
        evidence_char_budget=evidence_char_budget,
    )
    return [
        SystemMessage(content=SYS_POLICY),
        SystemMessage(content=PLAIN_SUMMARY_ATTACH_CONTRACT),
        HumanMessage(content=normalize_query_text(user_input) or "Summarize the retrieved evidence."),
        SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(compact_evidence, max_snippet_chars=snippet_char_limit)}"),
    ]


def parse_plain_summary_segments(content: str, *, limit: int) -> list[str]:
    raw_lines = str(content or "").replace("\r", "\n").splitlines()
    line_segments: list[str] = []
    for line in raw_lines:
        stripped = _LEADING_BULLET_PATTERN.sub("", line).strip()
        if not stripped:
            continue
        line_segments.append(stripped)
        if len(line_segments) >= limit:
            break
    if len(line_segments) >= limit or len(line_segments) > 1:
        return line_segments[:limit]

    normalized = " ".join(str(content or "").replace("\r", "\n").split()).strip()
    if not normalized:
        return []

    sentence_segments = [
        segment.strip()
        for segment in _SENTENCE_SPLIT_PATTERN.split(normalized)
        if segment.strip()
    ]
    if len(sentence_segments) >= 2:
        return sentence_segments[:limit]
    if line_segments:
        return line_segments[:1]
    return sentence_segments[:limit] if sentence_segments else [normalized]
