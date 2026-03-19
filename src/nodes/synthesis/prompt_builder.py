from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ...contracts import GraphState
from ...contracts.boundary.runtime import get_runtime_state
from ...prompts import SYS_POLICY
from ..retrieval import format_evidence_for_prompt
from ..session import keep_recent_messages


SYNTHESIS_CONTRACT = (
    "[Synthesis Contract]\n"
    "- Return structured output only.\n"
    "- claims must be sentence-level.\n"
    "- Each claim must cite one or more exact evidence source_id values from Retrieved Evidence.\n"
    "- Do not invent evidence ids.\n"
    "- If the evidence is insufficient, return claims=[].\n"
    "- Do not embed citation numbers like [1] in claim text; the renderer adds them."
)
PLAIN_SUMMARY_ATTACH_CONTRACT = (
    "[Plain Summary Attach Fallback]\n"
    "- Use only Retrieved Evidence.\n"
    "- Return plain text only.\n"
    "- Return at most 2 lines or 2 sentences.\n"
    "- Each line or sentence must describe one grounded takeaway.\n"
    "- Keep the same order as Retrieved Evidence.\n"
    "- Do not include citations, bullets, numbering, JSON, or markdown."
)
_LEADING_BULLET_PATTERN = re.compile(r"^\s*(?:[-*??|\d+[.)])\s*")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def normalize_query_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split()).strip(" ,.;:-")


def build_synthesis_messages(
    *,
    state: GraphState,
    action_rules: list[str],
    deduped_evidence: list[dict[str, Any]],
    attempt: int,
    max_turns: int,
) -> tuple[list[BaseMessage], int, int]:
    runtime = get_runtime_state(state)
    history_messages = [
        message for message in state.get("messages", []) if not isinstance(message, ToolMessage)
    ]
    history_before = len(history_messages)
    trimmed_history = keep_recent_messages(history_messages, max_turns=max_turns)

    model_messages: list[BaseMessage] = [SystemMessage(content=SYS_POLICY)]
    if runtime.memory_summary:
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{runtime.memory_summary}"))
    model_messages.extend(trimmed_history)
    if action_rules:
        model_messages.append(SystemMessage(content="[Action Request]\n- " + "\n- ".join(action_rules)))
    model_messages.append(
        SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(deduped_evidence)}")
    )
    model_messages.append(SystemMessage(content=SYNTHESIS_CONTRACT))
    if attempt > 1:
        model_messages.append(
            SystemMessage(
                content=(
                    "Retry synthesis after evidence validation failed. "
                    "Use retrieved evidence when available and avoid unsupported claims."
                )
            )
        )

    return model_messages, history_before, len(trimmed_history)


def build_plain_summary_attach_messages(
    *,
    user_input: str,
    deduped_evidence: list[dict[str, Any]],
) -> list[BaseMessage]:
    compact_evidence = deduped_evidence[:2]
    return [
        SystemMessage(content=SYS_POLICY),
        SystemMessage(content=PLAIN_SUMMARY_ATTACH_CONTRACT),
        HumanMessage(content=normalize_query_text(user_input) or "Summarize the retrieved evidence."),
        SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(compact_evidence)}"),
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
