from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from src.core.answer_schema import clean_grounded_text
from src.core.contracts import GraphState
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.request_contracts import AnswerContract, infer_answer_contract
from src.core.prompts import SYS_POLICY
from src.runtime.nodes.retrieval import format_evidence_for_prompt
from src.runtime.nodes.session import keep_recent_messages


SYNTHESIS_OUTPUT_TEMPLATE = (
    "[Synthesis Output Template]\n"
    "Return exactly one structured SynthesisOutput object.\n"
    "answer: concise response in the user's language.\n"
    "claims: 1-4 sentence-level claims, each with exact Retrieved Evidence source_id values.\n"
    "sections: use [] unless Turn Contract required_sections lists section kinds; include only those exact kinds.\n"
    "Never write placeholder references such as 'see above code' or '위 코드 참고'; include the concrete content instead.\n"
    "confidence: 0.0-1.0 when supported, otherwise null.\n"
    "Citations are supplied through evidence_ids; the renderer adds citation labels."
)
SYNTHESIS_SELECTION_TEMPLATE = (
    "[Selection And Assembly Mode]\n"
    "Treat Retrieved Evidence as a candidate pool, not as prose to rewrite freely.\n"
    "Prefer candidate_facts and code_metadata over raw snippets for code/options.\n"
    "Select supported facts, assemble short claims, then map them into requested sections.\n"
    "For hybrid docs plus upload/local answers: official_docs first, uploaded/local detail next, comparison last."
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


def _is_explicit_code_extraction_request(text: str) -> bool:
    normalized = str(text or "").strip().lower()
    return any(
        marker in normalized
        for marker in (
            "extract",
            "quote",
            "snippet",
            "verbatim",
            "exact",
            "raw code",
            "code snippet",
            "cell",
            "line",
        )
    )


def _prepare_evidence_for_prompt(
    items: list[dict[str, Any]],
    *,
    snippet_char_limit: int,
    evidence_char_budget: int | None = None,
    preserve_local_snippets: bool = False,
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
        if kind != "local" or not preserve_local_snippets:
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


def _render_sections(sections: list[str]) -> str:
    return ", ".join(sections) if sections else "none"


def _infer_required_routes_from_evidence(items: list[dict[str, Any]]) -> list[str]:
    routes: list[str] = []
    has_official = any(
        str(item.get("kind") or "").strip().lower() == "official"
        for item in items
        if isinstance(item, dict)
    )
    if has_official:
        routes.append("docs")

    local_tools = {
        str(item.get("tool") or "").strip()
        for item in items
        if isinstance(item, dict)
        and str(item.get("kind") or "").strip().lower() == "local"
    }
    if "upload_search" in local_tools:
        routes.append("upload")
    elif local_tools:
        routes.append("local")
    return routes


def _build_turn_contract_block(
    *,
    answer_contract: AnswerContract,
    action_rules: list[str],
    has_hybrid_evidence: bool,
    requires_upload_section: bool,
    attempt: int,
) -> str:
    lines = [
        "[Turn Contract]",
        f"- required_sections={_render_sections(answer_contract.required_sections)}",
        f"- ordered_steps={str(answer_contract.ordered_steps).lower()}",
        f"- split_by_source={str(answer_contract.split_by_source).lower()}",
        f"- hybrid_evidence={str(has_hybrid_evidence).lower()}",
        f"- upload_section_required={str(requires_upload_section).lower()}",
    ]
    if has_hybrid_evidence:
        lines.append("- hybrid_layout=official_docs -> upload/local detail -> comparison")
        if requires_upload_section:
            lines.append("- upload_code uses local/upload option_literals or call kwargs when present")
        else:
            lines.append("- put uploaded/local details inside the comparison section")
    if "code_example" in answer_contract.required_sections:
        lines.append("- code_block_required=true")
        lines.append("- code_example section must include at least one fenced code block with concrete sample code")
        lines.append("- explain the code briefly outside the code block; do not answer with prose only")
    if "options" in answer_contract.required_sections:
        lines.append("- options_section_required=true")
        lines.append("- options section should be grouped concise bullets, not one long paragraph")
        lines.append("- use exact option/parameter names from candidate_facts or doc_metadata when available")
    if action_rules:
        lines.append("- action_rules:")
        lines.extend(f"  - {rule}" for rule in action_rules)
    if attempt > 1:
        lines.append("- retry_note=evidence validation failed previously; keep only supported claims")
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
    preserve_local_snippets = _is_explicit_code_extraction_request(runtime.user_input)
    prompt_evidence = _prepare_evidence_for_prompt(
        deduped_evidence,
        snippet_char_limit=snippet_char_limit,
        evidence_char_budget=evidence_char_budget,
        preserve_local_snippets=preserve_local_snippets,
    )
    evidence_kinds = {
        str(item.get("kind") or "").strip().lower()
        for item in prompt_evidence
        if isinstance(item, dict)
    }
    has_hybrid_evidence = "official" in evidence_kinds and "local" in evidence_kinds
    answer_contract = infer_answer_contract(
        runtime.user_input,
        _infer_required_routes_from_evidence(prompt_evidence),
    )
    requires_upload_section = "upload_code" in answer_contract.required_sections
    history_messages = [
        message for message in state.get("messages", []) if not isinstance(message, ToolMessage)
    ]
    history_before = len(history_messages)
    trimmed_history = keep_recent_messages(history_messages, max_turns=max_turns)

    model_messages: list[BaseMessage] = [
        SystemMessage(content=SYS_POLICY),
        SystemMessage(content=SYNTHESIS_OUTPUT_TEMPLATE),
        SystemMessage(content=SYNTHESIS_SELECTION_TEMPLATE),
        SystemMessage(
            content=_build_turn_contract_block(
                answer_contract=answer_contract,
                action_rules=action_rules,
                has_hybrid_evidence=has_hybrid_evidence,
                requires_upload_section=requires_upload_section,
                attempt=attempt,
            )
        ),
    ]
    if runtime.memory_summary:
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{runtime.memory_summary}"))
    model_messages.extend(trimmed_history)
    rendered_prompt_evidence = format_evidence_for_prompt(
        prompt_evidence,
        max_snippet_chars=snippet_char_limit,
        preserve_local_snippets=preserve_local_snippets,
    )
    model_messages.append(
        SystemMessage(
            content=f"[Retrieved Evidence]\n{rendered_prompt_evidence}"
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
        preserve_local_snippets=_is_explicit_code_extraction_request(user_input),
    )
    preserve_local_snippets = _is_explicit_code_extraction_request(user_input)
    rendered_compact_evidence = format_evidence_for_prompt(
        compact_evidence,
        max_snippet_chars=snippet_char_limit,
        preserve_local_snippets=preserve_local_snippets,
    )
    return [
        SystemMessage(content=SYS_POLICY),
        SystemMessage(content=PLAIN_SUMMARY_ATTACH_CONTRACT),
        HumanMessage(content=normalize_query_text(user_input) or "Summarize the retrieved evidence."),
        SystemMessage(
            content=f"[Retrieved Evidence]\n{rendered_compact_evidence}"
        ),
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
