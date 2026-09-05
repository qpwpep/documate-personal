from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from src.core.answer_schema import clean_grounded_text
from src.core.conversation_memory import build_untrusted_memory_prompt_messages
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
    "For hybrid docs plus upload answers: official_docs first, uploaded detail next, comparison last."
)
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
    if answer_contract.required_sections:
        lines.append("- answer_field=overview; put source-specific details in sections")
        lines.append("- do not duplicate section bodies in answer")
    if has_hybrid_evidence:
        has_comparison_section = "comparison" in answer_contract.required_sections
        if has_comparison_section:
            lines.append("- hybrid_layout=official_docs -> upload detail -> comparison")
        else:
            lines.append("- hybrid_layout=cover official docs and upload details inside the requested sections")
        lines.append("- each hybrid section should include the necessary supported details without duplicating other sections")
        if requires_upload_section:
            lines.append("- upload_code uses upload option_literals or call kwargs when present")
        elif has_comparison_section:
            lines.append("- put uploaded details inside the comparison section")
    if "code_example" in answer_contract.required_sections:
        lines.append("- code_block_required=true")
        lines.append("- code_example section must include at least one fenced code block with concrete sample code")
        lines.append("- explain the code briefly outside the code block; do not answer with prose only")
    if "options" in answer_contract.required_sections:
        lines.append("- options_section_required=true")
        lines.append("- options section should be grouped concise bullets, not one long paragraph")
        lines.append("- use exact option/parameter names from candidate_facts or doc_metadata when available")
        lines.append("- options_answer_policy=answer first with confirmed items, then note evidence gaps")
        lines.append("- do not replace the requested options summary with a refusal or broad insufficiency caveat")
        lines.append("- if evidence is partial, list supported options/parameters and mark only uncertain entries as needs_more_evidence")
        lines.append("- if evidence shows a wrapper/delegated API relationship, mention it and use the delegated API docs when available")
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
        model_messages.extend(
            build_untrusted_memory_prompt_messages(runtime.memory_summary)
        )
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
