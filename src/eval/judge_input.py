"""Bounded judge transport views; original evidence remains unchanged."""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import tiktoken

from src.core.evidence import EvidenceRef, selected_source_anchors


JUDGE_INPUT_CONTRACT_VERSION = "evidence-selection-v1"
DEFAULT_JUDGE_MAX_INPUT_TOKENS = 128_000
DEFAULT_JUDGE_MAX_OUTPUT_TOKENS = 4_096
# Reserve more than the text-only Chat Completions framing estimate. The
# tokenizer count is still an estimate, not the provider's billing record.
MESSAGE_FRAMING_RESERVE_TOKENS = 256


@dataclass(frozen=True)
class PreparedJudgeInput:
    payload: dict[str, Any]
    json_text: str
    stats: dict[str, Any]


class JudgeInputBudgetError(ValueError):
    def __init__(self, stats: dict[str, Any]):
        self.stats = stats
        super().__init__(
            f"judge input needs approximately {stats['estimated_input_tokens']} tokens; "
            f"budget is {stats['max_input_tokens']}; required content was not truncated"
        )


def _evidence_view(evidence: EvidenceRef) -> dict[str, Any]:
    """Keep the exact selection, identity and location, never recapture a source."""
    element = evidence.element
    record = {
        "id": evidence.id,
        "snapshot": evidence.snapshot.model_dump(mode="json"),
        "element": {
            name: getattr(element, name)
            for name in ("element_id", "kind", "parent_id", "order", "heading_level", "heading_path", "language")
        },
        "selection": evidence.selection.model_dump(mode="json"),
        "excerpt": evidence.excerpt,
        "anchors": [anchor.model_dump(mode="json") for anchor in selected_source_anchors(evidence)],
    }
    if evidence.selection.cell_ids:
        selected = set(evidence.selection.cell_ids)
        record["is_partial"] = len(selected) < len(element.table.cells)
        record["selected_cells"] = [
            cell.model_dump(mode="json") for cell in element.table.cells if cell.cell_id in selected
        ]
    return record


def project_judge_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Replace repeated EvidenceRefs with references to an exact-selection table.

    This is a transport DTO, not a shortened EvidenceRef: its IDs and offsets
    still identify the original validated records. No source selections, answer
    text, oracle requirements, conversation turns, or diagnostics are truncated.
    """
    originals: dict[str, dict[str, Any]] = {}
    evidence_table: dict[str, dict[str, Any]] = {}

    def project(value: Any) -> Any:
        if isinstance(value, list):
            return [project(item) for item in value]
        if not isinstance(value, dict):
            return value
        if {"id", "snapshot", "element", "selection"} <= value.keys():
            identifier = value["id"]
            if not isinstance(identifier, str):
                raise ValueError("judge evidence identity must be a string")
            if identifier in originals:
                if originals[identifier] != value:
                    raise ValueError(f"conflicting judge evidence identity: {identifier}")
            else:
                evidence = EvidenceRef.model_validate(value)
                originals[identifier] = value
                evidence_table[identifier] = _evidence_view(evidence)
            return {"evidence_id": identifier}

        result = {}
        for key, item in value.items():
            # The final document already appears next to turn_result. Keep the
            # outcome and its problem, but reference that exact document once.
            if key == "turn_result" and isinstance(item, dict) and isinstance(value.get("response"), dict):
                if item.get("response") is not None:
                    if item["response"] != value["response"]:
                        raise ValueError("turn_result.response differs from its paired response")
                    item = {
                        **item,
                        "response": {"response_content_hash": value["response"]["content_hash"]},
                    }
            result[key] = project(item)
        return result

    result = project(payload)
    result["input_contract_version"] = JUDGE_INPUT_CONTRACT_VERSION
    result["evidence"] = evidence_table
    return result


def prepare_judge_input(
    payload: dict[str, Any], *, model_name: str, system_prompt: str,
    max_input_tokens: int = DEFAULT_JUDGE_MAX_INPUT_TOKENS,
    max_output_tokens: int = DEFAULT_JUDGE_MAX_OUTPUT_TOKENS,
) -> PreparedJudgeInput:
    """Budget the exact serialized input, including the system prompt and reserve."""
    if max_input_tokens < 1 or max_output_tokens < 1:
        raise ValueError("judge token budgets must be positive")
    projected = project_judge_payload(payload)
    serialized = json.dumps(projected, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    tokenizer_fallback = False
    tokenizer_error = None
    try:
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Same encoding fallback as LangChain for current OpenAI chat families.
            name = "o200k_base" if model_name.lower().startswith(("gpt-4o", "gpt-4.1", "gpt-5", "gpt-6")) else "cl100k_base"
            encoding = tiktoken.get_encoding(name)
            tokenizer_fallback = True
    except Exception as exc:
        # A cold tokenizer cache can require an unavailable external vocabulary
        # download. Byte-based BPE cannot exceed one token per UTF-8 byte.
        # Use that conservative bound instead of skipping the budget check.
        encoding = None
        tokenizer_fallback = True
        tokenizer_error = type(exc).__name__
    # Source text can contain strings that look like tokenizer special tokens.
    # They remain ordinary, untrusted document text rather than control tokens.
    estimated = MESSAGE_FRAMING_RESERVE_TOKENS + sum(
        len(encoding.encode(text, disallowed_special=())) if encoding is not None else len(text.encode("utf-8"))
        for text in (system_prompt, serialized)
    )
    stats = {
        "contract_version": JUDGE_INPUT_CONTRACT_VERSION,
        "tokenizer": encoding.name if encoding is not None else "utf8_byte_upper_bound",
        "tokenizer_fallback": tokenizer_fallback,
        "tokenizer_error": tokenizer_error,
        "estimated_input_tokens": estimated,
        "max_input_tokens": max_input_tokens,
        "output_token_limit": max_output_tokens,
        "message_framing_reserve_tokens": MESSAGE_FRAMING_RESERVE_TOKENS,
        "payload_bytes": len(serialized.encode("utf-8")),
        "evidence_count": len(projected["evidence"]),
        "truncated": False,
    }
    if estimated > max_input_tokens:
        raise JudgeInputBudgetError(stats)
    return PreparedJudgeInput(projected, serialized, stats)
