from __future__ import annotations

import json
import math
from typing import Any


def _truncate_prompt_snippet(value: str, *, max_chars: int) -> str:
    snippet = str(value or "").strip()
    if not snippet or max_chars < 1 or len(snippet) <= max_chars:
        return snippet
    if max_chars <= 3:
        return snippet[:max_chars]
    bridge = " ... "
    if max_chars <= len(bridge) + 2:
        return snippet[: max_chars - 3].rstrip() + "..."

    available = max_chars - len(bridge)
    head_chars = max(1, available // 2)
    tail_chars = max(1, available - head_chars)
    head = snippet[:head_chars].rstrip()
    tail = snippet[-tail_chars:].lstrip()
    if head and tail:
        return f"{head}{bridge}{tail}"
    return snippet[: max_chars - 3].rstrip() + "..."


def _normalize_cell_id(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def _normalize_relevance_score(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return max(0.0, min(1.0, score))


def _is_local_evidence(item: dict[str, Any]) -> bool:
    return str(item.get("kind") or "").strip().lower() == "local"


def _coerce_code_metadata(value: Any) -> dict[str, Any]:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _coerce_doc_metadata(value: Any) -> dict[str, Any]:
    payload = value
    if isinstance(value, str):
        try:
            payload = json.loads(value)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


def _option_literals_from_code_metadata(code_metadata: dict[str, Any]) -> list[str]:
    options: list[str] = []
    seen: set[str] = set()
    for option in code_metadata.get("option_literals") or []:
        option_text = " ".join(str(option or "").split())
        compact = "".join(option_text.lower().split())
        if not option_text or compact in seen:
            continue
        options.append(option_text)
        seen.add(compact)
    return options


def _call_candidate(call: dict[str, Any]) -> str:
    call_name = str(call.get("call_name") or "").strip()
    if not call_name:
        return ""
    kwargs = call.get("kwargs")
    if not isinstance(kwargs, dict) or not kwargs:
        return call_name
    rendered_kwargs = ", ".join(
        f"{key}={value}"
        for key, value in kwargs.items()
        if str(key).strip() and str(value).strip()
    )
    return f"{call_name}({rendered_kwargs})" if rendered_kwargs else call_name


def _doc_entry_fact(prefix: str, entry: dict[str, Any]) -> str:
    name = str(entry.get("name") or "").strip()
    if not name:
        return ""
    detail = str(
        entry.get("description")
        or entry.get("type")
        or entry.get("default")
        or ""
    ).strip()
    return f"{prefix} {name}: {detail}" if detail else f"{prefix} {name}"


def _candidate_facts_from_doc_metadata(doc_metadata: dict[str, Any]) -> list[str]:
    facts: list[str] = []
    signature = str(doc_metadata.get("signature") or "").strip()
    if signature:
        facts.append(f"signature: {signature}")
    for key, prefix in (
        ("parameters", "param"),
        ("options", "option"),
        ("returns", "return"),
    ):
        for entry in doc_metadata.get(key) or []:
            if not isinstance(entry, dict):
                continue
            fact = _doc_entry_fact(prefix, entry)
            if fact:
                facts.append(fact)
            if len(facts) >= 10:
                return facts
    for note in doc_metadata.get("notes") or []:
        note_text = " ".join(str(note or "").split()).strip()
        if note_text:
            facts.append(f"note: {note_text}")
        if len(facts) >= 10:
            break
    return facts


def _candidate_facts_for_item(
    item: dict[str, Any],
    *,
    snippet: str,
    code_metadata: dict[str, Any],
    max_snippet_chars: int,
) -> list[str]:
    if _is_local_evidence(item):
        facts = _option_literals_from_code_metadata(code_metadata)
        if not facts:
            facts = [
                _call_candidate(call)
                for call in code_metadata.get("calls") or []
                if isinstance(call, dict)
            ]
        return [fact for fact in facts if fact][:4]

    doc_metadata = _coerce_doc_metadata(item.get("doc_metadata"))
    if doc_metadata:
        facts = _candidate_facts_from_doc_metadata(doc_metadata)
        if facts:
            return facts

    fact = _truncate_prompt_snippet(snippet, max_chars=min(max_snippet_chars, 220))
    return [fact] if fact else []


# Final guardrail for prompt rendering. Synthesis callers should pass
# max_snippet_chars explicitly so this formatter stays aligned with upstream budgets.
def format_evidence_for_prompt(
    items: list[dict[str, Any]],
    *,
    max_snippet_chars: int = 280,
    preserve_local_snippets: bool = False,
) -> str:
    if not items:
        return "No retrieved evidence."

    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        kind = str(item.get("kind") or "unknown")
        source = str(item.get("url_or_path") or "unknown-source")
        source_id = str(item.get("source_id") or "").strip()
        title = str(item.get("title") or "").strip()
        raw_snippet = str(item.get("snippet") or "").strip()
        snippet = raw_snippet if preserve_local_snippets and _is_local_evidence(item) else _truncate_prompt_snippet(
            raw_snippet,
            max_chars=max_snippet_chars,
        )
        code_metadata = _coerce_code_metadata(item.get("code_metadata"))
        doc_metadata = _coerce_doc_metadata(item.get("doc_metadata"))
        candidate_facts = _candidate_facts_for_item(
            item,
            snippet=snippet,
            code_metadata=code_metadata,
            max_snippet_chars=max_snippet_chars,
        )
        score = _normalize_relevance_score(item.get("score"))
        chunk_id = item.get("chunk_id")
        cell_id = _normalize_cell_id(item.get("cell_id"))
        start_offset = item.get("start_offset")
        end_offset = item.get("end_offset")
        header = f"{index}. [{kind}] {title} - {source}" if title else f"{index}. [{kind}] {source}"
        lines.append(header)
        if source_id:
            lines.append(f"   source_id: {source_id}")
        if score is not None:
            lines.append(f"   relevance_score: {score:.3f}")
        if cell_id is not None or chunk_id is not None:
            location_parts: list[str] = []
            if cell_id is not None:
                location_parts.append(f"cell_id={cell_id}")
            if chunk_id is not None:
                location_parts.append(f"chunk_id={chunk_id}")
            if start_offset is not None and end_offset is not None:
                location_parts.append(f"offsets={start_offset}-{end_offset}")
            if location_parts:
                lines.append(f"   location: {', '.join(location_parts)}")
        if candidate_facts:
            lines.append(f"   candidate_facts: {'; '.join(candidate_facts)}")
        if doc_metadata:
            doc_family = str(doc_metadata.get("doc_family") or "").strip()
            symbol = str(doc_metadata.get("symbol") or "").strip()
            signature = str(doc_metadata.get("signature") or "").strip()
            if doc_family:
                lines.append(f"   doc_family: {doc_family}")
            if symbol:
                lines.append(f"   api_symbol: {symbol}")
            if signature:
                lines.append(f"   signature: {signature}")
            parameter_facts = [
                _doc_entry_fact("param", entry)
                for entry in doc_metadata.get("parameters") or []
                if isinstance(entry, dict)
            ]
            option_facts = [
                _doc_entry_fact("option", entry)
                for entry in doc_metadata.get("options") or []
                if isinstance(entry, dict)
            ]
            if parameter_facts:
                lines.append(f"   parameter_facts: {'; '.join(parameter_facts[:10])}")
            if option_facts:
                lines.append(f"   option_facts: {'; '.join(option_facts[:10])}")
        if code_metadata:
            lines.append(
                "   code_metadata: "
                + json.dumps(code_metadata, ensure_ascii=False, sort_keys=True)
            )
        if snippet:
            lines.append(f"   snippet: {snippet}")
    return "\n".join(lines)
