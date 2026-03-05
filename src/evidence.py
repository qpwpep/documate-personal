from __future__ import annotations

import json
import re
from typing import Any, Iterable, Literal
from urllib.parse import urlparse

from pydantic import BaseModel


EvidenceKind = Literal["official", "local"]


class EvidenceItem(BaseModel):
    kind: EvidenceKind
    tool: str
    source_id: str
    url_or_path: str
    title: str | None = None
    snippet: str | None = None
    score: float | None = None


def normalize_source_id(url_or_path: str) -> str:
    raw = str(url_or_path or "").strip()
    if not raw:
        return ""

    parsed = urlparse(raw)
    if parsed.scheme.lower() in {"http", "https"} and parsed.netloc:
        host = parsed.netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        path = re.sub(r"/+", "/", parsed.path or "/")
        if not path.startswith("/"):
            path = "/" + path
        return f"url:{parsed.scheme.lower()}://{host}{path}"

    normalized_path = re.sub(r"/+", "/", raw.replace("\\", "/")).strip().lower()
    return f"path:{normalized_path}"


def truncate_snippet(text: str | None, *, max_length: int = 500) -> str | None:
    if text is None:
        return None
    snippet = str(text).strip()
    if not snippet:
        return None
    if len(snippet) > max_length:
        return snippet[:max_length] + " ..."
    return snippet


def dedupe_evidence(items: Iterable[EvidenceItem]) -> list[EvidenceItem]:
    deduped: list[EvidenceItem] = []
    seen: set[tuple[str, str, str]] = set()
    for item in items:
        key = (item.kind, item.tool, item.source_id)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def evidence_to_dicts(items: Iterable[EvidenceItem]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def parse_evidence_payload(
    raw_payload: Any,
    *,
    context: str,
    errors: list[str] | None = None,
) -> list[EvidenceItem]:
    payload = raw_payload
    if isinstance(raw_payload, str):
        stripped = raw_payload.strip()
        if not stripped:
            return []
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            if errors is not None:
                errors.append(f"{context}: invalid JSON payload ({exc})")
            return []

    if isinstance(payload, dict):
        maybe_list = payload.get("evidence")
        if isinstance(maybe_list, list):
            payload = maybe_list
        else:
            if errors is not None:
                errors.append(f"{context}: payload must be a list of evidence objects")
            return []

    if not isinstance(payload, list):
        if errors is not None:
            errors.append(f"{context}: payload must be a list of evidence objects")
        return []

    parsed: list[EvidenceItem] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            if errors is not None:
                errors.append(f"{context}[{index}]: item must be an object")
            continue
        try:
            parsed.append(EvidenceItem.model_validate(item))
        except Exception as exc:
            if errors is not None:
                errors.append(f"{context}[{index}]: invalid evidence item ({exc})")

    return dedupe_evidence(parsed)
