from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from openai.lib._pydantic import to_strict_json_schema

from src.core.answer_schema import AnswerDocument
from src.runtime.nodes.session import extract_text_content


def _build_synthesis_response_schema() -> dict[str, Any]:
    schema = to_strict_json_schema(AnswerDocument)

    def normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node
        # Each block has a distinct type tag, so anyOf and oneOf are equivalent.
        # OpenAI's wire subset supports nested anyOf; Pydantic retains its
        # discriminator internally for fast validation of the same document.
        result = {("anyOf" if key == "oneOf" else key): normalize(value)
                  for key, value in node.items() if key not in {"discriminator", "default", "const"}}
        if "const" in node:
            result["enum"] = [node["const"]]
        return result

    return {"name": "AnswerDocument", "strict": True, "schema": normalize(schema)}


def build_structured_synthesizer(llm_synthesizer: Any) -> Any:
    if hasattr(llm_synthesizer, "with_structured_output"):
        return llm_synthesizer.with_structured_output(
            _build_synthesis_response_schema(), method="json_schema", include_raw=True, strict=True,
        )
    return llm_synthesizer


def coerce_answer_document(raw_value: Any) -> AnswerDocument:
    """Only a valid document may become user-visible content."""
    if isinstance(raw_value, AnswerDocument):
        return raw_value
    if isinstance(raw_value, dict):
        return AnswerDocument.model_validate(raw_value)
    content = extract_text_content(getattr(raw_value, "content", raw_value))
    return AnswerDocument.model_validate_json(str(content or "").strip())


def coerce_structured_synthesis_result(result: Any) -> tuple[Any, AIMessage | None, Exception | None]:
    if isinstance(result, AIMessage):
        return result, result, None
    if not isinstance(result, dict) or not {"raw", "parsed", "parsing_error"}.intersection(result):
        return result, None, None
    raw = result.get("raw")
    parsed = result.get("parsed")
    error = result.get("parsing_error")
    if error is not None and not isinstance(error, Exception):
        error = RuntimeError(str(error))
    return parsed, raw if isinstance(raw, AIMessage) else None, error
