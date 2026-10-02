from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from src.infra.structured_schema import compile_output_schema
from src.infra.llm_boundary import bind_structured_output

from src.core.answer_schema import AnswerDocument
from src.runtime.nodes.session import extract_text_content


def _build_synthesis_response_schema() -> dict[str, Any]:
    return compile_output_schema(AnswerDocument)


def build_structured_synthesizer(llm_synthesizer: Any) -> Any:
    return bind_structured_output(llm_synthesizer, AnswerDocument)


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
