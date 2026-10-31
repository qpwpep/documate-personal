from __future__ import annotations

from typing import Any

from src.infra.llm_boundary import bind_structured_output

from src.core.answer_schema import AnswerDocument


def build_structured_synthesizer(llm_synthesizer: Any) -> Any:
    return bind_structured_output(llm_synthesizer, AnswerDocument)


def coerce_answer_document(parsed: AnswerDocument | dict[str, Any]) -> AnswerDocument:
    """Validate the already-parsed document returned by the common LLM boundary."""
    return parsed if isinstance(parsed, AnswerDocument) else AnswerDocument.model_validate(parsed)
