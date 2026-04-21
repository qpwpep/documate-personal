from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from openai.lib._pydantic import to_strict_json_schema

from src.core.answer_schema import SynthesisOutput
from src.runtime.nodes.session import extract_text_content


def _build_synthesis_response_schema() -> dict[str, Any]:
    return {
        # Pass a strict JSON schema dict instead of the Pydantic class itself so
        # Responses API parsing stays on the non-ParsedResponse path.
        "name": "SynthesisOutput",
        "strict": True,
        "schema": to_strict_json_schema(SynthesisOutput),
    }


def build_structured_synthesizer(llm_synthesizer: Any) -> Any:
    if hasattr(llm_synthesizer, "with_structured_output"):
        try:
            return llm_synthesizer.with_structured_output(
                _build_synthesis_response_schema(),
                method="json_schema",
                include_raw=True,
                strict=True,
            )
        except Exception:
            return llm_synthesizer
    return llm_synthesizer


def coerce_synthesis_output(raw_value: Any) -> SynthesisOutput:
    if isinstance(raw_value, SynthesisOutput):
        return raw_value
    if isinstance(raw_value, dict):
        try:
            return SynthesisOutput.model_validate(raw_value)
        except Exception:
            return SynthesisOutput(answer=str(raw_value))

    content = extract_text_content(getattr(raw_value, "content", raw_value))
    stripped = str(content or "").strip()
    if not stripped:
        return SynthesisOutput(answer="", claims=[], confidence=None)

    try:
        return SynthesisOutput.model_validate_json(stripped)
    except Exception:
        return SynthesisOutput(answer=stripped, claims=[], confidence=None)


def coerce_structured_synthesis_result(
    result: Any,
) -> tuple[Any, AIMessage | None, Exception | None]:
    if isinstance(result, AIMessage):
        return result, result, None
    if not isinstance(result, dict):
        return result, None, None

    if not {"raw", "parsed", "parsing_error"}.intersection(result.keys()):
        return result, None, None

    raw_message = result.get("raw")
    parsed = result.get("parsed")
    parsing_error = result.get("parsing_error")

    if isinstance(parsed, SynthesisOutput):
        parsed = parsed.model_dump(mode="json")

    if not isinstance(raw_message, AIMessage):
        raw_message = None

    if parsing_error is not None and isinstance(parsing_error, Exception):
        return parsed, raw_message, parsing_error
    if parsing_error is not None:
        return parsed, raw_message, RuntimeError(str(parsing_error))
    return parsed, raw_message, None
