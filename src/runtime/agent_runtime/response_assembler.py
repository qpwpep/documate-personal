from __future__ import annotations

from typing import Any
from src.core.answer_schema import AnswerResponse
from src.core.contracts.boundary.response import get_response_state


class ResponseAssembler:
    """Publish the checked document, never replace it with a later chat string."""

    def assemble(self, *, response: dict[str, Any], debug_info: dict[str, Any]) -> dict[str, Any]:
        state = get_response_state(response)
        result = AnswerResponse.model_validate(state.result.model_dump(mode="json"))
        return {"response": result.model_dump(mode="json"), "debug": debug_info}
