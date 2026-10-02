from __future__ import annotations

from typing import Any
from src.core.answer_schema import AnswerResponse, export_answer_text
from src.core.contracts.outcome import TurnResult
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.llm_errors import make_problem
from src.core.contracts.boundary.response import get_response_state


class ResponseAssembler:
    """Publish the checked document, never replace it with a later chat string."""

    def assemble(self, *, response: dict[str, Any], debug_info: dict[str, Any]) -> dict[str, Any]:
        state = get_response_state(response)
        result = AnswerResponse.model_validate(state.result.model_dump(mode="json"))
        if state.problem is not None or state.kind == "failure":
            problem = state.problem or make_problem("internal_error", "response_assembly")
            partial = bool(result.citations) and problem.code != "model_refusal"
            outcome = TurnResult(status="partial" if partial else "refused" if problem.code == "model_refusal" else "failed",
                                 response=result if partial else None, problem=problem, message=problem.message)
        elif state.kind == "clarification":
            contract = get_runtime_state(response).request_contract
            outcome = TurnResult(status="needs_input", message=export_answer_text(result),
                                 missing_slots=[item.slot for item in contract.missing_info] if contract else [])
        else:
            outcome = TurnResult(response=result)
        if outcome.response is None:
            debug_info = {**debug_info, "answer_provenance": None}
        return {**outcome.model_dump(mode="json"), "debug": debug_info}
