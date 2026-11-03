from __future__ import annotations

from typing import Any
from src.core.answer_schema import AnswerResponse
from src.core.answer_schema.rendering import render_answer_text
from src.core.contracts.outcome import TurnResult
from src.core.contracts.boundary.planner import get_planner_state
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
            missing_slots: list[str] = [item.slot for item in contract.missing_info] if contract else []
            if get_planner_state(response).diagnostics.reason == "upload_retriever_missing":
                missing_slots.append("upload")
            outcome = TurnResult(status="needs_input", message=render_answer_text(result),
                                 missing_slots=missing_slots)
        elif any(receipt.kind == "save_text" and receipt.status in {"error", "unknown"}
                 for receipt in result.actions):
            problem = make_problem("save_failed", "action_postprocess")
            outcome = TurnResult(status="partial", response=result, problem=problem, message=problem.message)
        else:
            outcome = TurnResult(response=result)
        if outcome.response is None:
            debug_info = {**debug_info, "answer_provenance": None}
        return {**outcome.model_dump(mode="json"), "debug": debug_info}
