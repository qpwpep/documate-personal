"""A terminal turn result; technical failures are not answer documents."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.core.answer_schema import AnswerResponse
from src.core.llm_errors import ExecutionProblem


class TurnResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["completed", "needs_input", "failed", "partial", "refused"] = "completed"
    response: AnswerResponse | None = None
    message: str = ""
    problem: ExecutionProblem | None = None
    request_id: str = ""
    missing_slots: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def consistent_result(self) -> "TurnResult":
        if self.status in {"completed", "partial"} and self.response is None:
            raise ValueError("completed and partial results require a checked response")
        if self.status in {"failed", "refused", "needs_input"} and self.response is not None:
            raise ValueError("failure and clarification messages are not answer documents")
        if self.status in {"failed", "refused", "partial"}:
            if self.problem is None:
                raise ValueError("unsuccessful results require a problem")
        elif self.problem is not None:
            raise ValueError("successful interpretations cannot contain a technical problem")
        if self.status == "needs_input" and not self.message.strip():
            raise ValueError("clarification requires a specific question")
        if self.status == "refused" and self.problem.code != "model_refusal":
            raise ValueError("refused results require an explicit model refusal")
        return self
