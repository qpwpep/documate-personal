"""The internal execution result stays typed until the transport boundary."""
from __future__ import annotations

from dataclasses import dataclass

from src.core.contracts.debug import DebugPayload
from src.core.contracts.outcome import TurnResult


@dataclass(frozen=True, slots=True)
class AgentExecutionResult:
    outcome: TurnResult
    debug: DebugPayload
