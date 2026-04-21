from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from src.core.answer_schema import AgentResponsePayloadModel, SynthesisOutput
from src.core.contracts.debug import LLMCallMetadata
from src.core.evidence import EvidenceItem
from src.core.planner_schema import PlannerOutput


@dataclass(slots=True)
class SynthesisContext:
    attempt: int
    user_input: str
    messages: list[Any]
    guided_followup: str
    slack_target_available: bool
    parse_errors: list[str]
    planner_parse_errors: list[str]
    planner_output: PlannerOutput
    retrieval_required: bool
    primary_evidence_items: list[EvidenceItem]
    grounded_fallback_evidence_items: list[EvidenceItem]


@dataclass(slots=True)
class PreparedSynthesisInputs:
    attempt: int
    user_input: str
    parse_errors: list[str]
    planner_parse_errors: list[str]
    retrieval_required: bool
    primary_evidence_items: list[EvidenceItem]
    grounded_fallback_evidence_items: list[EvidenceItem]
    deduped_evidence: list[dict[str, Any]]
    model_messages: list[BaseMessage]
    history_before: int
    history_after: int


@dataclass(slots=True)
class SynthesisPipelineResult:
    payload: AgentResponsePayloadModel
    synthesis_output: SynthesisOutput
    final_answer: str
    latency_trace: list[dict[str, Any]]
    retrieval_errors: list[str] = field(default_factory=list)
    planner_errors: list[str] = field(default_factory=list)
    synthesis_errors: list[str] = field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = field(default_factory=list)
