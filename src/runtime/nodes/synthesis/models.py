from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import BaseMessage

from src.core.answer_schema import AnswerResponse
from src.core.contracts.debug import LLMCallMetadata
from src.core.evidence import EvidenceRef, SearchHit
from src.core.planner_schema import PlannerOutput
from src.core.request_contracts import RequestContract
from src.runtime.nodes.synthesis.budgets import SynthesisBudgetProfile


@dataclass(slots=True)
class SynthesisContext:
    attempt: int
    user_input: str
    messages: list[Any]
    guided_followup: str
    planner_blocked: bool
    parse_errors: list[str]
    planner_parse_errors: list[str]
    planner_output: PlannerOutput
    retrieval_required: bool
    hits: list[SearchHit]
    request_contract: RequestContract | None
    source_response: AnswerResponse | None


@dataclass(slots=True)
class PreparedSynthesisInputs:
    attempt: int
    user_input: str
    budget_profile: SynthesisBudgetProfile
    parse_errors: list[str]
    planner_parse_errors: list[str]
    retrieval_required: bool
    evidence_packet: list[EvidenceRef]
    model_messages: list[BaseMessage]
    history_before: int
    history_after: int
    request_contract: RequestContract
    evidence_requirement_map: dict[str, list[str]] = field(default_factory=dict)
    reference_aliases: dict[str, str] = field(default_factory=dict)
    missing_requirement_ids: list[str] = field(default_factory=list)


@dataclass(slots=True)
class SynthesisPipelineResult:
    result: AnswerResponse
    evidence_packet: list[EvidenceRef]
    latency_trace: list[dict[str, Any]]
    evidence_requirement_map: dict[str, list[str]] = field(default_factory=dict)
    retrieval_errors: list[str] = field(default_factory=list)
    planner_errors: list[str] = field(default_factory=list)
    synthesis_errors: list[str] = field(default_factory=list)
    llm_calls: list[LLMCallMetadata] = field(default_factory=list)
    kind: str = "draft"
    normal_evidence_missing_requirement_ids: list[str] | None = None
