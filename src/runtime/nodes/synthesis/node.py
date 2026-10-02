from __future__ import annotations

import logging
import time
from typing import Any

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.infra.logging_utils import log_event
from src.runtime.nodes.synthesis.budgets import ExcerptLimits, resolve_evidence_budgets
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
from src.runtime.nodes.synthesis.pipeline import run_synthesis_pipeline
from src.runtime.nodes.synthesis.schema_adapter import build_structured_synthesizer
from src.runtime.nodes.synthesis.short_circuit import maybe_short_circuit_synthesis
from src.runtime.nodes.synthesis.state import build_synthesis_updates

logger = logging.getLogger(__name__)


def make_synthesize_node(
    llm_synthesizer: Any, llm_synthesizer_compact: Any | None = None,
    verbose: bool = False, max_turns: int = 6, *, excerpt_limits: ExcerptLimits,
):
    structured_synthesizer = build_structured_synthesizer(llm_synthesizer)
    structured_synthesizer_compact = (
        build_structured_synthesizer(llm_synthesizer_compact)
        if llm_synthesizer_compact is not None else None
    )

    def synthesize(state: GraphState) -> GraphState:
        started = time.perf_counter()
        debug = get_debug_state(state)
        context = build_synthesis_context(state=state)
        immediate = maybe_short_circuit_synthesis(state=state, debug=debug, context=context, stage_started=started)
        if immediate is not None:
            return immediate
        normal_budget, compact_budget = resolve_evidence_budgets(
            plan=context.planner_output, limits=excerpt_limits,
        )
        prepared = prepare_synthesis_inputs(
            state=state, context=context, budget=normal_budget, max_turns=max_turns,
        )
        emitter = get_runtime_state(state).progress_emitter
        if emitter is not None and hasattr(emitter, "emit_progress_snapshot"):
            emitter.emit_progress_snapshot(
                stage="synthesis", summary="원문 근거를 연결해 답변을 작성하는 중...",
                evidence_count=len(prepared.evidence_packet),
            )
        if verbose and prepared.history_before != prepared.history_after:
            log_event(logger, logging.INFO, "synthesize_trimmed_messages", before=prepared.history_before, after=prepared.history_after)
        compact = None
        if llm_synthesizer_compact is not None:
            compact = prepare_synthesis_inputs(
                state=state, context=context, budget=compact_budget, max_turns=max_turns,
            )
        outcome = run_synthesis_pipeline(
            structured_synthesizer=structured_synthesizer,
            structured_synthesizer_compact=structured_synthesizer_compact,
            prepared=prepared, compact_prepared=compact, stage_started=started,
        )
        return build_synthesis_updates(
            debug=debug, result=outcome.result, evidence_packet=outcome.evidence_packet,
            attempt=prepared.attempt, latency_trace=outcome.latency_trace,
            retrieval_errors=outcome.retrieval_errors, planner_errors=outcome.planner_errors,
            synthesis_errors=outcome.synthesis_errors,
            evidence_requirement_map=outcome.evidence_requirement_map,
            normal_evidence_missing_requirement_ids=outcome.normal_evidence_missing_requirement_ids,
            kind=outcome.kind,
            problem=outcome.problem,
            llm_diagnostics=outcome.llm_diagnostics,
            request_id=prepared.request_contract.request_id,
            contract_revision=prepared.request_contract.revision,
            body_kind=prepared.request_contract.body.kind,
            evidence_source=context.evidence_source,
        )
    return synthesize
