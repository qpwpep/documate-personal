from __future__ import annotations

import logging
import time
from typing import Any

from ...contracts import GraphState
from ...contracts.boundary.debug import get_debug_state
from ...logging_utils import log_event
from .context import build_synthesis_context, prepare_synthesis_inputs
from .models import PreparedSynthesisInputs
from .payload_builder import build_structured_synthesizer
from .pipeline import run_synthesis_pipeline
from .short_circuit import maybe_short_circuit_synthesis
from .state import build_synthesis_updates


logger = logging.getLogger(__name__)


def _resolve_prompt_evidence_char_budget(
    *,
    synthesis_max_tokens: int,
    prompt_snippet_char_limit: int,
) -> tuple[int, int]:
    snippet_limit = min(prompt_snippet_char_limit, max(120, synthesis_max_tokens * 2))
    evidence_budget = max(700, min(2800, synthesis_max_tokens * 4))
    return snippet_limit, evidence_budget


def _resolve_compact_prompt_budgets(
    *,
    snippet_limit: int,
    evidence_budget: int,
) -> tuple[int, int]:
    return (
        max(80, int(snippet_limit) // 2),
        max(350, int(evidence_budget) // 2),
    )


def make_synthesize_node(
    llm_synthesizer: Any,
    llm_synthesizer_compact: Any | None = None,
    verbose: bool = False,
    max_turns: int = 6,
    synthesis_max_tokens: int = 900,
    prompt_snippet_char_limit: int = 400,
    has_default_slack_destination: bool = False,
):
    structured_synthesizer = build_structured_synthesizer(llm_synthesizer)
    structured_synthesizer_compact = (
        build_structured_synthesizer(llm_synthesizer_compact)
        if llm_synthesizer_compact is not None
        else None
    )
    effective_snippet_limit, evidence_char_budget = _resolve_prompt_evidence_char_budget(
        synthesis_max_tokens=synthesis_max_tokens,
        prompt_snippet_char_limit=prompt_snippet_char_limit,
    )
    compact_snippet_limit, compact_evidence_char_budget = _resolve_compact_prompt_budgets(
        snippet_limit=effective_snippet_limit,
        evidence_budget=evidence_char_budget,
    )

    def synthesize(state: GraphState) -> GraphState:
        stage_started = time.perf_counter()
        debug = get_debug_state(state)
        context = build_synthesis_context(
            state=state,
            has_default_slack_destination=has_default_slack_destination,
        )

        short_circuit = maybe_short_circuit_synthesis(
            state=state,
            debug=debug,
            context=context,
            stage_started=stage_started,
        )
        if short_circuit is not None:
            return short_circuit

        prepared = prepare_synthesis_inputs(
            state=state,
            context=context,
            max_turns=max_turns,
            prompt_snippet_char_limit=effective_snippet_limit,
            prompt_evidence_char_budget=evidence_char_budget,
        )
        if verbose and prepared.history_before != prepared.history_after:
            log_event(
                logger,
                logging.INFO,
                "synthesize_trimmed_messages",
                before=prepared.history_before,
                after=prepared.history_after,
            )

        compact_prepared: PreparedSynthesisInputs | None = None
        if structured_synthesizer_compact is not None:
            compact_prepared = prepare_synthesis_inputs(
                state=state,
                context=context,
                max_turns=max_turns,
                prompt_snippet_char_limit=compact_snippet_limit,
                prompt_evidence_char_budget=compact_evidence_char_budget,
            )

        pipeline_result = run_synthesis_pipeline(
            structured_synthesizer=structured_synthesizer,
            structured_synthesizer_compact=structured_synthesizer_compact,
            prepared=prepared,
            compact_prepared=compact_prepared,
            stage_started=stage_started,
        )
        return build_synthesis_updates(
            debug=debug,
            payload=pipeline_result.payload,
            synthesis_output=pipeline_result.synthesis_output,
            final_answer=pipeline_result.final_answer,
            attempt=prepared.attempt,
            latency_trace=pipeline_result.latency_trace,
            retrieval_errors=pipeline_result.retrieval_errors,
            planner_errors=pipeline_result.planner_errors,
            synthesis_errors=pipeline_result.synthesis_errors,
            llm_calls=pipeline_result.llm_calls,
        )

    return synthesize
