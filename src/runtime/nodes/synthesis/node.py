from __future__ import annotations

import logging
import time
from typing import Any

from src.core.contracts import GraphState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.request_contracts import infer_answer_contract
from src.infra.logging_utils import log_event
from src.runtime.nodes.synthesis.budgets import compact_synthesis_budget_profile, resolve_synthesis_budget_profile
from src.runtime.nodes.synthesis.context import build_synthesis_context, prepare_synthesis_inputs
from src.runtime.nodes.synthesis.models import PreparedSynthesisInputs
from src.runtime.nodes.synthesis.pipeline import run_synthesis_pipeline
from src.runtime.nodes.synthesis.schema_adapter import build_structured_synthesizer
from src.runtime.nodes.synthesis.short_circuit import maybe_short_circuit_synthesis
from src.runtime.nodes.synthesis.state import build_synthesis_updates


logger = logging.getLogger(__name__)


def _bind_synthesizer_max_tokens(llm_synthesizer: Any, *, max_tokens: int) -> Any:
    if hasattr(llm_synthesizer, "bind"):
        try:
            return llm_synthesizer.bind(max_tokens=max(1, int(max_tokens)))
        except Exception:
            return llm_synthesizer
    return llm_synthesizer


def make_synthesize_node(
    llm_synthesizer: Any,
    llm_synthesizer_compact: Any | None = None,
    verbose: bool = False,
    max_turns: int = 6,
    synthesis_max_tokens: int = 900,
    prompt_snippet_char_limit: int = 400,
    has_default_slack_destination: bool = False,
):
    structured_synthesizer_cache: dict[int, Any] = {}
    compact_structured_synthesizer_cache: dict[int, Any] = {}

    def structured_synthesizer_for(max_tokens: int) -> Any:
        cache_key = max(1, int(max_tokens))
        if cache_key not in structured_synthesizer_cache:
            structured_synthesizer_cache[cache_key] = build_structured_synthesizer(
                _bind_synthesizer_max_tokens(llm_synthesizer, max_tokens=cache_key)
            )
        return structured_synthesizer_cache[cache_key]

    def compact_structured_synthesizer_for(max_tokens: int) -> Any | None:
        if llm_synthesizer_compact is None:
            return None
        cache_key = max(1, int(max_tokens))
        if cache_key not in compact_structured_synthesizer_cache:
            compact_structured_synthesizer_cache[cache_key] = build_structured_synthesizer(
                _bind_synthesizer_max_tokens(llm_synthesizer_compact, max_tokens=cache_key)
            )
        return compact_structured_synthesizer_cache[cache_key]

    def synthesize(state: GraphState) -> GraphState:
        stage_started = time.perf_counter()
        debug = get_debug_state(state)
        context = build_synthesis_context(
            state=state,
            has_default_slack_destination=has_default_slack_destination,
        )
        budget_profile = resolve_synthesis_budget_profile(
            user_input=context.user_input,
            planner_output=context.planner_output,
            synthesis_max_tokens=synthesis_max_tokens,
        )
        compact_budget_profile = compact_synthesis_budget_profile(budget_profile)
        structured_synthesizer = structured_synthesizer_for(budget_profile.max_tokens)
        structured_synthesizer_compact = compact_structured_synthesizer_for(
            compact_budget_profile.max_tokens
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
            budget_profile=budget_profile,
            max_turns=max_turns,
            prompt_snippet_char_limit=min(prompt_snippet_char_limit, budget_profile.snippet_chars),
            prompt_evidence_char_budget=budget_profile.evidence_chars,
        )
        progress_emitter = get_runtime_state(state).progress_emitter
        if progress_emitter is not None and hasattr(progress_emitter, "emit_progress_snapshot"):
            required_routes = [task.route for task in context.planner_output.tasks]
            contract = infer_answer_contract(context.user_input, required_routes)
            sections = list(contract.required_sections)
            progress_emitter.emit_progress_snapshot(
                stage="synthesis",
                summary=(
                    "답변 섹션 준비: " + ", ".join(sections)
                    if sections
                    else "답변 근거를 종합하는 중..."
                ),
                sections=sections,
                evidence_count=len(prepared.deduped_evidence),
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
                budget_profile=compact_budget_profile,
                max_turns=max_turns,
                prompt_snippet_char_limit=min(prompt_snippet_char_limit, compact_budget_profile.snippet_chars),
                prompt_evidence_char_budget=compact_budget_profile.evidence_chars,
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
