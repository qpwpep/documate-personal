from __future__ import annotations

from src.core.contracts import GraphState, RuntimeState
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state, parse_planner_output
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.answer_schema import AnswerResponse
from src.core.evidence import SearchHit
from src.core.request_contracts import RequestContract, resolve_body_response
from src.runtime.nodes.synthesis.budgets import SynthesisBudgetProfile
from src.runtime.nodes.synthesis.evidence_selection import select_evidence_hits, tasks_for_hit
from src.runtime.nodes.synthesis.models import PreparedSynthesisInputs, SynthesisContext
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages, select_evidence_packet


def _build_action_rules(*, contract: RequestContract) -> list[str]:
    rules: list[str] = []
    if contract.action_requested("save_text"):
        rules.append("Produce the complete content to save in this turn. The server reports whether saving succeeds.")
    if contract.action_requested("slack_notify"):
        rules.append(
            "Produce the complete content to send. The server resolves the destination and reports delivery separately."
        )
    return rules


def resolve_contract_source_response(runtime: RuntimeState) -> AnswerResponse | None:
    """Resolve only the server-bound document revision named in the contract."""
    contract = runtime.request_contract
    if contract is None or not contract.can_prepare_body():
        return None
    return resolve_body_response(
        contract, previous_response=runtime.previous_response,
        pending_response=runtime.pending_action.response if runtime.pending_action else None,
    )


def build_synthesis_context(*, state: GraphState, has_default_slack_destination: bool) -> SynthesisContext:
    runtime = get_runtime_state(state)
    planner = get_planner_state(state)
    retry = get_retry_state(state)
    raw_hits = get_retrieval_state(state).hit_log[retry.hit_start_index:]
    parse_errors: list[str] = []
    hits: list[SearchHit] = []
    for index, raw in enumerate(raw_hits):
        try:
            hits.append(raw if isinstance(raw, SearchHit) else SearchHit.model_validate(raw))
        except Exception as exc:
            parse_errors.append(f"retrieved_hits[{index}]: {exc}")
    planner_errors: list[str] = []
    plan = parse_planner_output(planner.output, planner_errors)
    return SynthesisContext(
        attempt=get_response_state(state).synthesis_attempt + 1,
        user_input=runtime.user_input,
        messages=list(state.get("messages", [])),
        guided_followup=str(planner.guided_followup or "").strip(),
        planner_blocked=planner.diagnostics.reason in {"upload_retriever_missing", "planner_unavailable"},
        parse_errors=parse_errors,
        planner_parse_errors=planner_errors,
        planner_output=plan,
        retrieval_required=bool(plan.use_retrieval and plan.tasks),
        hits=select_evidence_hits(user_input=runtime.user_input, hits=hits, planner_output=plan),
        request_contract=runtime.request_contract,
        source_response=resolve_contract_source_response(runtime),
    )


def prepare_synthesis_inputs(
    *, state: GraphState, context: SynthesisContext, budget_profile: SynthesisBudgetProfile,
    max_turns: int, prompt_snippet_char_limit: int, prompt_evidence_char_budget: int | None,
) -> PreparedSynthesisInputs:
    if context.request_contract is None or not context.request_contract.can_prepare_body():
        raise ValueError("synthesis requires a prepared body request")
    requirements_by_evidence = {}
    for hit in context.hits:
        associated = requirements_by_evidence.setdefault(hit.evidence.id, {})
        for task in tasks_for_hit(hit, context.planner_output):
            associated.setdefault(task.requirement_id, task)
    packet, requirement_ids = select_evidence_packet(
        [hit.evidence for hit in context.hits],
        max_items=budget_profile.max_evidence_items,
        snippet_char_limit=prompt_snippet_char_limit,
        evidence_char_budget=budget_profile.evidence_chars if prompt_evidence_char_budget is None else prompt_evidence_char_budget,
        query=context.user_input,
        requirements_by_evidence={key: list(tasks.values()) for key, tasks in requirements_by_evidence.items()},
    )
    if context.source_response is not None:
        # The fixed source response retains complete citation ranges when transformed.
        packet = list({item.id: item for item in [
            *packet, *(citation.evidence for citation in context.source_response.citations),
        ]}.values())
    reference_aliases = {f"e{index}": source.id for index, source in enumerate(packet, 1)}
    retrieval_required = context.retrieval_required or bool(
        context.source_response and context.source_response.retrieval_required
    )
    messages, before, after = build_synthesis_messages(
        state=state,
        action_rules=_build_action_rules(contract=context.request_contract),
        evidence_packet=packet,
        attempt=context.attempt,
        max_turns=max_turns,
        requirement_ids_by_evidence=requirement_ids,
        reference_aliases=reference_aliases,
        source_response=context.source_response,
        retrieval_required=retrieval_required,
    )
    return PreparedSynthesisInputs(
        attempt=context.attempt, user_input=context.user_input, budget_profile=budget_profile,
        parse_errors=context.parse_errors, planner_parse_errors=context.planner_parse_errors,
        retrieval_required=retrieval_required, evidence_packet=packet,
        evidence_requirement_map=requirement_ids,
        reference_aliases=reference_aliases,
        model_messages=messages, history_before=before, history_after=after,
        request_contract=context.request_contract,
    )
