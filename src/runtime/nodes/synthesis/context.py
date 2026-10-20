from __future__ import annotations

from src.core.contracts import GraphState, RuntimeState
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.provenance import AnswerSource
from src.core.answer_schema import AnswerResponse
from src.core.decisions import evidence_plan_fingerprint, prepared_packet_fingerprint
from src.core.llm_errors import LLMCallError, make_problem
from src.core.request_contracts import RequestContract, resolve_body_response
from src.runtime.nodes.synthesis.models import PreparedSynthesisInputs, SynthesisContext
from src.runtime.nodes.synthesis.prompt_builder import build_synthesis_messages


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


def build_synthesis_context(*, state: GraphState) -> SynthesisContext:
    runtime = get_runtime_state(state)
    planner = get_planner_state(state)
    retry = get_retry_state(state)
    retrieval = get_retrieval_state(state)
    hits = list(retrieval.hit_log[retry.hit_start_index:])
    plan = planner.output
    source_response = resolve_contract_source_response(runtime)
    prepared = retrieval.prepared_evidence
    contract = runtime.request_contract
    needs_prepared = (
        contract is not None and contract.can_prepare_body()
        and not contract.can_acknowledge() and not contract.can_cancel_pending()
        and planner.diagnostics.reason != "upload_retriever_missing"
    )
    if needs_prepared and (
        prepared is None
        or prepared.request_id != contract.request_id
        or prepared.contract_revision != contract.revision
        or prepared.plan_fingerprint != evidence_plan_fingerprint(plan)
        or prepared.source_response_hash != (source_response.content_hash if source_response else None)
        or any(packet.fingerprint != prepared_packet_fingerprint(packet.evidence, packet.requirement_map)
               for packet in (prepared.normal, prepared.compact) if packet is not None)
    ):
        raise LLMCallError(make_problem("internal_error", "synthesis"))
    evidence_source = None
    if source_response is not None:
        # Capture the actual bound source before actions can replace or clear pending state.
        evidence_source = AnswerSource(
            ref=runtime.request_contract.body.source.ref,
            response_hash=source_response.content_hash,
            citation_ids=[citation.evidence.id for citation in source_response.citations],
        )
    return SynthesisContext(
        attempt=get_response_state(state).synthesis_attempt + 1,
        user_input=runtime.user_input,
        messages=list(state.get("messages", [])),
        guided_followup=str(planner.guided_followup or "").strip(),
        planner_blocked=planner.diagnostics.reason == "upload_retriever_missing",
        planner_output=plan,
        retrieval_required=bool(plan.use_retrieval and plan.tasks),
        hits=hits,
        request_contract=runtime.request_contract,
        source_response=source_response,
        evidence_source=evidence_source,
        prepared_evidence=prepared,
    )


def prepare_synthesis_inputs(
    *, state: GraphState, context: SynthesisContext, max_turns: int, compact: bool = False,
) -> PreparedSynthesisInputs:
    if context.request_contract is None or not context.request_contract.can_prepare_body():
        raise ValueError("synthesis requires a prepared body request")
    prepared = context.prepared_evidence
    if prepared is None:
        raise LLMCallError(make_problem("internal_error", "synthesis"))
    selection = prepared.compact if compact else prepared.normal
    if selection is None:
        raise ValueError("compact evidence has not passed its own sufficiency checks")
    # The pre-validation stage owns all selection and semantic sufficiency.
    # Never shorten or re-rank an approved packet during message construction.
    packet = list(selection.evidence)
    requirement_ids = selection.requirement_map
    reference_aliases = {f"e{index}": source.id for index, source in enumerate(packet, 1)}
    retrieval_required = context.retrieval_required or bool(
        context.source_response and context.source_response.retrieval_required
    )
    messages, before, after = build_synthesis_messages(
        state=state,
        request_contract=context.request_contract,
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
        attempt=context.attempt, user_input=context.user_input,
        retrieval_required=retrieval_required, evidence_packet=packet,
        evidence_requirement_map=requirement_ids,
        reference_aliases=reference_aliases,
        model_messages=messages, history_before=before, history_after=after,
        request_contract=context.request_contract,
        missing_requirement_ids=list(selection.failed_requirement_ids),
    )
