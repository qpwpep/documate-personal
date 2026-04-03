from __future__ import annotations

from ...contracts import GraphState
from ...contracts.boundary.graph import get_retry_state
from ...contracts.boundary.planner import get_planner_state, parse_planner_output
from ...contracts.boundary.response import get_response_state
from ...contracts.boundary.retrieval import get_retrieval_state
from ...contracts.boundary.runtime import get_runtime_state
from ...evidence import EvidenceItem, dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from ...prompts import needs_save, needs_slack
from ...sequence_utils import slice_from_index
from ..actions import get_slack_destinations
from .models import PreparedSynthesisInputs, SynthesisContext
from .payload_builder import select_grounded_fallback_evidence_items, select_primary_evidence_items
from .prompt_builder import build_synthesis_messages


def _build_action_rules(*, user_input: str, slack_target_available: bool) -> list[str]:
    action_rules: list[str] = []
    if needs_save(user_input):
        action_rules.append(
            "The user requested saving. Produce the final answer content to save now. "
            "Do not ask follow-up questions about what to save. If the target is unspecified, "
            "the save target is the final answer you are generating in this turn."
        )
    if needs_slack(user_input):
        if slack_target_available:
            action_rules.append(
                "A Slack destination is available. Produce the final message body to send now and "
                "do not ask for destination confirmation."
            )
        else:
            action_rules.append(
                "No Slack destination is available yet. Ask one concise follow-up asking only for "
                "channel_id, user_id, or email. Do not ask for message content."
            )
    return action_rules


def build_synthesis_context(
    *,
    state: GraphState,
    has_default_slack_destination: bool,
) -> SynthesisContext:
    runtime = get_runtime_state(state)
    planner = get_planner_state(state)
    retrieval = get_retrieval_state(state)
    response = get_response_state(state)
    retry_context = get_retry_state(state)

    user_input = runtime.user_input
    attempt = int(response.synthesis_attempt) + 1
    guided_followup = str(planner.guided_followup or "").strip()
    explicit_slack_destinations = get_slack_destinations(runtime.session_metadata)
    slack_target_available = (
        explicit_slack_destinations.has_destination() or has_default_slack_destination
    )

    evidence_start_index = int(retry_context.evidence_start_index)
    current_attempt_evidence_payload = slice_from_index(
        retrieval.evidence_log,
        evidence_start_index,
    )
    parse_errors: list[str] = []
    parsed_evidence = dedupe_evidence(
        parse_evidence_payload(
            current_attempt_evidence_payload,
            context="retrieved_evidence",
            errors=parse_errors,
        )
    )
    evidence_items = [item for item in parsed_evidence if isinstance(item, EvidenceItem)]

    planner_parse_errors: list[str] = []
    planner_output = parse_planner_output(planner.output, planner_parse_errors)
    retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)

    return SynthesisContext(
        attempt=attempt,
        user_input=user_input,
        messages=list(state.get("messages", [])),
        guided_followup=guided_followup,
        slack_target_available=slack_target_available,
        parse_errors=parse_errors,
        planner_parse_errors=planner_parse_errors,
        planner_output=planner_output,
        retrieval_required=retrieval_required,
        primary_evidence_items=select_primary_evidence_items(
            user_input=user_input,
            evidence_items=evidence_items,
            planner_output=planner_output,
        ),
        grounded_fallback_evidence_items=select_grounded_fallback_evidence_items(
            user_input=user_input,
            evidence_items=evidence_items,
            planner_output=planner_output,
        ),
    )


def prepare_synthesis_inputs(
    *,
    state: GraphState,
    context: SynthesisContext,
    max_turns: int,
    prompt_snippet_char_limit: int,
    prompt_evidence_char_budget: int | None,
) -> PreparedSynthesisInputs:
    deduped_evidence = evidence_to_dicts(context.primary_evidence_items)
    model_messages, history_before, history_after = build_synthesis_messages(
        state=state,
        action_rules=_build_action_rules(
            user_input=context.user_input,
            slack_target_available=context.slack_target_available,
        ),
        deduped_evidence=deduped_evidence,
        attempt=context.attempt,
        max_turns=max_turns,
        snippet_char_limit=prompt_snippet_char_limit,
        evidence_char_budget=prompt_evidence_char_budget,
    )
    return PreparedSynthesisInputs(
        attempt=context.attempt,
        user_input=context.user_input,
        parse_errors=context.parse_errors,
        planner_parse_errors=context.planner_parse_errors,
        retrieval_required=context.retrieval_required,
        primary_evidence_items=context.primary_evidence_items,
        grounded_fallback_evidence_items=context.grounded_fallback_evidence_items,
        deduped_evidence=deduped_evidence,
        model_messages=model_messages,
        history_before=history_before,
        history_after=history_after,
    )
