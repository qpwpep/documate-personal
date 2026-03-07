from __future__ import annotations

from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from ..evidence import dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from ..prompts import SYS_POLICY, needs_save, needs_slack
from .actions import build_action_only_answer, extract_slack_destinations, is_action_only_request
from .retrieval import format_evidence_for_prompt
from .session import extract_text_content, keep_recent_messages
from .state import State, coerce_retry_context, safe_list, slice_from_index


def make_synthesize_node(
    llm_synthesizer: Any,
    verbose: bool,
    max_turns: int = 6,
    has_default_slack_destination: bool = False,
):
    def synthesize(state: State):
        attempt = int(state.get("synthesis_attempt", 0)) + 1
        model_messages: List[BaseMessage] = [
            message for message in state.get("messages", []) if not isinstance(message, ToolMessage)
        ]

        if not model_messages or not isinstance(model_messages[0], SystemMessage):
            model_messages = [SystemMessage(content=SYS_POLICY)] + model_messages

        if state.get("memory_summary"):
            model_messages = [
                model_messages[0],
                SystemMessage(content=f"[Conversation Summary]\n{state['memory_summary']}"),
            ] + model_messages[1:]

        user_input = str(state.get("user_input", "") or "")
        guided_followup = str(state.get("guided_followup") or "").strip()
        explicit_slack_destinations = extract_slack_destinations(state.get("messages", []))
        slack_target_available = any(explicit_slack_destinations.values()) or has_default_slack_destination
        if guided_followup:
            response = AIMessage(content=guided_followup)
            return {
                "messages": [response],
                "final_answer": guided_followup,
                "synthesis_attempt": attempt,
                "needs_retry": False,
            }
        if is_action_only_request(user_input):
            final_answer = build_action_only_answer(
                user_input=user_input,
                messages=state.get("messages", []),
                slack_target_available=slack_target_available,
            )
            response = AIMessage(content=final_answer)
            return {
                "messages": [response],
                "final_answer": final_answer,
                "synthesis_attempt": attempt,
                "needs_retry": False,
            }

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
        if action_rules:
            model_messages.append(SystemMessage(content="[Action Request]\n- " + "\n- ".join(action_rules)))

        retry_context = coerce_retry_context(state.get("retry_context"))
        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        current_attempt_evidence_payload = slice_from_index(
            safe_list(state.get("retrieved_evidence")),
            evidence_start_index,
        )

        parse_errors: list[str] = []
        parsed_evidence = parse_evidence_payload(
            current_attempt_evidence_payload,
            context="retrieved_evidence",
            errors=parse_errors,
        )
        deduped_evidence = evidence_to_dicts(dedupe_evidence(parsed_evidence))

        model_messages.append(
            SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(deduped_evidence)}")
        )
        if attempt > 1:
            model_messages.append(
                SystemMessage(
                    content=(
                        "Retry synthesis after evidence validation failed. "
                        "Use retrieved evidence when available and avoid unsupported claims."
                    )
                )
            )

        before = len(model_messages)
        model_messages = keep_recent_messages(model_messages, max_turns=max_turns)
        after = len(model_messages)
        if verbose and before != after:
            print(f"[synthesize] trimmed model messages: {before} -> {after}")

        response_obj = llm_synthesizer.invoke(model_messages)
        if isinstance(response_obj, AIMessage):
            response = response_obj
        else:
            response = AIMessage(content=extract_text_content(getattr(response_obj, "content", response_obj)))
        final_answer = extract_text_content(response.content)

        updates: State = {
            "messages": [response],
            "final_answer": final_answer,
            "synthesis_attempt": attempt,
            "needs_retry": False,
        }
        if parse_errors:
            updates["retrieval_errors"] = parse_errors
        return updates

    return synthesize
