from __future__ import annotations

import logging
import time
from typing import Any, List

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage

from ..answer_schema import (
    AgentResponsePayloadModel,
    ClaimItem,
    SynthesisOutput,
    average_claim_confidence,
    build_empty_response_payload,
    render_payload_from_claims,
)
from ..evidence import dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from ..latency import (
    elapsed_ms,
    make_stage_latency_event,
    make_synthesis_attempt_latency_event,
)
from ..logging_utils import log_event
from ..prompts import SYS_POLICY, needs_save, needs_slack
from .actions import build_action_only_answer, get_slack_destinations, is_action_only_request
from .retrieval import format_evidence_for_prompt
from .session import extract_text_content, keep_recent_messages
from .state import (
    LLMCallMetadata,
    State,
    build_llm_call_metadata,
    coerce_planner_output,
    coerce_retry_context,
    safe_list,
    slice_from_index,
)


logger = logging.getLogger(__name__)


SYNTHESIS_CONTRACT = (
    "[Synthesis Contract]\n"
    "- Return structured output only.\n"
    "- claims must be sentence-level.\n"
    "- Each claim must cite one or more exact evidence source_id values from Retrieved Evidence.\n"
    "- Do not invent evidence ids.\n"
    "- If the evidence is insufficient, return claims=[].\n"
    "- Do not embed citation numbers like [1] in claim text; the renderer adds them."
)


def _build_structured_synthesizer(llm_synthesizer: Any) -> Any:
    if hasattr(llm_synthesizer, "with_structured_output"):
        try:
            return llm_synthesizer.with_structured_output(
                SynthesisOutput,
                method="json_schema",
                include_raw=True,
                strict=True,
            )
        except Exception:
            return llm_synthesizer
    return llm_synthesizer


def _coerce_synthesis_output(raw_value: Any) -> SynthesisOutput:
    if isinstance(raw_value, SynthesisOutput):
        return raw_value
    if isinstance(raw_value, dict):
        try:
            return SynthesisOutput.model_validate(raw_value)
        except Exception:
            return SynthesisOutput(answer=str(raw_value))

    content = extract_text_content(getattr(raw_value, "content", raw_value))
    stripped = str(content or "").strip()
    if not stripped:
        return SynthesisOutput(answer="", claims=[], confidence=None)

    try:
        return SynthesisOutput.model_validate_json(stripped)
    except Exception:
        return SynthesisOutput(answer=stripped, claims=[], confidence=None)


def _coerce_structured_synthesis_result(
    result: Any,
) -> tuple[Any, AIMessage | None, Exception | None]:
    if not isinstance(result, dict):
        return result, None, None

    if not {"raw", "parsed", "parsing_error"}.intersection(result.keys()):
        return result, None, None

    raw_message = result.get("raw")
    parsed = result.get("parsed")
    parsing_error = result.get("parsing_error")

    if not isinstance(raw_message, AIMessage):
        raw_message = None

    if parsing_error is not None and isinstance(parsing_error, Exception):
        return parsed, raw_message, parsing_error
    if parsing_error is not None:
        return parsed, raw_message, RuntimeError(str(parsing_error))
    return parsed, raw_message, None


def _payload_to_state_dict(payload: AgentResponsePayloadModel) -> dict[str, Any]:
    return payload.model_dump(mode="json")


def _is_timeout_error(exc: Exception) -> bool:
    if isinstance(exc, TimeoutError):
        return True
    lowered = str(exc).lower()
    return "timeout" in lowered or "timed out" in lowered


def _build_grounded_timeout_payload(evidence_items: list[Any]) -> AgentResponsePayloadModel:
    grounded_claims: list[ClaimItem] = []
    for item in evidence_items[:2]:
        source_id = str(getattr(item, "source_id", "") or "").strip()
        if not source_id:
            continue

        snippet = str(getattr(item, "snippet", "") or "").replace("\n", " ").strip()
        title = str(getattr(item, "title", "") or "").strip()
        fallback_text = snippet or title or str(getattr(item, "url_or_path", "") or "").strip()
        if not fallback_text:
            continue

        grounded_claims.append(
            ClaimItem(
                text=fallback_text,
                evidence_ids=[source_id],
                confidence=getattr(item, "score", None),
            )
        )

    if not grounded_claims:
        return build_empty_response_payload(
            answer="응답 생성이 시간 제한을 초과했습니다. 확인된 근거만으로 요약을 완성하지 못했습니다."
        )

    confidence = average_claim_confidence(grounded_claims)
    payload = render_payload_from_claims(
        claims=grounded_claims,
        evidence_items=evidence_items,
        confidence=confidence,
    )
    payload.confidence = confidence
    return payload


def _build_local_fallback_payload(
    *,
    evidence_items: list[Any],
    retrieval_required: bool,
    generic_answer: str,
) -> AgentResponsePayloadModel:
    if evidence_items:
        return _build_grounded_timeout_payload(evidence_items)
    if retrieval_required:
        return build_empty_response_payload(answer="")
    return build_empty_response_payload(answer=generic_answer)


def _render_synthesis_payload(
    synthesis_output: SynthesisOutput,
    evidence_items: list[Any],
) -> tuple[AgentResponsePayloadModel, str]:
    payload_confidence = synthesis_output.confidence
    if payload_confidence is None:
        payload_confidence = average_claim_confidence(synthesis_output.claims)

    if synthesis_output.claims:
        payload = render_payload_from_claims(
            claims=synthesis_output.claims,
            evidence_items=evidence_items,
            confidence=payload_confidence,
        )
        return payload, payload.answer

    fallback_answer = str(synthesis_output.answer or "").strip()
    payload = build_empty_response_payload(
        answer=fallback_answer,
        confidence=payload_confidence,
    )
    return payload, payload.answer


def _build_synthesis_messages(
    *,
    state: State,
    action_rules: list[str],
    deduped_evidence: list[dict[str, Any]],
    attempt: int,
    max_turns: int,
) -> tuple[list[BaseMessage], int, int]:
    history_messages = [
        message for message in state.get("messages", []) if not isinstance(message, ToolMessage)
    ]
    history_before = len(history_messages)
    trimmed_history = keep_recent_messages(history_messages, max_turns=max_turns)

    model_messages: list[BaseMessage] = [SystemMessage(content=SYS_POLICY)]
    if state.get("memory_summary"):
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{state['memory_summary']}"))
    model_messages.extend(trimmed_history)
    if action_rules:
        model_messages.append(SystemMessage(content="[Action Request]\n- " + "\n- ".join(action_rules)))
    model_messages.append(
        SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(deduped_evidence)}")
    )
    model_messages.append(SystemMessage(content=SYNTHESIS_CONTRACT))
    if attempt > 1:
        model_messages.append(
            SystemMessage(
                content=(
                    "Retry synthesis after evidence validation failed. "
                    "Use retrieved evidence when available and avoid unsupported claims."
                )
            )
        )

    return model_messages, history_before, len(trimmed_history)


def make_synthesize_node(
    llm_synthesizer: Any,
    verbose: bool,
    max_turns: int = 6,
    has_default_slack_destination: bool = False,
):
    structured_synthesizer = _build_structured_synthesizer(llm_synthesizer)

    def synthesize(state: State):
        stage_started = time.perf_counter()
        attempt = int(state.get("synthesis_attempt", 0)) + 1
        user_input = str(state.get("user_input", "") or "")
        guided_followup = str(state.get("guided_followup") or "").strip()
        explicit_slack_destinations = get_slack_destinations(state.get("session_metadata"))
        slack_target_available = any(explicit_slack_destinations.values()) or has_default_slack_destination

        if guided_followup:
            payload = build_empty_response_payload(answer=guided_followup)
            response = AIMessage(content=guided_followup)
            return {
                "messages": [response],
                "final_answer": guided_followup,
                "response_payload": _payload_to_state_dict(payload),
                "synthesis_output": SynthesisOutput(answer=guided_followup).model_dump(mode="json"),
                "synthesis_attempt": attempt,
                "needs_retry": False,
                "latency_trace": [
                    make_stage_latency_event(
                        stage="synthesis",
                        attempt=attempt,
                        latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                        status="guided_followup",
                    )
                ],
            }

        if is_action_only_request(user_input):
            final_answer = build_action_only_answer(
                user_input=user_input,
                messages=state.get("messages", []),
                slack_target_available=slack_target_available,
            )
            payload = build_empty_response_payload(answer=final_answer)
            response = AIMessage(content=final_answer)
            return {
                "messages": [response],
                "final_answer": final_answer,
                "response_payload": _payload_to_state_dict(payload),
                "synthesis_output": SynthesisOutput(answer=final_answer).model_dump(mode="json"),
                "synthesis_attempt": attempt,
                "needs_retry": False,
                "latency_trace": [
                    make_stage_latency_event(
                        stage="synthesis",
                        attempt=attempt,
                        latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                        status="action_only",
                    )
                ],
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

        retry_context = coerce_retry_context(state.get("retry_context"))
        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
        current_attempt_evidence_payload = slice_from_index(
            safe_list(state.get("retrieved_evidence")),
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
        deduped_evidence = evidence_to_dicts(parsed_evidence)

        planner_parse_errors: list[str] = []
        planner_output = coerce_planner_output(state.get("planner_output"), planner_parse_errors)
        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)

        model_messages, history_before, history_after = _build_synthesis_messages(
            state=state,
            action_rules=action_rules,
            deduped_evidence=deduped_evidence,
            attempt=attempt,
            max_turns=max_turns,
        )
        if verbose and history_before != history_after:
            log_event(
                logger,
                logging.INFO,
                "synthesize_trimmed_messages",
                before=history_before,
                after=history_after,
            )

        synthesis_errors: list[str] = []
        llm_calls: list[LLMCallMetadata] = []
        structured_ms: int | None = None
        fallback_ms: int | None = None
        synthesis_mode = "structured_only"

        try:
            structured_started = time.perf_counter()
            structured_result = structured_synthesizer.invoke(model_messages)
            structured_ms = elapsed_ms(structured_started, time.perf_counter())
            raw_response_obj, raw_message, structured_error = _coerce_structured_synthesis_result(
                structured_result
            )
            if raw_message is not None:
                llm_calls.append(
                    build_llm_call_metadata(
                        stage="synthesis",
                        attempt=attempt,
                        path="structured",
                        message=raw_message,
                    )
                )
            if structured_error is not None:
                raise structured_error
            synthesis_output = _coerce_synthesis_output(raw_response_obj)
            payload, final_answer = _render_synthesis_payload(synthesis_output, parsed_evidence)
        except Exception as exc:
            structured_ms = elapsed_ms(structured_started, time.perf_counter())
            if _is_timeout_error(exc):
                synthesis_mode = "timeout_grounded_fallback"
                synthesis_errors.append(f"synthesize: structured output timed out ({exc})")
                fallback_started = time.perf_counter()
                payload = _build_local_fallback_payload(
                    evidence_items=parsed_evidence,
                    retrieval_required=retrieval_required,
                    generic_answer="응답 생성이 시간 제한을 초과했습니다. 질문 범위를 조금 좁혀 다시 시도해 주세요.",
                )
                fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
                synthesis_output = SynthesisOutput(
                    answer=payload.answer,
                    claims=payload.claims,
                    confidence=payload.confidence,
                )
                final_answer = payload.answer
            else:
                synthesis_mode = "structured_error_plain_fallback"
                synthesis_errors.append(f"synthesize: structured output failed ({exc})")
                try:
                    fallback_started = time.perf_counter()
                    raw_response_obj = llm_synthesizer.invoke(model_messages)
                    fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
                    if isinstance(raw_response_obj, AIMessage):
                        llm_calls.append(
                            build_llm_call_metadata(
                                stage="synthesis",
                                attempt=attempt,
                                path="plain_fallback",
                                message=raw_response_obj,
                            )
                        )
                    synthesis_output = _coerce_synthesis_output(raw_response_obj)
                    payload, final_answer = _render_synthesis_payload(synthesis_output, parsed_evidence)
                except Exception as fallback_exc:
                    fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
                    synthesis_errors.append(f"synthesize: plain fallback failed ({fallback_exc})")
                    payload = _build_local_fallback_payload(
                        evidence_items=parsed_evidence,
                        retrieval_required=retrieval_required,
                        generic_answer="응답 생성 중 오류가 발생했습니다. 질문 범위를 조금 좁혀 다시 시도해 주세요.",
                    )
                    synthesis_output = SynthesisOutput(
                        answer=payload.answer,
                        claims=payload.claims,
                        confidence=payload.confidence,
                    )
                    final_answer = payload.answer

        response = AIMessage(content=final_answer)
        total_ms = elapsed_ms(stage_started, time.perf_counter())
        latency_trace = [
            make_synthesis_attempt_latency_event(
                attempt=attempt,
                mode=synthesis_mode,
                structured_ms=structured_ms,
                fallback_ms=fallback_ms,
                total_ms=total_ms,
            ),
            make_stage_latency_event(
                stage="synthesis",
                attempt=attempt,
                latency_ms=total_ms,
                status=synthesis_mode,
            ),
        ]

        updates: State = {
            "messages": [response],
            "final_answer": final_answer,
            "response_payload": _payload_to_state_dict(payload),
            "synthesis_output": synthesis_output.model_dump(mode="json"),
            "synthesis_attempt": attempt,
            "needs_retry": False,
            "latency_trace": latency_trace,
        }

        combined_retrieval_errors = parse_errors + planner_parse_errors
        if combined_retrieval_errors:
            updates["retrieval_errors"] = combined_retrieval_errors
        if synthesis_errors:
            updates["synthesis_errors"] = synthesis_errors
        if llm_calls:
            updates["llm_calls"] = llm_calls
        return updates

    return synthesize
