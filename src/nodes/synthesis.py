from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ..answer_schema import (
    AgentResponsePayloadModel,
    ClaimItem,
    SynthesisOutput,
    average_claim_confidence,
    build_deterministic_grounded_payload,
    build_empty_response_payload,
    normalize_confidence,
    render_payload_from_claims,
)
from ..contracts.debug import LLMCallMetadata, build_llm_call_metadata
from ..contracts.graph_state import (
    GraphState,
    ResponseState,
    coerce_planner_output,
    debug_state,
    normalize_state_updates,
    planner_state,
    response_state,
    retrieval_state,
    retry_state,
    runtime_state,
    slice_from_index,
)
from ..contracts.routes import route_for_tool
from ..evidence import EvidenceItem, dedupe_evidence, evidence_to_dicts, parse_evidence_payload
from ..latency import (
    elapsed_ms,
    make_stage_latency_event,
    make_synthesis_attempt_latency_event,
)
from ..logging_utils import log_event
from ..planner_schema import PlannerOutput
from ..prompts import SYS_POLICY, needs_save, needs_slack
from .actions import build_action_only_answer, get_slack_destinations, is_action_only_request
from .retrieval import format_evidence_for_prompt
from .session import extract_text_content, keep_recent_messages


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
PLAIN_SUMMARY_ATTACH_CONTRACT = (
    "[Plain Summary Attach Fallback]\n"
    "- Use only Retrieved Evidence.\n"
    "- Return plain text only.\n"
    "- Return at most 2 lines or 2 sentences.\n"
    "- Each line or sentence must describe one grounded takeaway.\n"
    "- Keep the same order as Retrieved Evidence.\n"
    "- Do not include citations, bullets, numbering, JSON, or markdown."
)
_LEADING_BULLET_PATTERN = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


def _normalize_query_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split()).strip(" ,.;:-")


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
    if isinstance(result, AIMessage):
        return result, result, None
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


def _build_synthesis_messages(
    *,
    state: GraphState,
    action_rules: list[str],
    deduped_evidence: list[dict[str, Any]],
    attempt: int,
    max_turns: int,
) -> tuple[list[BaseMessage], int, int]:
    runtime = runtime_state(state)
    history_messages = [
        message for message in state.get("messages", []) if not isinstance(message, ToolMessage)
    ]
    history_before = len(history_messages)
    trimmed_history = keep_recent_messages(history_messages, max_turns=max_turns)

    model_messages: list[BaseMessage] = [SystemMessage(content=SYS_POLICY)]
    if runtime.memory_summary:
        model_messages.append(SystemMessage(content=f"[Conversation Summary]\n{runtime.memory_summary}"))
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


def _build_plain_summary_attach_messages(
    *,
    user_input: str,
    deduped_evidence: list[dict[str, Any]],
) -> list[BaseMessage]:
    compact_evidence = deduped_evidence[:2]
    return [
        SystemMessage(content=SYS_POLICY),
        SystemMessage(content=PLAIN_SUMMARY_ATTACH_CONTRACT),
        HumanMessage(content=_normalize_query_text(user_input) or "Summarize the retrieved evidence."),
        SystemMessage(content=f"[Retrieved Evidence]\n{format_evidence_for_prompt(compact_evidence)}"),
    ]


def _parse_plain_summary_segments(content: str, *, limit: int) -> list[str]:
    raw_lines = str(content or "").replace("\r", "\n").splitlines()
    line_segments: list[str] = []
    for line in raw_lines:
        stripped = _LEADING_BULLET_PATTERN.sub("", line).strip()
        if not stripped:
            continue
        line_segments.append(stripped)
        if len(line_segments) >= limit:
            break
    if len(line_segments) >= limit or len(line_segments) > 1:
        return line_segments[:limit]

    normalized = " ".join(str(content or "").replace("\r", "\n").split()).strip()
    if not normalized:
        return []

    sentence_segments = [
        segment.strip()
        for segment in _SENTENCE_SPLIT_PATTERN.split(normalized)
        if segment.strip()
    ]
    if len(sentence_segments) >= 2:
        return sentence_segments[:limit]
    if line_segments:
        return line_segments[:1]
    return sentence_segments[:limit] if sentence_segments else [normalized]


def _route_for_evidence(item: EvidenceItem) -> str:
    return route_for_tool(str(item.tool or ""))


def _select_primary_evidence_items(
    *,
    evidence_items: list[EvidenceItem],
    planner_output: PlannerOutput,
) -> list[EvidenceItem]:
    if not evidence_items:
        return []

    if planner_output.use_retrieval and planner_output.tasks:
        selected: list[EvidenceItem] = []
        seen_routes: set[str] = set()
        for task in planner_output.tasks:
            route = str(task.route or "")
            if route in seen_routes:
                continue
            match = next((item for item in evidence_items if _route_for_evidence(item) == route), None)
            if match is not None:
                selected.append(match)
                seen_routes.add(route)
        if selected:
            return selected[:2]

    return evidence_items[:2]


def _select_grounded_fallback_evidence_items(
    *,
    evidence_items: list[EvidenceItem],
    planner_output: PlannerOutput,
) -> list[EvidenceItem]:
    return _select_primary_evidence_items(
        evidence_items=evidence_items,
        planner_output=planner_output,
    ) or evidence_items[:2]


def _build_plain_summary_attach_payload(
    *,
    content: str,
    evidence_items: list[EvidenceItem],
) -> AgentResponsePayloadModel | None:
    if not evidence_items:
        return None

    limited_evidence = evidence_items[:2]
    segments = _parse_plain_summary_segments(content, limit=len(limited_evidence))
    if not segments:
        return None

    adopted_pairs = list(zip(segments[: len(limited_evidence)], limited_evidence))
    if not adopted_pairs:
        return None

    claims: list[ClaimItem] = []
    for segment, evidence_item in adopted_pairs:
        source_id = str(evidence_item.source_id or "").strip()
        if not source_id:
            continue
        claims.append(
            ClaimItem(
                text=segment,
                evidence_ids=[source_id],
                confidence=normalize_confidence(evidence_item.score, clamp=True),
            )
        )

    if not claims:
        return None

    confidence = average_claim_confidence(claims)
    payload = render_payload_from_claims(
        claims=claims,
        evidence_items=limited_evidence,
        confidence=confidence,
    )
    payload.confidence = confidence
    return payload


def _build_local_fallback_payload(
    *,
    evidence_items: list[EvidenceItem],
    retrieval_required: bool,
    generic_answer: str,
) -> AgentResponsePayloadModel:
    if evidence_items:
        return build_deterministic_grounded_payload(
            evidence_items=evidence_items,
            fallback_answer=generic_answer,
        )
    if retrieval_required:
        return build_empty_response_payload(answer="")
    return build_empty_response_payload(answer=generic_answer)


def _should_use_deterministic_grounded_direct(
    *,
    planner_output: PlannerOutput,
    evidence_items: list[EvidenceItem],
) -> bool:
    if not planner_output.use_retrieval or not planner_output.tasks:
        return False
    if not 1 <= len(evidence_items) <= 2:
        return False
    selected_routes = {task.route for task in planner_output.tasks}
    return selected_routes in ({"upload"}, {"docs", "upload"})


def _render_synthesis_payload(
    synthesis_output: SynthesisOutput,
    evidence_items: list[EvidenceItem],
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


def _build_response_state(
    *,
    payload: AgentResponsePayloadModel,
    synthesis_output: SynthesisOutput,
    final_answer: str,
    attempt: int,
) -> ResponseState:
    return ResponseState(
        final_answer=final_answer,
        payload=payload,
        synthesis_output=synthesis_output,
        synthesis_attempt=attempt,
    )


def _build_synthesis_updates(
    *,
    debug,
    payload: AgentResponsePayloadModel,
    synthesis_output: SynthesisOutput,
    final_answer: str,
    attempt: int,
    latency_trace: list[dict[str, Any]],
    retrieval_errors: list[str] | None = None,
    planner_errors: list[str] | None = None,
    synthesis_errors: list[str] | None = None,
    llm_calls: list[LLMCallMetadata] | None = None,
) -> GraphState:
    return normalize_state_updates({
        "messages": [AIMessage(content=final_answer)],
        "response": _build_response_state(
            payload=payload,
            synthesis_output=synthesis_output,
            final_answer=final_answer,
            attempt=attempt,
        ),
        "debug": debug.model_copy(
            update={
                "retrieval_errors": [*debug.retrieval_errors, *(retrieval_errors or [])],
                "planner_errors": [*debug.planner_errors, *(planner_errors or [])],
                "synthesis_errors": [*debug.synthesis_errors, *(synthesis_errors or [])],
                "llm_calls": [*debug.llm_calls, *(llm_calls or [])],
                "latency_trace": [*debug.latency_trace, *latency_trace],
            }
        ),
    })


def make_synthesize_node(
    llm_synthesizer: Any,
    llm_synthesizer_compact: Any | None = None,
    verbose: bool = False,
    max_turns: int = 6,
    has_default_slack_destination: bool = False,
):
    structured_synthesizer = _build_structured_synthesizer(llm_synthesizer)
    fallback_llm = llm_synthesizer_compact or llm_synthesizer

    def synthesize(state: GraphState) -> GraphState:
        stage_started = time.perf_counter()
        runtime = runtime_state(state)
        planner = planner_state(state)
        retrieval = retrieval_state(state)
        response = response_state(state)
        debug = debug_state(state)
        attempt = int(response.synthesis_attempt) + 1
        user_input = runtime.user_input
        guided_followup = str(planner.guided_followup or "").strip()
        explicit_slack_destinations = get_slack_destinations(runtime.session_metadata)
        slack_target_available = any(explicit_slack_destinations.values()) or has_default_slack_destination

        if guided_followup:
            payload = build_empty_response_payload(answer=guided_followup)
            synthesis_output = SynthesisOutput(answer=guided_followup)
            return _build_synthesis_updates(
                debug=debug,
                payload=payload,
                synthesis_output=synthesis_output,
                final_answer=guided_followup,
                attempt=attempt,
                latency_trace=[
                    make_stage_latency_event(
                        stage="synthesis",
                        attempt=attempt,
                        latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                        status="guided_followup",
                    )
                ],
            )

        if is_action_only_request(user_input):
            final_answer = build_action_only_answer(
                user_input=user_input,
                messages=state.get("messages", []),
                slack_target_available=slack_target_available,
            )
            payload = build_empty_response_payload(answer=final_answer)
            synthesis_output = SynthesisOutput(answer=final_answer)
            return _build_synthesis_updates(
                debug=debug,
                payload=payload,
                synthesis_output=synthesis_output,
                final_answer=final_answer,
                attempt=attempt,
                latency_trace=[
                    make_stage_latency_event(
                        stage="synthesis",
                        attempt=attempt,
                        latency_ms=elapsed_ms(stage_started, time.perf_counter()),
                        status="action_only",
                    )
                ],
            )

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

        retry_context = retry_state(state)
        evidence_start_index = int(retry_context.get("evidence_start_index", 0))
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
        planner_output = coerce_planner_output(planner.output, planner_parse_errors)
        retrieval_required = bool(planner_output.use_retrieval and planner_output.tasks)

        primary_evidence_items = _select_primary_evidence_items(
            evidence_items=evidence_items,
            planner_output=planner_output,
        )
        grounded_fallback_evidence_items = _select_grounded_fallback_evidence_items(
            evidence_items=evidence_items,
            planner_output=planner_output,
        )
        deduped_evidence = evidence_to_dicts(primary_evidence_items)

        if _should_use_deterministic_grounded_direct(
            planner_output=planner_output,
            evidence_items=primary_evidence_items,
        ):
            payload = build_deterministic_grounded_payload(
                evidence_items=primary_evidence_items,
                fallback_answer="",
            )
            final_answer = payload.answer
            total_ms = elapsed_ms(stage_started, time.perf_counter())
            synthesis_output = SynthesisOutput(
                answer=payload.answer,
                claims=payload.claims,
                confidence=payload.confidence,
            )
            return _build_synthesis_updates(
                debug=debug,
                payload=payload,
                synthesis_output=synthesis_output,
                final_answer=final_answer,
                attempt=attempt,
                latency_trace=[
                    make_synthesis_attempt_latency_event(
                        attempt=attempt,
                        mode="deterministic_grounded_direct",
                        structured_ms=0,
                        fallback_ms=None,
                        total_ms=total_ms,
                    ),
                    make_stage_latency_event(
                        stage="synthesis",
                        attempt=attempt,
                        latency_ms=total_ms,
                        status="deterministic_grounded_direct",
                    ),
                ],
            )

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
            payload, final_answer = _render_synthesis_payload(
                synthesis_output,
                primary_evidence_items,
            )
        except Exception as exc:
            structured_ms = elapsed_ms(structured_started, time.perf_counter())
            synthesis_errors.append(
                (
                    f"synthesize: structured output timed out ({exc})"
                    if "timeout" in str(exc).lower() or "timed out" in str(exc).lower()
                    else f"synthesize: structured output failed ({exc})"
                )
            )
            try:
                fallback_started = time.perf_counter()
                fallback_result = fallback_llm.invoke(
                    _build_plain_summary_attach_messages(
                        user_input=user_input,
                        deduped_evidence=deduped_evidence,
                    )
                )
                fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
                if isinstance(fallback_result, AIMessage):
                    llm_calls.append(
                        build_llm_call_metadata(
                            stage="synthesis",
                            attempt=attempt,
                            path="plain_summary_attach_fallback",
                            message=fallback_result,
                        )
                    )
                fallback_content = extract_text_content(getattr(fallback_result, "content", fallback_result))
                fallback_payload = _build_plain_summary_attach_payload(
                    content=fallback_content,
                    evidence_items=primary_evidence_items,
                )
                if fallback_payload is None:
                    raise RuntimeError("plain summary attach payload could not be built")
                payload = fallback_payload
                synthesis_output = SynthesisOutput(
                    answer=payload.answer,
                    claims=payload.claims,
                    confidence=payload.confidence,
                )
                final_answer = payload.answer
                synthesis_mode = "plain_summary_attach_fallback"
            except Exception as fallback_exc:
                if fallback_ms is None:
                    fallback_ms = elapsed_ms(fallback_started, time.perf_counter())
                synthesis_errors.append(f"synthesize: plain summary attach failed ({fallback_exc})")
                payload = _build_local_fallback_payload(
                    evidence_items=grounded_fallback_evidence_items,
                    retrieval_required=retrieval_required,
                    generic_answer="응답 생성 중 오류가 발생했습니다. 질문 범위를 조금 좁혀 다시 시도해 주세요.",
                )
                synthesis_output = SynthesisOutput(
                    answer=payload.answer,
                    claims=payload.claims,
                    confidence=payload.confidence,
                )
                final_answer = payload.answer
                synthesis_mode = "deterministic_grounded_fallback"

        total_ms = elapsed_ms(stage_started, time.perf_counter())
        latency_trace = [
            make_synthesis_attempt_latency_event(
                attempt=attempt,
                mode=synthesis_mode,  # type: ignore[arg-type]
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

        return _build_synthesis_updates(
            debug=debug,
            payload=payload,
            synthesis_output=synthesis_output,
            final_answer=final_answer,
            attempt=attempt,
            latency_trace=latency_trace,
            retrieval_errors=parse_errors,
            planner_errors=planner_parse_errors,
            synthesis_errors=synthesis_errors,
            llm_calls=llm_calls,
        )

    return synthesize
