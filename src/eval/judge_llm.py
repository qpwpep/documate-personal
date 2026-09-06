from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.core.answer_schema import AnswerResponse
from src.core.evidence import SearchHit
from .config_models import BenchmarkCase
from .result_models import JudgeSubscores

_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for an AI agent benchmark.
Evaluate the assistant response using the provided case metadata, structured response payload, observed evidence, and runtime diagnostics.
Return ONLY JSON with this schema:
{
  "score": <float 0..1>,
  "reason": "<short reason>",
  "subscores": {
    "answer_quality": <float 0..1>,
    "groundedness": <float 0..1>,
    "citation_traceability": <float 0..1>,
    "tool_choice": <float 0..1>,
    "format_language": <float 0..1>
  }
}

Scoring guidance:
- answer_quality: whether the response actually answers the user's request with useful substance rather than copying snippets.
- groundedness: whether the exact displayed content units are supported by their referenced source snapshots. Resolved references are not semantic proof; not_evaluated means no support assessment has run.
- citation_traceability: whether displayed refs resolve to the same versioned source and canonical element in observed hits, with cited text ranges or table cells contained in the observed selection. Budgeted subranges have different reference IDs but remain traceable. Logical element positions are valid even when physical page coordinates are unavailable.
- tool_choice: whether the executed tools and retrieval routes match case expectations.
- format_language: whether the response follows the requested structure and restates in the user's language.

Failure guidance:
- Penalize heavily if the response mainly lists links or pasted snippets instead of synthesizing.
- Penalize if the response does not restate in the user's language.
- For docs-focused cases, prioritize official documentation summaries over generic web-style summaries.
- For hybrid cases, assess whether the displayed content actually compares the official source and uploaded code; no particular block title or layout is required.
- Evaluate response.content directly. It is the exact document rendered to the user and exported for delivery.
- For tool_action cases, do not expect citations or retrieval grounding when the case itself does not require them.
- For tool_action cases, expect usable content and a separate action receipt; do not require a receipt appended to the body.
- For live Slack delivery cases, a slack_notify action with status success is completion; skipped/error or a missing action is incomplete delivery.
- For Korean queries, a non-Korean answer should score 0 on format_language.
- Use validator_reason, retrieval_diagnostics, planner_diagnostics, and synthesis_mode as evidence when scoring.
- If the supplied evaluation input is incomplete or inconsistent, reflect that in the reason, but still score the visible response quality.
"""


def _extract_text_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return str(content)


def _parse_json_payload(text: str) -> dict[str, Any] | None:
    stripped = text.strip()
    if not stripped:
        return None

    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", stripped)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _normalize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_jsonable(item) for item in value]
    if hasattr(value, "model_dump"):
        return _normalize_jsonable(value.model_dump(mode="json"))
    return str(value)


def _is_payload_complete(payload: dict[str, Any]) -> bool:
    response = payload.get("response")
    if not isinstance(response, dict):
        return False
    required_top_level = (
        "case",
        "response",
        "observed_hits",
        "retrieval_diagnostics",
        "planner_diagnostics",
        "validator_reason",
        "synthesis_mode",
    )
    if any(key not in payload for key in required_top_level):
        return False
    return all(key in response for key in ("content", "citations", "checks", "actions", "content_hash"))


class LLMJudge:
    def __init__(self, model_name: str, enabled: bool = True):
        self.enabled = enabled
        self.model_name = model_name
        self.client = None

        if not enabled:
            return

        self.client = ChatOpenAI(
            model=model_name,
            temperature=0,
            timeout=60,
            max_retries=2,
        )

    def score_case(
        self,
        *,
        case: BenchmarkCase,
        tool_calls: list[str],
        response: AnswerResponse | None,
        observed_hits: list[SearchHit] | None = None,
        retrieval_diagnostics: list[dict[str, Any]] | None = None,
        planner_diagnostics: dict[str, Any] | None = None,
        validator_reason: str | None = None,
        synthesis_mode: str | None = None,
        resolved_unit_count: int = 0,
        missing_reference_unit_count: int = 0,
        unchecked_unit_count: int = 0,
        tool_call_count: int | None = None,
        slack_delivery_required: bool = False,
    ) -> tuple[float | None, str | None, str | None, JudgeSubscores | None]:
        if not self.enabled:
            return None, None, None, None
        if self.client is None:
            return None, None, "invalid_eval: judge client is not initialized", None

        user_prompt = self.build_case_payload(
            case=case,
            tool_calls=tool_calls,
            response=response,
            observed_hits=observed_hits,
            retrieval_diagnostics=retrieval_diagnostics,
            planner_diagnostics=planner_diagnostics,
            validator_reason=validator_reason,
            synthesis_mode=synthesis_mode,
            resolved_unit_count=resolved_unit_count,
            missing_reference_unit_count=missing_reference_unit_count,
            unchecked_unit_count=unchecked_unit_count,
            tool_call_count=tool_call_count,
            slack_delivery_required=slack_delivery_required,
        )
        if not self.is_payload_complete(user_prompt):
            return None, None, "invalid_eval: judge payload is incomplete", None

        try:
            result = self.client.invoke(
                [
                    SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(user_prompt, ensure_ascii=False)),
                ]
            )
        except Exception as exc:
            return None, None, f"invalid_eval: judge invocation failed ({exc})", None

        parsed = _parse_json_payload(_extract_text_content(result.content))
        if not parsed:
            return None, None, "invalid_eval: judge returned non-JSON content", None

        subscores_raw = parsed.get("subscores")
        try:
            subscores = JudgeSubscores.model_validate(subscores_raw)
        except Exception as exc:
            return None, None, f"invalid_eval: judge subscores are missing or invalid ({exc})", None

        try:
            score = float(parsed.get("score"))
        except (TypeError, ValueError):
            score = subscores.average()

        reason = parsed.get("reason")
        reason_text = str(reason) if reason is not None else None
        bounded_score = max(0.0, min(1.0, score))
        return bounded_score, reason_text, None, subscores

    @staticmethod
    def build_case_payload(
        *,
        case: BenchmarkCase,
        tool_calls: list[str],
        response: AnswerResponse | None,
        observed_hits: list[SearchHit] | None = None,
        retrieval_diagnostics: list[dict[str, Any]] | list[Any] | None = None,
        planner_diagnostics: dict[str, Any] | Any | None = None,
        validator_reason: str | None = None,
        synthesis_mode: str | None = None,
        resolved_unit_count: int = 0,
        missing_reference_unit_count: int = 0,
        unchecked_unit_count: int = 0,
        tool_call_count: int | None = None,
        slack_delivery_required: bool = False,
    ) -> dict[str, Any]:
        return {
            "case": {
                "case_id": case.case_id,
                "category": case.category,
                "query": case.query,
                "expected_tools": case.expected_tools,
                "forbidden_tools": case.forbidden_tools,
                "judge_rubric": case.judge_rubric,
                "judge_min_score": case.judge_min_score,
            },
            "response": _normalize_jsonable(response),
            "observed_hits": _normalize_jsonable(observed_hits or []),
            "called_tools": list(tool_calls),
            "tool_call_count": int(tool_call_count or len(tool_calls)),
            "retrieval_diagnostics": _normalize_jsonable(retrieval_diagnostics or []),
            "planner_diagnostics": _normalize_jsonable(planner_diagnostics),
            "validator_reason": validator_reason,
            "synthesis_mode": synthesis_mode,
            "slack_delivery_required": bool(slack_delivery_required),
            "content_stats": {
                "resolved_unit_count": resolved_unit_count,
                "missing_reference_unit_count": missing_reference_unit_count,
                "unchecked_unit_count": unchecked_unit_count,
            },
        }

    @staticmethod
    def is_payload_complete(payload: dict[str, Any]) -> bool:
        return _is_payload_complete(payload)
