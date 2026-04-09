from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from ..answer_schema import AnswerSection, ClaimItem
from ..evidence import EvidenceItem
from .schemas import BenchmarkCase, JudgeSubscores

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
- groundedness: whether the claims stay supported by the supplied evidence and diagnostics.
- citation_traceability: whether claims can be traced to response evidence and observed evidence.
- tool_choice: whether the executed tools and retrieval routes match case expectations.
- format_language: whether the response follows the requested structure and restates in the user's language.

Failure guidance:
- Penalize heavily if the response mainly lists links or pasted snippets instead of synthesizing.
- Penalize if the response does not restate in the user's language.
- For docs-focused cases, prioritize official documentation summaries over generic web-style summaries.
- For hybrid cases, expect the official explanation and the uploaded/local comparison to be clearly separated.
- For hybrid cases, treat a missing comparison section as a significant quality failure.
- Use response.sections as the primary structure signal when it is present.
- For tool_action cases, do not expect citations or retrieval grounding when the case itself does not require them.
- For tool_action cases, prefer responses that contain a usable body first and a clear execution receipt such as a saved path or Slack destination after it.
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
        "observed_evidence",
        "retrieval_diagnostics",
        "planner_diagnostics",
        "validator_reason",
        "synthesis_mode",
    )
    if any(key not in payload for key in required_top_level):
        return False
    return all(key in response for key in ("text", "claims", "evidence", "sections"))


def _serialize_claims(claims: list[ClaimItem] | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in claims or []:
        if isinstance(item, ClaimItem):
            serialized.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            serialized.append(_normalize_jsonable(item))
    return serialized


def _serialize_evidence(
    evidence: list[EvidenceItem] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in evidence or []:
        if isinstance(item, EvidenceItem):
            serialized.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            serialized.append(_normalize_jsonable(item))
    return serialized


def _serialize_sections(
    sections: list[AnswerSection] | list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for item in sections or []:
        if isinstance(item, AnswerSection):
            serialized.append(item.model_dump(mode="json"))
        elif isinstance(item, dict):
            serialized.append(_normalize_jsonable(item))
    return serialized


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

    def score_case_detailed(
        self,
        *,
        case: BenchmarkCase,
        response_text: str,
        tool_calls: list[str],
        claims: list[dict[str, Any]] | None = None,
        response_evidence: list[dict[str, Any]] | None = None,
        sections: list[dict[str, Any]] | None = None,
        observed_evidence: list[dict[str, Any]] | None = None,
        retrieval_diagnostics: list[dict[str, Any]] | None = None,
        planner_diagnostics: dict[str, Any] | None = None,
        validator_reason: str | None = None,
        synthesis_mode: str | None = None,
        valid_claim_count: int | None = None,
        invalid_claim_count: int | None = None,
        tool_call_count: int | None = None,
    ) -> tuple[float | None, str | None, str | None, JudgeSubscores | None]:
        if not self.enabled:
            return None, None, None, None
        if self.client is None:
            return None, None, "invalid_eval: judge client is not initialized", None

        user_prompt = self.build_case_payload(
            case=case,
            response_text=response_text,
            tool_calls=tool_calls,
            claims=claims,
            response_evidence=response_evidence,
            sections=sections,
            observed_evidence=observed_evidence,
            retrieval_diagnostics=retrieval_diagnostics,
            planner_diagnostics=planner_diagnostics,
            validator_reason=validator_reason,
            synthesis_mode=synthesis_mode,
            valid_claim_count=valid_claim_count,
            invalid_claim_count=invalid_claim_count,
            tool_call_count=tool_call_count,
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
        response_text: str,
        tool_calls: list[str],
        claims: list[ClaimItem] | list[dict[str, Any]] | None = None,
        response_evidence: list[EvidenceItem] | list[dict[str, Any]] | None = None,
        sections: list[AnswerSection] | list[dict[str, Any]] | None = None,
        observed_evidence: list[EvidenceItem] | list[dict[str, Any]] | None = None,
        retrieval_diagnostics: list[dict[str, Any]] | list[Any] | None = None,
        planner_diagnostics: dict[str, Any] | Any | None = None,
        validator_reason: str | None = None,
        synthesis_mode: str | None = None,
        valid_claim_count: int | None = None,
        invalid_claim_count: int | None = None,
        tool_call_count: int | None = None,
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
            "response": {
                "text": response_text,
                "claims": _serialize_claims(claims),
                "evidence": _serialize_evidence(response_evidence),
                "sections": _serialize_sections(sections),
            },
            "observed_evidence": _serialize_evidence(observed_evidence),
            "called_tools": list(tool_calls),
            "tool_call_count": int(tool_call_count or len(tool_calls)),
            "retrieval_diagnostics": _normalize_jsonable(retrieval_diagnostics or []),
            "planner_diagnostics": _normalize_jsonable(planner_diagnostics),
            "validator_reason": validator_reason,
            "synthesis_mode": synthesis_mode,
            "claim_stats": {
                "valid_claim_count": int(valid_claim_count or 0),
                "invalid_claim_count": int(invalid_claim_count or 0),
            },
        }

    @staticmethod
    def is_payload_complete(payload: dict[str, Any]) -> bool:
        return _is_payload_complete(payload)

    def score_case(
        self,
        case: BenchmarkCase,
        response_text: str,
        tool_calls: list[str],
        **kwargs: Any,
    ) -> tuple[float | None, str | None, str | None, JudgeSubscores | None]:
        return self.score_case_detailed(
            case=case,
            response_text=response_text,
            tool_calls=tool_calls,
            **kwargs,
        )
