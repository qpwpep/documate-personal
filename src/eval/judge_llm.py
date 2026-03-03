from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .schemas import BenchmarkCase


_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for an AI agent benchmark.
Evaluate the assistant response using the provided case metadata.
Return ONLY JSON with this schema:
{"score": <float 0..1>, "reason": "<short reason>"}

Scoring guidance:
- 1.0: fully satisfies intent, tool behavior expectation, and rubric
- 0.7~0.9: mostly good with minor omission
- 0.4~0.6: partially correct
- 0.1~0.3: major issues
- 0.0: irrelevant/failed
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
        case: BenchmarkCase,
        response_text: str,
        tool_calls: list[str],
    ) -> tuple[float | None, str | None, str | None]:
        if not self.enabled:
            return None, None, "LLM judge disabled"
        if self.client is None:
            return None, None, "LLM judge client is not initialized"

        user_prompt = {
            "case_id": case.case_id,
            "category": case.category,
            "query": case.query,
            "expected_tools": case.expected_tools,
            "forbidden_tools": case.forbidden_tools,
            "judge_rubric": case.judge_rubric,
            "response_text": response_text,
            "called_tools": tool_calls,
        }

        try:
            result = self.client.invoke(
                [
                    SystemMessage(content=_JUDGE_SYSTEM_PROMPT),
                    HumanMessage(content=json.dumps(user_prompt, ensure_ascii=False)),
                ]
            )
        except Exception as exc:
            return None, None, f"LLM judge invocation failed: {exc}"

        parsed = _parse_json_payload(_extract_text_content(result.content))
        if not parsed:
            return None, None, "LLM judge returned non-JSON content"

        try:
            score = float(parsed.get("score"))
        except (TypeError, ValueError):
            return None, None, "LLM judge score is missing or invalid"

        reason = parsed.get("reason")
        reason_text = str(reason) if reason is not None else None
        bounded_score = max(0.0, min(1.0, score))
        return bounded_score, reason_text, None
