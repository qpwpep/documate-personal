"""Bounded predicate decisions with strict answers and provider usage records."""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from src.core.contracts.usage import normalize_token_usage
from src.core.decisions import DecisionResult
from src.core.llm_errors import LLMCallError, LLMDiagnostic, make_problem
from src.infra.llm_boundary import classify_call_error, record_llm_diagnostic, request_call_budget
from src.infra.logging_utils import log_event
from src.runtime.agent_runtime.llm_usage import record_llm_call


logger = logging.getLogger(__name__)
_VALIDATION_STAGES = {"pre_synthesis_validation", "post_synthesis_validation"}


@dataclass(frozen=True)
class _UsageObservation:
    """The usage recorder accepts message-shaped metadata without model text."""
    response_metadata: dict[str, Any]
    usage_metadata: dict[str, Any]


def _safe_identifier(value: Any) -> str | None:
    if value is None:
        return None
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_./:-]{1,200}", value) else "<redacted>"


def _parse_json(content: bytes) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate_field")
            result[key] = value
        return result

    def reject_constant(_value):
        raise ValueError("nonfinite_number")

    return json.loads(content, object_pairs_hook=unique_object, parse_constant=reject_constant)


def _parse_answers(payload: Any, *, names: set[str]) -> DecisionResult:
    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        raise ValueError("answers_required")
    answers = payload["answers"]
    if len(answers) != len(names):
        raise ValueError("answer_count_mismatch")
    probabilities: dict[str, float] = {}
    refused_names: set[str] = set()
    seen_names: set[str] = set()
    for answer in answers:
        if not isinstance(answer, dict):
            raise ValueError("answer_object_required")
        name = answer.get("name")
        if not isinstance(name, str) or name not in names or name in seen_names:
            raise ValueError("answer_name_mismatch")
        seen_names.add(name)
        if answer.get("type") == "refusal":
            refused_names.add(name)
            continue
        probability = answer.get("probability")
        if (answer.get("type") != "predicate" or type(probability) not in {int, float}
                or not math.isfinite(probability) or not 0 <= probability <= 1):
            raise ValueError("predicate_probability_invalid")
        probabilities[name] = float(probability)
    return DecisionResult(probabilities=probabilities, refused_names=refused_names)


def _usage_observation(payload: Any, *, model: str, request_id: str | None) -> _UsageObservation:
    # Only token counters are retained. Neither input, answers nor arbitrary
    # provider fields enter diagnostics or the public usage records.
    raw_usage = payload.get("usage") if isinstance(payload, dict) else None
    counters = {key: value for key in ("input_tokens", "output_tokens", "total_tokens")
                if isinstance(raw_usage, dict) and type(value := raw_usage.get(key)) is int and value >= 0}
    return _UsageObservation(
        response_metadata={"model": model, "endpoint": "decisions", "request_id": request_id},
        usage_metadata=counters,
    )


class DecisionsClient:
    """One Decisions API boundary shared by both validation stages.

    Batches and transient retries consume the request's existing stage budget.
    Malformed responses fail the whole call. Valid refusals are retained by
    question name so each validation stage can apply its recovery policy.
    """

    max_questions_per_call = 200

    def __init__(self, client: OpenAI, *, model: str = "gpt-6-luna", timeout_seconds: float = 20) -> None:
        if model != "gpt-6-luna":
            raise ValueError("Decisions supports only gpt-6-luna")
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("Decisions timeout must be positive and finite")
        self._client = client.with_options(max_retries=0)
        self.model_name = model
        self.timeout_seconds = timeout_seconds

    def evaluate(self, *, input: str, questions: list[dict[str, str]], stage: str,
                 attempt: int = 1, reserve_calls: int = 0) -> DecisionResult:
        """Leave reserved calls available for subsequent validation checks."""
        if type(reserve_calls) is not int or reserve_calls < 0:
            raise ValueError("Decisions reserve_calls must be a nonnegative integer")
        if stage not in _VALIDATION_STAGES:
            raise ValueError("Decisions requires a validation stage")
        if not isinstance(input, str) or not input.strip():
            raise ValueError("Decisions input must be nonempty text")
        if not questions:
            raise ValueError("Decisions requires at least one question")
        names: set[str] = set()
        for question in questions:
            if (not isinstance(question, dict) or question.get("type") != "predicate"
                    or not isinstance(question.get("name"), str) or not question["name"].strip()
                    or not isinstance(question.get("instructions"), str) or not question["instructions"].strip()
                    or question["name"] in names):
                raise ValueError("Decisions questions require unique names and predicate instructions")
            names.add(question["name"])

        budget = request_call_budget(stage)
        probabilities: dict[str, float] = {}
        refused_names: set[str] = set()
        for start in range(0, len(questions), self.max_questions_per_call):
            batch = questions[start:start + self.max_questions_per_call]
            batch_names = {question["name"] for question in batch}
            remaining_batches = math.ceil((len(questions) - start - len(batch)) / self.max_questions_per_call)
            while True:
                if not budget.can_attempt or budget.remaining_calls <= reserve_calls:
                    diagnostic = LLMDiagnostic(code="call_budget_exhausted", stage=stage, model=self.model_name,
                        endpoint="decisions", attempt=budget.calls,
                        budget_reason="calls" if budget.remaining_calls <= reserve_calls else "deadline")
                    record_llm_diagnostic(diagnostic)
                    raise LLMCallError(make_problem("call_budget_exhausted", stage), diagnostic)
                budget.calls += 1
                current_attempt = max(attempt, budget.calls)
                request_id = None
                try:
                    with record_llm_call(stage=stage, attempt=current_attempt, path="decisions") as call:
                        # Keep the native SDK request and inspect raw bytes before
                        # its permissive parser could hide duplicate JSON keys.
                        response = self._client.decisions.with_raw_response.create(
                            model=self.model_name, input=input, questions=batch,
                            timeout=min(self.timeout_seconds, budget.remaining_seconds))
                        request_id = _safe_identifier(response.headers.get("x-request-id"))
                        try:
                            payload = _parse_json(response.content)
                            observation = _usage_observation(payload, model=self.model_name, request_id=request_id)
                            call.complete(observation)
                            batch_result = _parse_answers(payload, names=batch_names)
                        except ValueError as exc:
                            raise LLMCallError(make_problem("model_output_invalid", stage), LLMDiagnostic(
                                code="model_output_invalid", stage=stage, model=self.model_name,
                                endpoint="decisions", attempt=current_attempt, exception_type=type(exc).__name__,
                                validation_paths=["/answers:contract_validation"],
                            )) from exc
                    usage = normalize_token_usage(observation.usage_metadata, observation.response_metadata)
                    log_event(logger, logging.INFO, "decisions_completed", stage=stage, endpoint="decisions",
                              model=self.model_name, question_count=len(batch), attempt=current_attempt,
                              probability_count=len(batch_result.probabilities),
                              refusal_count=len(batch_result.refused_names),
                              request_id=request_id, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                    probabilities.update(batch_result.probabilities)
                    refused_names.update(batch_result.refused_names)
                    break
                except Exception as exc:
                    error = classify_call_error(exc, stage=stage, llm=self, attempt=current_attempt)
                    delay = error.problem.retry_after_seconds
                    if delay is None:
                        delay = min(2 ** (budget.calls - 1) * 0.25, 2.0)
                    # Retrying is useful only if every later batch can still run
                    # without spending the calls reserved for later checks.
                    can_retry = (error.problem.code in {"provider_unavailable", "provider_rate_limited"}
                                 and budget.can_attempt
                                 and budget.remaining_calls > reserve_calls + remaining_batches
                                 and delay < budget.remaining_seconds)
                    diagnostic = error.diagnostic.model_copy(update={
                        "model": self.model_name, "endpoint": "decisions", "attempt": current_attempt,
                        "provider_request_id": error.diagnostic.provider_request_id or request_id,
                        "recovery": "retry" if can_retry else "stop",
                        "exception_type": error.diagnostic.exception_type or type(exc).__name__,
                    })
                    record_llm_diagnostic(diagnostic)
                    if can_retry:
                        time.sleep(delay)
                        continue
                    raise LLMCallError(error.problem, diagnostic) from exc
        return DecisionResult(probabilities=probabilities, refused_names=refused_names)
