"""Connect validation-owned inputs to the execution's diagnostic recorder."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import Literal

from src.core.answer_schema.models import document_hash
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.graph_state import GraphState
from src.core.contracts.validation_diagnostics import ValidationObservation
from src.runtime.agent_runtime.validation_diagnostics import (
    current_validation_observation, record_candidates, record_packet, record_validation,
    validation_diagnostics_enabled,
)
from src.runtime.nodes.validation.models import ValidationAssessment, ValidationSnapshot


def observe_validation_node(
    *, stage: Literal["pre_synthesis_validation", "post_synthesis_validation"], model: str, threshold: float,
) -> Callable[[Callable[[GraphState], GraphState]], Callable[[GraphState], GraphState]]:
    def decorate(node: Callable[[GraphState], GraphState]) -> Callable[[GraphState], GraphState]:
        @wraps(node)
        def observed(state: GraphState) -> GraphState:
            if not validation_diagnostics_enabled():
                return node(state)
            runtime, response, retry = get_runtime_state(state), get_response_state(state), get_retry_state(state)
            post = stage == "post_synthesis_validation"
            body = response.result.content.model_copy(deep=True) if post else None
            observation = ValidationObservation(
                sequence=1, stage=stage, attempt=max(1, response.synthesis_attempt) if post else retry.attempt + 1,
                revision="generated" if post else "preparation", model=model, threshold=threshold,
                user_input=runtime.user_input,
                request_contract=runtime.request_contract.model_copy(deep=True) if runtime.request_contract else None,
                tasks=[task.model_copy(deep=True) for task in get_planner_state(state).output.tasks],
                body=body, body_hash=document_hash(body) if body is not None else None,
                terminal_failed_requirement_ids=list(retry.terminal_failed_requirement_ids),
            )
            with record_validation(observation):
                record_candidates(get_retrieval_state(state).hit_log[retry.hit_start_index:])
                if post:
                    record_packet("used", response.evidence_packet, response.evidence_requirement_map)
                return node(state)
        return observed
    return decorate


@contextmanager
def observe_filtered_validation(snapshot: ValidationSnapshot, *, model: str, threshold: float,
                                attempt: int) -> Iterator[None]:
    if not validation_diagnostics_enabled():
        yield
        return
    body = snapshot.response_result.content.model_copy(deep=True) if snapshot.response_result else None
    observation = ValidationObservation(
        sequence=1, stage="post_synthesis_validation", attempt=attempt, revision="filtered", model=model,
        threshold=threshold, user_input=snapshot.user_input,
        request_contract=snapshot.request_contract.model_copy(deep=True) if snapshot.request_contract else None,
        tasks=[task.model_copy(deep=True) for task in snapshot.planner_output.tasks],
        body=body, body_hash=document_hash(body) if body is not None else None,
        terminal_failed_requirement_ids=list(snapshot.terminal_failed_requirement_ids),
    )
    with record_validation(observation):
        record_packet("used", snapshot.evidence_packet, snapshot.evidence_requirement_map)
        yield


def record_assessment(assessment: ValidationAssessment, *, local: bool = True) -> None:
    observation = current_validation_observation()
    if observation is None:
        return
    if local:
        observation.local_failures = [
            *assessment.missing_content,
            *(f"forbidden:{name}" for name in assessment.forbidden_content),
        ]
        observation.local_failed_unit_ids = sorted(assessment.invalid_unit_paths)
        if assessment.checked_result is not None:
            observation.local_failures.extend(
                f"{check.unit_id}:{issue}" for check in assessment.checked_result.checks
                for issue in check.issues if check.unit_id in assessment.invalid_unit_paths
            )
    observation.failed_requirement_ids = sorted(assessment.failed_requirement_ids)
