from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from src.core.contracts import GraphState, PlannerState
from src.core.contracts.boundary.debug import get_debug_state
from src.core.contracts.boundary.graph import get_retry_state
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.retrieval import get_retrieval_state
from src.core.contracts.boundary.runtime import get_runtime_state
from src.core.contracts.debug import (
    PlannerDiagnostic, RetryState,
    empty_planner_diagnostic,
)
from src.core.planner_schema import InitialPlannerOutput, PlannerOutput, RetrievalRewriteOutput, RetrievalTask
from src.core.request_contracts import RequestContract, validate_contract_evidence
from src.core.slack_contract import SlackDelivery
from src.infra.logging_utils import log_event
from src.infra.llm_boundary import run_structured_call
from src.runtime.nodes.planner.guardrails import apply_retrieval_availability
from src.runtime.nodes.planner.models import PlannerDecision
from src.runtime.nodes.planner.prompt_builder import build_planner_messages, planner_previous_answer, planner_user_utterances
from src.runtime.nodes.planner.query_sanitizer import ground_planner_requirement_aspects
from src.runtime.nodes.planner.request_resolution import (
    current_recipient_evidence_ids, pending_body_changed, pending_recipient_changed, resolve_request_contract,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PlannerRunContext:
    has_retriever: bool
    planner_attempt: int
    upload_file_ids: tuple[str, ...] = ()


def _resolve_planner_strategy(
    *, llm_planner: Any, llm_planner_retry: Any | None,
    state: GraphState, context: PlannerRunContext, max_turns: int,
) -> tuple[PlannerDecision, RequestContract]:
    runtime = get_runtime_state(state)
    is_replan = runtime.request_contract is not None
    original_tasks: list[RetrievalTask] = []
    failed_ids: set[str] = set()
    current_tasks: dict[str, RetrievalTask] = {}
    if is_replan:
        if llm_planner_retry is None:
            raise RuntimeError("retrieval retry model is not configured")
        retry = get_retry_state(state)
        original_tasks = retry.original_tasks
        failed_ids = set(retry.failed_requirement_ids)
        if not failed_ids or not failed_ids.issubset(task.requirement_id for task in original_tasks):
            raise ValueError("retrieval retry must identify failed original requirements")
        current_tasks = {task.requirement_id: task for task in get_planner_state(state).output.tasks}
        if any(task.requirement_id not in current_tasks for task in original_tasks if task.requirement_id not in failed_ids):
            raise ValueError("retrieval retry must retain the current successful tasks")

    removed_aspects = False

    def validate(payload: Any) -> tuple[PlannerOutput, RequestContract]:
        nonlocal removed_aspects
        if is_replan:
            rewrites = RetrievalRewriteOutput.model_validate(payload)
            revised = {rewrite.requirement_id: rewrite for rewrite in rewrites.rewrites}
            if set(revised) != failed_ids:
                raise ValueError("rewrites must contain exactly the failed requirement IDs: " + ", ".join(sorted(failed_ids)))
            contract = runtime.request_contract
            output = PlannerOutput(use_retrieval=True, tasks=[
                task.model_copy(update={"query": revised[task.requirement_id].query,
                                        "k": revised[task.requirement_id].k})
                if task.requirement_id in revised else task.model_copy(update={
                    "query": current_tasks[task.requirement_id].query,
                    "k": current_tasks[task.requirement_id].k,
                })
                for task in original_tasks
            ], request_contract=contract.to_wire())
        else:
            initial = InitialPlannerOutput.model_validate(payload)
            utterances = planner_user_utterances(state)
            errors = validate_contract_evidence(initial.request_contract, utterances)
            if errors:
                raise ValueError("; ".join(errors))
            grounded = ground_planner_requirement_aspects(
                initial, user_utterances=utterances,
                current_turn_id=runtime.current_turn_id,
                previous_answer=planner_previous_answer(state),
            )
            removed_aspects = any(left.requirement.aspects != right.requirement.aspects
                                  for left, right in zip(initial.tasks, grounded.tasks, strict=True))
            initial = grounded
            # Binding is part of model-output validation. A failed call never reaches it.
            contract = resolve_request_contract(initial.request_contract, state, utterances=utterances)
            output = PlannerOutput(use_retrieval=initial.use_retrieval, tasks=initial.tasks,
                                   request_contract=contract.to_wire())
        return output, contract

    output, contract = run_structured_call(
        llm_planner_retry if is_replan else llm_planner,
        build_planner_messages(state, max_turns=max_turns),
        stage="planner", validate=validate, attempt=context.planner_attempt,
    )
    return PlannerDecision(
        output=output, status="llm", guided_followup=None,
        diagnostics=PlannerDiagnostic(status="llm", planner_warnings=["unrequested_constraints_removed"] if removed_aspects else []),
    ), contract


def _apply_planner_guardrail(
    *,
    decision: PlannerDecision,
    context: PlannerRunContext,
) -> PlannerDecision:
    planner_output = decision.output
    unknown_files = {file_id for task in planner_output.tasks for file_id in task.requirement.file_ids
                     if file_id not in context.upload_file_ids}
    if unknown_files:
        return replace(
            decision, output=PlannerOutput.fallback(request_contract=planner_output.request_contract),
            diagnostics=decision.diagnostics.model_copy(update={
                "reason": "upload_file_scope_invalid", "required_routes": ["upload"],
                "planner_warnings": list(dict.fromkeys([*decision.diagnostics.planner_warnings, "unknown_upload_file_ids"])),
            }),
            guided_followup="요청한 첨부 파일을 현재 세션에서 확인할 수 없습니다. 첨부 목록에서 사용할 파일을 다시 지정해 주세요.",
        )
    return apply_retrieval_availability(
        replace(decision, output=planner_output),
        has_retriever=context.has_retriever,
    )


def _reset_retry_window(
    *,
    existing_retry_context: RetryState,
    retrieval_evidence_count: int,
    retrieval_error_count: int,
    retrieval_diagnostic_count: int,
) -> RetryState:
    retry_context = existing_retry_context.model_copy(
        update={
            "needs_retry": False,
            "hit_start_index": retrieval_evidence_count,
            "retrieval_error_start_index": retrieval_error_count,
            "retrieval_diagnostic_start_index": retrieval_diagnostic_count,
        }
    )
    if retry_context.attempt <= 0:
        retry_context = retry_context.model_copy(
            update={
                "retrieval_feedback": "",
                "score_avg": None,
                "retry_reason": None,
                "failed_routes": [],
                "failed_requirement_ids": [],
                "preserved_hits": [],
                "preserved_retrieval_diagnostics": [],
            }
        )
    return retry_context


def make_planner_node(
    llm_planner: Any,
    verbose: bool,
    max_turns: int = 6,
    *, llm_planner_retry: Any | None = None,
):
    def planner(state: GraphState) -> GraphState:
        runtime = get_runtime_state(state)
        retrieval = get_retrieval_state(state)
        debug = get_debug_state(state)
        existing_retry_context = get_retry_state(state)
        context = PlannerRunContext(
            has_retriever=bool(runtime.retriever),
            planner_attempt=existing_retry_context.attempt + 1,
            upload_file_ids=tuple(item.file_id for item in runtime.upload_files),
        )

        decision, contract = _resolve_planner_strategy(
            llm_planner=llm_planner, llm_planner_retry=llm_planner_retry,
            state=state, context=context, max_turns=max_turns,
        )
        is_initial_plan = runtime.request_contract is None
        if is_initial_plan:
            if contract.can_acknowledge() or contract.can_cancel_pending():
                decision = replace(decision, output=PlannerOutput.fallback(request_contract=contract.to_wire()))
            elif not contract.can_prepare_body():
                question = contract.clarification_question or "요청의 작업과 대상을 더 구체적으로 알려 주세요."
                decision = replace(
                    decision, output=PlannerOutput.fallback(request_contract=contract.to_wire()),
                    guided_followup=question,
                    diagnostics=decision.diagnostics.model_copy(update={"reason": "clarification_required"}),
                )
            decision = _apply_planner_guardrail(
                decision=decision,
                context=context,
            )
        else:
            # Retry changes only query/k. The bound contract and execution resources
            # were checked when this request's original requirements were accepted.
            required_routes = list(dict.fromkeys(task.route for task in decision.output.tasks))
            decision = replace(decision, diagnostics=decision.diagnostics.model_copy(update={
                "intent_required": bool(required_routes), "required_routes": required_routes,
            }))

        if verbose:
            log_event(
                logger,
                logging.INFO,
                "planner",
                status=decision.status,
                use_retrieval=decision.output.use_retrieval,
                task_count=len(decision.output.tasks),
                required_routes=decision.diagnostics.required_routes,
                override=decision.diagnostics.override_applied,
            )

        retry_context = _reset_retry_window(
            existing_retry_context=existing_retry_context,
            retrieval_evidence_count=len(retrieval.hit_log),
            retrieval_error_count=len(debug.retrieval_errors),
            retrieval_diagnostic_count=len(debug.retrieval_diagnostics),
        )
        if not retry_context.original_tasks and decision.output.use_retrieval:
            retry_context = retry_context.model_copy(update={
                "original_tasks": list(decision.output.tasks),
            })

        runtime_updates = {"request_contract": contract}
        if (is_initial_plan and runtime.pending_action is not None and contract.relation in {"correction", "supplement"}
                and contract.target_request_id == runtime.pending_action.contract.request_id and contract.failure is None):
            completed = tuple(name for name in runtime.pending_action.completed_actions
                              if getattr(contract.actions, name).intent != "requested")
            pending_updates = {"contract": contract, "completed_actions": completed}
            body_changed = pending_body_changed(runtime.pending_action, contract)
            if pending_recipient_changed(runtime.pending_action, contract):
                pending_updates["slack_delivery"] = None
            elif (runtime.pending_action.slack_delivery is not None
                  and runtime.pending_action.slack_delivery.selection is None
                  and runtime.pending_action.slack_delivery.intent.state == "unresolved"
                  and contract.slack_recipient.state == "explicit"
                  and current_recipient_evidence_ids(contract, current_turn_id=runtime.current_turn_id,
                                                     current_utterance=runtime.user_input)):
                # A fresh confirmation may resolve a prior text/input conflict;
                # an existing selected identity never needs resolution again.
                pending_updates["slack_delivery"] = None
            elif ("slack_notify" in runtime.pending_action.completed_actions
                  and contract.actions.slack_notify.intent == "requested"
                  and runtime.pending_action.slack_delivery is not None
                  and runtime.pending_action.slack_delivery.status == "sent"):
                # Current, grounded send authorization creates a new obligation;
                # retain the selected identity instead of reusing its old success.
                pending_updates["slack_delivery"] = SlackDelivery.model_validate({
                    **runtime.pending_action.slack_delivery.model_dump(),
                    "status": "pending", "failure": None, "message_ts": None,
                })
            if body_changed or ("save_text" in runtime.pending_action.completed_actions
                                and contract.actions.save_text.intent == "requested"):
                # Replace the obligation when the change is accepted, including
                # corrections whose body will only be ready on a later turn.
                pending_updates.update(save_operation=None, save_receipt=None)
            if body_changed or contract.body.kind != "copy_answer":
                pending_updates["body_prepared"] = False
                pending_updates["phase"] = "awaiting_body" if contract.can_prepare_body() else "awaiting_input"
            runtime_updates["pending_action"] = runtime.pending_action.model_copy(update=pending_updates)
        updates: GraphState = {
            "runtime": runtime.model_copy(update=runtime_updates),
            "planner": PlannerState(
                output=decision.output,
                status=decision.status,
                diagnostics=decision.diagnostics or empty_planner_diagnostic(status=decision.status),
                guided_followup=decision.guided_followup,
            ),
            "retry": retry_context,
        }
        return updates

    return planner
