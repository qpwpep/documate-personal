from __future__ import annotations

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, StateGraph, add_messages
from langgraph.types import Command

from src.core.conversation_memory import ConversationMemoryPolicy, plan_compaction
from src.core.contracts.graph_state import GraphState
from src.core.contracts.routing import RoutingDecision, RoutingSource, RoutingTarget
from src.core.contracts.boundary.graph import get_retry_state, normalize_graph_update
from src.core.contracts.boundary.planner import get_planner_state
from src.core.contracts.boundary.response import get_response_state
from src.core.contracts.boundary.runtime import get_runtime_state


def _summary_route(state: GraphState, policy: ConversationMemoryPolicy) -> tuple[RoutingTarget, str]:
    plan = plan_compaction(
        state.get("messages", []), get_runtime_state(state).memory_summary, policy,
    )
    if plan.should_compact:
        return "summarize_old_messages", f"memory_high_watermark:{','.join(plan.trigger_reasons)}"
    return "planner", "memory_below_high_watermarks"


def _planner_route(state: GraphState) -> tuple[RoutingTarget, str]:
    planner = get_planner_state(state)
    if str(planner.guided_followup or "").strip():
        return "pre_synthesis_validation", "guided_followup_present"
    if planner.output.use_retrieval and planner.output.tasks:
        return "retrieve_dispatch", f"retrieval_required:{len(planner.output.tasks)}_task(s)"
    return "synthesize", "retrieval_not_required"


def _pre_synthesis_route(state: GraphState) -> tuple[RoutingTarget, str]:
    retry = get_retry_state(state)
    if retry.needs_retry:
        return "planner", str(retry.retry_reason or "retry_requested")
    if get_response_state(state).result.content.blocks:
        return "action_postprocess", "terminal_response_available"
    return "synthesize", "validation_passed"


def _post_synthesis_route(state: GraphState) -> tuple[RoutingTarget, str]:
    retry = get_retry_state(state)
    if retry.needs_retry:
        target = "synthesize" if retry.retry_scope == "reuse_hits_resynthesize" else "planner"
        return target, str(retry.retry_reason or "retry_requested")
    return "action_postprocess", str(retry.retry_reason or "validation_passed")


def _business_node(node: Callable[[GraphState], GraphState]):
    def wrapped(state: GraphState) -> GraphState:
        updates = node(state)
        if not isinstance(updates, dict):
            raise TypeError("Graph business nodes must return a state update dictionary")
        if "route_decisions" in updates:
            raise ValueError("Only the graph routing adapter can write route_decisions")
        return normalize_graph_update(updates)

    return wrapped


def _route_after(
    source: RoutingSource,
    node: Callable[[GraphState], GraphState],
    decide: Callable[[GraphState], tuple[RoutingTarget, str]],
):
    run_node = _business_node(node)

    def wrapped(state: GraphState) -> Command:
        updates = run_node(state)
        # These routing inputs are replacement channels, not nested patches.
        route_state: GraphState = {
            key: updates[key] if key in updates else state[key]
            for key in ("runtime", "planner", "retry", "response")
            if key in updates or key in state
        }
        if source == "add_user_message":
            # add_user_message returns a message delta. Use the same public
            # reducer as GraphState rather than losing the existing history.
            route_state["messages"] = add_messages(
                state.get("messages", []), updates.get("messages", []),
            )
        target, reason = decide(route_state)
        decision = RoutingDecision(
            sequence=len(state.get("route_decisions", [])) + 1,
            source=source,
            target=target,
            reason=reason,
        )
        # One result owns both the next node and the committed observation.
        # The append channel receives only this occurrence, never the history.
        return Command(update={**updates, "route_decisions": [decision]}, goto=decision.target)

    return wrapped


def build_graph(
    state_type: Any,
    add_user_node: Any,
    summarize_node: Any,
    planner_node: Any,
    retrieve_dispatch_node: Any,
    synthesize_node: Any,
    *,
    memory_policy: ConversationMemoryPolicy,
    pre_synthesis_validation_node: Any,
    post_synthesis_validation_node: Any,
    action_postprocess_node: Any,
):
    builder = StateGraph(state_type)

    # destinations describes the diagram. These four nodes route exclusively
    # through Command; adding outgoing edges would schedule extra work.
    builder.add_node(
        "add_user_message",
        _route_after("add_user_message", add_user_node, lambda state: _summary_route(state, memory_policy)),
        destinations=("summarize_old_messages", "planner"),
    )
    builder.set_entry_point("add_user_message")
    builder.add_node("summarize_old_messages", _business_node(summarize_node))
    builder.add_node(
        "planner", _route_after("planner", planner_node, _planner_route),
        destinations=("pre_synthesis_validation", "retrieve_dispatch", "synthesize"),
    )
    builder.add_node("retrieve_dispatch", _business_node(retrieve_dispatch_node))
    builder.add_node(
        "pre_synthesis_validation",
        _route_after("pre_synthesis_validation", pre_synthesis_validation_node, _pre_synthesis_route),
        destinations=("planner", "synthesize", "action_postprocess"),
    )
    builder.add_node("synthesize", _business_node(synthesize_node))
    builder.add_node(
        "post_synthesis_validation",
        _route_after("post_synthesis_validation", post_synthesis_validation_node, _post_synthesis_route),
        destinations=("planner", "synthesize", "action_postprocess"),
    )
    builder.add_node("action_postprocess", _business_node(action_postprocess_node))

    builder.add_edge("summarize_old_messages", "planner")
    builder.add_edge("retrieve_dispatch", "pre_synthesis_validation")
    builder.add_edge("synthesize", "post_synthesis_validation")
    builder.add_edge("action_postprocess", END)
    return builder.compile()
