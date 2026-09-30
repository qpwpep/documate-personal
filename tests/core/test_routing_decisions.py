"""Observe routing through compiled graphs, including partial node updates."""

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from src.core.contracts import DebugState, GraphState, PlannerState, ResponseState, RetryState
from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.contracts.boundary.response import get_response_state
from src.core.conversation_memory import ConversationMemoryPolicy
from src.core.planner_schema import PlannerOutput, RetrievalTask
from src.runtime.make_graph import build_graph
from src.runtime.nodes.session import add_user_message

from .test_graph_routing import _run_graph


def _graph(**nodes):
    def synthesize(state):
        response = get_response_state(state)
        return {"response": response.model_copy(update={"synthesis_attempt": response.synthesis_attempt + 1})}

    return build_graph(
        state_type=GraphState,
        memory_policy=ConversationMemoryPolicy(),
        **{
            "add_user_node": add_user_message,
            "summarize_node": lambda _state: {},
            "planner_node": lambda _state: {},
            "retrieve_dispatch_node": lambda _state: {},
            "pre_synthesis_validation_node": lambda _state: {},
            "synthesize_node": synthesize,
            "post_synthesis_validation_node": lambda _state: {"retry": RetryState()},
            "action_postprocess_node": lambda _state: {},
            **nodes,
        },
    )


@pytest.mark.parametrize("guided", [False, True])
def test_planner_partial_update_preserves_the_incoming_routing_inputs(guided):
    planner = PlannerState(
        output=PlannerOutput(use_retrieval=True, tasks=[RetrievalTask(route="docs", query="docs", k=1)]),
        guided_followup="Which library?" if guided else None,
    )
    # An injected graph node is allowed to return only a diagnostic patch.
    graph = _graph(planner_node=lambda _state: {"debug": {"planner_errors": []}})
    result, visited = _run_graph(graph, build_graph_state_input(user_input="question", planner=planner))

    target = "pre_synthesis_validation" if guided else "retrieve_dispatch"
    assert result["route_decisions"][1].target == target
    assert visited[2] == target


def test_explicit_empty_planner_replaces_old_guided_followup_and_tasks():
    initial = build_graph_state_input(user_input="question", planner={"guided_followup": "old question"})
    graph = _graph(planner_node=lambda _state: {"planner": PlannerState()})

    result, visited = _run_graph(graph, initial)

    assert result["route_decisions"][1].target == "synthesize"
    assert "pre_synthesis_validation" not in visited


def test_post_validation_partial_update_reuses_retry_but_explicit_clear_stops_the_loop():
    def validate(state):
        if state["response"].synthesis_attempt == 1:
            return {"debug": {"validation_events": ["repair requested"]}}
        return {"retry": RetryState()}

    result, visited = _run_graph(_graph(post_synthesis_validation_node=validate), build_graph_state_input(
        user_input="question",
        retry={"needs_retry": True, "retry_scope": "reuse_hits_resynthesize", "retry_reason": "missing_content"},
    ))

    decisions = [decision for decision in result["route_decisions"] if decision.source == "post_synthesis_validation"]
    assert [decision.target for decision in decisions] == ["synthesize", "action_postprocess"]
    assert visited.count("synthesize") == 2
    assert visited.count("planner") == 1
    assert "retrieve_dispatch" not in visited


def test_debug_replacement_cannot_erase_previously_committed_decisions():
    graph = _graph(action_postprocess_node=lambda _state: {"debug": DebugState()})

    result, _visited = _run_graph(graph, build_graph_state_input(user_input="question"))

    assert len(result["route_decisions"]) == 3
    assert result["debug"] == DebugState()


def test_message_replacement_uses_add_messages_instead_of_counting_a_duplicate_turn():
    messages = []
    for index in range(6):
        messages.extend([HumanMessage(id=f"user-{index}", content=f"question-{index}"), AIMessage(content="answer")])
    initial = build_graph_state_input(user_input="replaced question", current_turn_id="user-5", messages=messages)

    result, visited = _run_graph(_graph(), initial)

    assert result["route_decisions"][0].target == "planner"
    assert "summarize_old_messages" not in visited
    assert len(result["messages"]) == len(messages)
    assert result["messages"][-2].content == "replaced question"


@pytest.mark.parametrize("node_name", ["planner_node", "action_postprocess_node"])
def test_business_nodes_cannot_reappend_the_route_history(node_name):
    graph = _graph(**{node_name: lambda state: {"route_decisions": state["route_decisions"]}})

    with pytest.raises(ValueError, match="Only the graph routing adapter"):
        graph.invoke(build_graph_state_input(user_input="question"))


def test_failing_source_does_not_commit_a_decision_and_prior_decisions_remain():
    def fail(_state):
        raise RuntimeError("planner failed")

    graph = _graph(planner_node=fail)
    last_state = None
    with pytest.raises(RuntimeError, match="planner failed"):
        for last_state in graph.stream(build_graph_state_input(user_input="question"), stream_mode="values"):
            pass

    assert [(decision.source, decision.target) for decision in last_state["route_decisions"]] == [
        ("add_user_message", "planner"),
    ]


def test_graph_visualization_retains_the_possible_command_destinations():
    edges = {(edge.source, edge.target) for edge in _graph().get_graph().edges}

    assert ("post_synthesis_validation", "synthesize") in edges
    assert ("post_synthesis_validation", "planner") in edges
    assert ("post_synthesis_validation", "action_postprocess") in edges
    assert ("add_user_message", "summarize_old_messages") in edges
