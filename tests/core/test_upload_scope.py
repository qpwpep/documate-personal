import pytest
from langchain_core.messages import HumanMessage

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import InitialPlannerOutput, RetrievalRequirement
from src.core.request_contracts import UserTurnSnapshot
from src.runtime.nodes.planner import make_planner_node
from .helpers import _CapturePlannerLLM, structured_model


class _RepairingPlanner:
    def __init__(self, *plans):
        self.plans = iter(plans)

    def invoke(self, messages, *, timeout=None):
        return next(self.plans)


def _upload_plan(requirement, *, query):
    from src.core.request_contracts import WireRequestContract

    planned_requirement = RetrievalRequirement.model_validate(requirement)
    return {"use_retrieval": True, "tasks": [{"route": "upload", "requirement_id": "source",
            "requirement": planned_requirement.model_dump(mode="json"), "query": query, "k": 4}],
            "task_interpretations": [{"requirement_id": "source", "scope": "target",
                "aspect_roles": [{"aspect": aspect, "role": "source_literal"}
                                 for aspect in planned_requirement.aspects]}],
            "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
            "aspect_references": []}


def test_active_upload_filename_is_repaired_to_file_scope_before_search():
    """A known filename cannot become a required code symbol just because it contains a dot."""
    from types import SimpleNamespace

    query = "첨부한 billing_rules.py의 계산 방식을 설명해줘."
    state = build_graph_state_input(
        user_input=query, current_turn_id="current",
        user_turns=(UserTurnSnapshot(turn_id="current", text=query),),
        messages=[HumanMessage(id="current", content=query)],
        retriever=SimpleNamespace(), upload_files=[{
            "file_id": "billing", "name": "billing_rules.py", "size_bytes": 10,
            "content_hash": "sha256:" + "a" * 64, "source_uri": "upload://session/billing",
        }],
    )
    planner = _RepairingPlanner(
        _upload_plan({"symbols": ["billing_rules.py"], "match": "symbol", "file_ids": ["billing"]}, query=query),
        _upload_plan({"file_ids": ["billing"]}, query=query),
    )
    result = make_planner_node(structured_model(planner.invoke, InitialPlannerOutput), False)(state)
    task, = result["planner"].output.tasks
    assert task.requirement.symbols == []
    assert task.requirement.file_ids == ["billing"]
    assert task.requirement.match == "topic"
    assert task.query == query


@pytest.mark.parametrize("condition", ["기본 할인율", "the default timeout"])
def test_semantic_upload_condition_is_repaired_without_weakening_the_question(condition):
    """A natural-language property stays in the question, not a code identifier obligation."""
    from types import SimpleNamespace

    query = f"현재 첨부의 calculate_total 함수에서 {condition}에 대해 알려줘. 출처를 인용해줘."
    state = build_graph_state_input(
        user_input=query, current_turn_id="current",
        user_turns=(UserTurnSnapshot(turn_id="current", text=query),),
        messages=[HumanMessage(id="current", content=query)], retriever=SimpleNamespace(),
    )
    planner = _RepairingPlanner(
        _upload_plan({"symbols": ["calculate_total"], "match": "definition", "aspects": [condition]}, query=query),
        _upload_plan({"symbols": ["calculate_total"], "match": "definition"}, query=query),
    )
    result = make_planner_node(structured_model(planner.invoke, InitialPlannerOutput), False)(state)
    task, = result["planner"].output.tasks
    assert task.requirement.aspects == []
    assert task.requirement.symbols == ["calculate_total"]
    assert task.query == query


@pytest.mark.parametrize(("fragment", "match"), [
    ("is not None", "definition"),
    ("match value", "symbol"),
    ("기본할인율", "definition"),
    ("discount=0.10", "symbol"),
    ("value\nother", "definition"),
    ("value\rother", "symbol"),
])
def test_planner_preserves_python_fragments_as_literal_requirements(fragment, match):
    """A literal source fragment need not be a standalone Python statement."""
    from types import SimpleNamespace

    query = f"첨부한 calculate_total의 `{fragment}` 코드 조각을 찾아 인용해줘."
    state = build_graph_state_input(
        user_input=query, current_turn_id="current",
        user_turns=(UserTurnSnapshot(turn_id="current", text=query),),
        messages=[HumanMessage(id="current", content=query)], retriever=SimpleNamespace(),
    )
    planner = _CapturePlannerLLM(_upload_plan({
        "symbols": ["calculate_total"], "match": match, "aspects": [fragment],
    }, query=query))

    result = make_planner_node(planner, False)(state)

    task, = result["planner"].output.tasks
    assert task.requirement_id == "source"
    assert task.requirement.symbols == ["calculate_total"]
    assert task.requirement.match == match
    assert task.requirement.aspects == [fragment]
    assert task.query == query
