import pytest

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput, RetrievalRewriteOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.planner import make_planner_node
from .helpers import _CapturePlannerLLM


class _RepairingPlanner:
    def __init__(self, *plans):
        self.plans = iter(plans)

    def invoke(self, messages, *, timeout=None):
        return next(self.plans)


def _upload_plan(requirement, *, query):
    from src.core.request_contracts import WireRequestContract

    return {"use_retrieval": True, "tasks": [{"route": "upload", "requirement_id": "source",
            "requirement": requirement, "query": query, "k": 4}],
            "request_contract": WireRequestContract(slack_recipient={"state": "omitted"}).model_dump(mode="json"),
            "aspect_references": []}


def test_active_upload_filename_is_repaired_to_file_scope_before_search():
    """A known filename cannot become a required code symbol just because it contains a dot."""
    from types import SimpleNamespace

    query = "첨부한 billing_rules.py의 계산 방식을 설명해줘."
    state = build_graph_state_input(user_input=query, retriever=SimpleNamespace(), upload_files=[{
        "file_id": "billing", "name": "billing_rules.py", "size_bytes": 10,
        "content_hash": "sha256:" + "a" * 64, "source_uri": "upload://session/billing",
    }])
    planner = _RepairingPlanner(
        _upload_plan({"symbols": ["billing_rules.py"], "match": "symbol", "file_ids": ["billing"]}, query=query),
        _upload_plan({"file_ids": ["billing"]}, query=query),
    )
    result = make_planner_node(planner, False)(state)
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
    state = build_graph_state_input(user_input=query, retriever=SimpleNamespace())
    planner = _RepairingPlanner(
        _upload_plan({"symbols": ["calculate_total"], "match": "definition", "aspects": [condition]}, query=query),
        _upload_plan({"symbols": ["calculate_total"], "match": "definition"}, query=query),
    )
    result = make_planner_node(planner, False)(state)
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
    state = build_graph_state_input(user_input=query, retriever=SimpleNamespace())
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


def test_planner_cannot_widen_an_unavailable_file_scope_to_active_uploads():
    """An unavailable file stays scoped through planning and execution without borrowing active files."""
    from types import SimpleNamespace
    from src.infra.tools.local_rag.tool import build_upload_search_tool
    from src.runtime.nodes.retrieval import make_retrieve_dispatch_node

    class ScopedVectorStore:
        def similarity_search_with_score(self, query, *, k, file_ids):
            assert file_ids == ["missing"]
            return []

    task = RetrievalTask(route="upload", query="explain setup", k=4, requirement={"file_ids": ["missing"]})
    retriever = SimpleNamespace(vectorstore=ScopedVectorStore())
    state = build_graph_state_input(user_input=task.query, retriever=retriever)

    state.update(make_planner_node(_CapturePlannerLLM(PlannerOutput(use_retrieval=True, tasks=[task])), False)(state))
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))

    assert state["planner"].output.tasks == [task]
    assert state["planner"].guided_followup is None
    assert state["retrieval"].hit_log == []
    diagnostic = state["debug"].retrieval_diagnostics[0]
    assert diagnostic.status == "no_result"
    assert diagnostic.missing_requirements == ["file:missing"]


def test_retrieval_retry_keeps_requested_files_when_query_changes():
    """A rewritten search still queries and reports missing evidence for every requested file."""
    from types import SimpleNamespace
    from src.infra.tools.local_rag.tool import build_upload_search_tool
    from src.runtime.nodes.retrieval import make_retrieve_dispatch_node

    searches = []

    class ScopedVectorStore:
        def similarity_search_with_score(self, query, *, k, file_ids):
            searches.append({"query": query, "k": k, "file_ids": file_ids})
            return []

    first = RetrievalTask(route="upload", query="explain setup", k=4, requirement={"file_ids": ["alpha", "beta"]})
    uploads = tuple({"file_id": file_id, "name": f"{file_id}.py", "size_bytes": 10,
                     "content_hash": "sha256:" + "a" * 64, "source_uri": f"upload://session/{file_id}"}
                    for file_id in ["alpha", "beta"])
    state = build_graph_state_input(user_input=first.query, retriever=SimpleNamespace(vectorstore=ScopedVectorStore()), upload_files=uploads,
                                    request_contract=RequestContract(),
                                    retry={"attempt": 1, "original_tasks": [first.model_dump()], "failed_requirement_ids": [first.requirement_id]})

    rewrites = RetrievalRewriteOutput(rewrites=[{"requirement_id": first.requirement_id, "query": "setup usage", "k": 5}])
    state.update(make_planner_node(None, verbose=False, llm_planner_retry=_CapturePlannerLLM(rewrites))(state))
    state.update(make_retrieve_dispatch_node(None, build_upload_search_tool(), False)(state))

    assert searches
    assert all(search["query"] == "setup usage" and search["k"] == 5 for search in searches)
    assert all(set(search["file_ids"]) <= {"alpha", "beta"} for search in searches)
    assert {file_id for search in searches for file_id in search["file_ids"]} == {"alpha", "beta"}
    assert state["retrieval"].hit_log == []
    diagnostic, = state["debug"].retrieval_diagnostics
    assert diagnostic.requirement_id == first.requirement_id
    assert diagnostic.status == "no_result"
    assert set(diagnostic.missing_requirements) == {"file:alpha", "file:beta"}
