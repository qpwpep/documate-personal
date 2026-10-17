

from src.core.contracts.boundary.graph import build_graph_state_input
from src.core.planner_schema import PlannerOutput, RetrievalRewriteOutput, RetrievalTask
from src.core.request_contracts import RequestContract
from src.runtime.nodes.planner import make_planner_node
from .helpers import _CapturePlannerLLM


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
